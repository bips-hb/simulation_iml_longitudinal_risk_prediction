#.......................................................................................
# Simulation study: strengths and weaknesses of KernelSHAP
# One dataset per pattern, each with 10 visits and 3 codes
#
#   P1 single cell      code 1 at visit 10                              -> strength
#   P2 opposite signs   code 1 at visit 8 raises, at visit 10 lowers    -> strength
#   P3 AND interaction  codes 1, 2, 3 together at the same visit        -> weakness
#   P4 OR redundancy    code 1 at 1-3 of visits 1-3 (one is enough)     -> weakness
#   P5 coupled codes    code 1 always followed by code 2, only code 1 matters -> weakness
#
# Models: true logit, GRU, TabPFN. Ground truth: shapiq.
# All five patterns are run one after the other (steps 3-11 for each pattern), and the
# whole simulation is repeated n_reps times with new data. Heatmaps: last repetition.
#.......................................................................................

# New tmux session
# tmux new -s kernelshap

# Reattach to the session
# tmux attach -t kernelshap

# Run code
# source eICU_env/bin/activate
# CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -u simulation_KernelSHAP.py 2>&1 | tee run_log_kernelshap.txt

# Leave session
# Press Crtl + b, then d

# Follow the log
# tail -f /home/hoepler/eICU_Real_World_Example/run_log_kernelshap.txt

#.......................................................................................
# 1. Import packages
#.......................................................................................

import os
os.environ["OMP_NUM_THREADS"] = "1"     # one thread per process (KernelSHAP runs on many processes)

import time
import numpy as np
import pandas as pd
import shapiq
import shap
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tabpfn import TabPFNClassifier

#.......................................................................................
# 2. Settings
#.......................................................................................

n = 1000              # number of persons
T = 10                # number of visits
D = 3                 # number of binary codes
p_noise = 0.02        # probability of a code at a visit (background noise)
p_pattern = 0.3       # share of persons who get the pattern
seed = 43756

beta_0 = -2.0         # baseline log odds
beta = 1.5            # effect of the pattern

n_reps = 20           # repetitions of the whole simulation
n_explain = 100       # number of test persons to explain
nsamples = 1000       # KernelSHAP budget
n_jobs = 32           # CPU cores for KernelSHAP
n_jobs_tabpfn = 1     # 1 if TabPFN runs on a GPU

# Note: Python counts from 0, so code 1 = index 0 and visit 1 = index 0

#.......................................................................................
# The true model:
# true_model takes one row of 30 columns per person (column = visit * 3 + code), as
# KernelSHAP needs it, and returns the true logit of the current pattern
# Z is a 3-dimensional array: Z[person, visit, code]
# For example for P1: Z[:, 9, 0]
#             - person ":": all persons
#             - visit "9": visit 10 (remember Python counts from 0, so 0 = visit 1, ..., 9 = visit 10)
#             - code "0": code 1 (0 = code 1, 1 = code 2, 2 = code 3)
#.......................................................................................

# Note that "term" is always one number per person: 0 or 1; for P2 it also can be -1
def true_model(X_flat):
    Z = X_flat.reshape(len(X_flat), T, D).astype(int)
    if pattern == "P1":
        term = Z[:, 9, 0]                                          # for every person, the value of code 1 at visit 10
    if pattern == "P2":
        term = Z[:, 7, 0] - Z[:, 9, 0]                             # for every person: +1 for code 1 at visit 8, -1 for code 1 at visit 10
    if pattern == "P3":
        term = (Z[:, :, 0] & Z[:, :, 1] & Z[:, :, 2]).any(axis=1)  # for every person: 1 if codes 1, 2 and 3 together at at least one visit
    if pattern == "P4":
        term = Z[:, 0:3, 0].any(axis=1)                            # for every person: 1 if code 1 at at least one of visits 1, 2, 3
    if pattern == "P5":
        term = Z[:, :, 0].any(axis=1)                              # for every person: 1 if code 1 at at least one visit (code 2 has no effect)
    return beta_0 + beta * term

all_results = []    # key numbers of all repetitions and patterns
examples = {}       # example person of each pattern (last repetition), for the heatmaps

# Repeat the whole simulation n_reps times; everything indented below runs in every repetition
for rep in range(n_reps):

    # New random numbers in every repetition, so every repetition gets new data
    rng = np.random.default_rng(seed + rep)

    for pattern in ["P1", "P2", "P3", "P4", "P5"]:
        print(f"\n----- Repetition {rep + 1} of {n_reps}, pattern {pattern} -----")

        #...................................................................................
        # 3. Generate the data and insert the pattern
        # X[i, t, d] = 1 if person i has code d at visit t
        # pattern_cells: the cells the pattern depends on (for the ground truth)
        #...................................................................................

        X = rng.binomial(n=1, p=p_noise, size=(n, T, D))
        pattern_cells = np.zeros((T, D), dtype=bool)

        if pattern == "P1":
            # code 1 at visit 10
            X[rng.random(n) < p_pattern, 9, 0] = 1
            pattern_cells[9, 0] = True

        if pattern == "P2":
            # code 1 at visit 8 and code 1 at visit 10, independently
            X[rng.random(n) < p_pattern, 7, 0] = 1
            X[rng.random(n) < p_pattern, 9, 0] = 1
            pattern_cells[7, 0] = True
            pattern_cells[9, 0] = True

        if pattern == "P3":
            # codes 1, 2, 3 together at one random visit
            for i in np.where(rng.random(n) < p_pattern)[0]:
                X[i, rng.integers(0, T), :] = 1
            pattern_cells[:, :] = True

        if pattern == "P4":
            # code 1 at 1, 2 or 3 of the visits 1-3
            for i in np.where(rng.random(n) < p_pattern)[0]:
                X[i, rng.choice(3, size=rng.integers(1, 4), replace=False), 0] = 1
            pattern_cells[0:3, 0] = True

        if pattern == "P5":
            # code 1 at a random visit 1-9; code 2 always at the visit after code 1
            for i in np.where(rng.random(n) < p_pattern)[0]:
                X[i, rng.integers(0, T - 1), 0] = 1
            X[:, :, 1] = 0
            X[:, 1:, 1] = X[:, :-1, 0]
            pattern_cells[:, 0] = True

        #...................................................................................
        # 4. The outcome
        #...................................................................................

        # true_model returns one logit per person: ß_0 + ß * term (so -1 without and -0.5 with it)
        logit = true_model(X.reshape(n, T * D)) # X.reshape(n, T * D) turns the data from 1000 x 10 x 3 into 1000 rows of 30 numbers, which is the form the models expects

        # Draw the outcome: logistic function turns the logit into a probability (logit(-2) -> prob. of 0.12; logit(-0.5) -> prob. of 0.38)
        y = rng.binomial(n=1, p=1 / (1 + np.exp(-logit)))
        print(f"Prevalence of the outcome: {y.mean():.3f}")

        #...................................................................................
        # 5. Split the data, background persons, persons to explain
        # Persons 1-700 training, 701-800 validation (GRU), 801-1000 test
        # Background: training persons without the pattern (true logit = beta_0, and for P2
        # no code 1 at visit 8 or 10)
        #...................................................................................

        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]

        if pattern == "P2":
            no_pattern = np.where((X_train[:, 7, 0] == 0) & (X_train[:, 9, 0] == 0))[0]
        else:
            no_pattern = np.where(logit[:800] == beta_0)[0]
        background = X_train[rng.choice(no_pattern, size=min(100, len(no_pattern)), replace=False)]
        print(f"Background: {len(background)} persons")

        X_explain = X_test[:n_explain]

        #...................................................................................
        # 6. Ground truth with shapiq
        # Exact Shapley values of the true logit, with the same players (30 cells) and the same
        # background persons as KernelSHAP. Per background person b, only the pattern cells
        # where x and b differ can change the prediction, so only they are players; averaged
        # over the background persons.
        #...................................................................................

        # Create an empty result table; one 10x3 table per explained person, all zeros
        shap_true = np.zeros((n_explain, T, D))

        # Loop over the explained persons; x is the current person's 10x3 table
        for e in range(n_explain):
            x = X_explain[e]

            # Loop over background persons; b is the current background person's 10x3 table
            for b in background:

                # "x != b" is true where the two persons differ; 
                # "& pattern_cells" keeps only cells the pattern depends on; 
                # "np.argwhere" gives the positions if these cells as(visit,code) pairs, these cells are the players, k is how many there are
                differ = np.argwhere((x != b) & pattern_cells)   
                k = len(differ)

                # If x and b don’t differ in any relevant cell, nothing can be explained for this pair, so skip to the next background person
                if k == 0:
                    continue

                # Tells shapiq what each coalition is worth. shapiq passes many coalitions at once, each a row of True/False, one entry per player (True = the player is “in”).
                # Value of each coalition: start from b, take the cells in the coalition from x
                def value_function(coalitions):
                    Z = np.repeat(b[None], len(coalitions), axis=0)
                    for j in range(k):
                        t, d = differ[j]
                        Z[coalitions[:, j].astype(bool), t, d] = x[t, d]
                    return true_model(Z.reshape(len(Z), T * D))

                # shapiq computes the exact Shapley values ("SV") of the k players by evaluating all 2ᵏ coalitions.
                sv = shapiq.ExactComputer(value_function, n_players=k)(index="SV", order=1)

                # Each player’s Shapley value sv[(j,)] is added to its cell, divided by the number of background persons. 
                # After the loop over all b, each cell holds the average over the background, which is exactly what KernelSHAP estimates.
                for j in range(k):
                    t, d = differ[j]
                    shap_true[e, t, d] += sv[(j,)] / len(background)

        # For every person, the Shapley values must add up to the person’s true logit minus the average true logit of the background (efficiency). 
        # np.allclose checks this for all explained persons at once and prints True if it holds.
        print("Ground truth adds up correctly:",
              np.allclose(shap_true.sum(axis=(1, 2)),
                          true_model(X_explain.reshape(n_explain, T * D)) - true_model(background.reshape(len(background), T * D)).mean()))

        #...................................................................................
        # 7. Fit the GRU with early stopping on the validation persons
        # GRU layer -> state after the last visit -> linear layer -> logit
        #...................................................................................

        # Convert the data to torch tensors (the format PyTorch works with), as decimal numbers.
        # X has the shape persons x visits x codes, here 700 x 10 x 3 and 100 x 10 x 3.
        X_fit = torch.tensor(X_train[:700], dtype=torch.float32)    # persons 1-700: used to train the GRU
        y_fit = torch.tensor(y_train[:700], dtype=torch.float32)    # their outcomes
        X_val = torch.tensor(X_train[700:], dtype=torch.float32)    # persons 701-800: validation, only used to decide when to stop
        y_val = torch.tensor(y_train[700:], dtype=torch.float32)    # their outcomes

        # Loss function for a binary outcome: binary cross-entropy. "WithLogits" means it takes
        # the logit (not the probability) as input and applies the logistic function itself.
        loss_function = torch.nn.BCEWithLogitsLoss()

        # Fix the random start values of the network weights, so the GRU is the same in every run
        torch.manual_seed(1)

        # The GRU layer: reads the visits one after another (3 codes per visit) and keeps a
        # summary of everything seen so far, the hidden state, a vector of 16 numbers.
        # batch_first=True: the data has the order persons x visits x codes.
        gru_layer = torch.nn.GRU(input_size=D, hidden_size=16, batch_first=True)

        # The output layer: turns the 16-number hidden state after the last visit into one
        # number, the logit of the outcome
        out_layer = torch.nn.Linear(16, 1)

        # The optimizer: Adam adjusts the weights of both layers to reduce the loss.
        # lr=0.01: learning rate, how big each adjustment step is.
        # weight_decay=0.001: a small penalty on large weights, against overfitting.
        optimizer = torch.optim.Adam(list(gru_layer.parameters()) + list(out_layer.parameters()),
                                     lr=0.01, weight_decay=0.001)

        # Early stopping: remember the lowest validation loss so far (start with infinity)
        best_val_loss = np.inf

        # Training: at most 500 epochs; in each epoch the GRU sees all 700 training persons once
        for epoch in range(500):

            # Reset the gradients from the previous epoch (PyTorch would otherwise add them up)
            optimizer.zero_grad()

            # Forward pass on the training persons:
            #   gru_layer(X_fit) returns two things: the hidden state after every visit [0] and
            #   the hidden state after the LAST visit [1], with shape 1 x 700 x 16.
            #   [1][0] takes the last hidden state as 700 x 16.
            #   out_layer(...) turns it into 700 x 1 logits; .squeeze(1) makes it 700 logits.
            # The loss compares these logits with the true outcomes y_fit.
            loss = loss_function(out_layer(gru_layer(X_fit)[1][0]).squeeze(1), y_fit)

            # Backward pass: compute how each weight should change to reduce the loss
            loss.backward()

            # Change the weights one step in that direction
            optimizer.step()

            # Loss on the validation persons, with the current weights.
            # torch.no_grad(): only predicting, no gradients needed (faster).
            # .item() turns the result into a plain Python number.
            with torch.no_grad():
                val_loss = loss_function(out_layer(gru_layer(X_val)[1][0]).squeeze(1), y_val).item()

            # If the validation loss is the lowest so far: remember it and save a copy of the
            # current weights of both layers (.clone() makes a real copy, not just a reference)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_gru = {k: v.clone() for k, v in gru_layer.state_dict().items()}
                best_out = {k: v.clone() for k, v in out_layer.state_dict().items()}

        # After training: go back to the weights from the epoch with the lowest validation loss.
        # This is the early stopping: later epochs, where the GRU started to overfit the
        # training persons, are discarded.
        gru_layer.load_state_dict(best_gru)
        out_layer.load_state_dict(best_out)

        # How good is the GRU? Predict the logits of the 200 test persons (without gradients)
        # and convert them to a numpy array
        with torch.no_grad():
            test_logit = out_layer(gru_layer(torch.tensor(X_test, dtype=torch.float32))[1][0]).squeeze(1).numpy()

        # AUC on the test persons: 0.5 = no better than chance, 1 = perfect ranking.
        # The AUC of the true logit is the best any model can reach with this data.
        print(f"GRU: test AUC {roc_auc_score(y_test, test_logit):.3f}")
        print(f"True logit: test AUC {roc_auc_score(y_test, logit[800:]):.3f}")

        #...................................................................................
        # 8. Fit TabPFN (one row of 30 columns per person)
        #...................................................................................

        # Create the TabPFN classifier. random_state=1 fixes its internal randomness
        # (e.g. how the columns are shuffled), so the results are the same in every run.
        tabpfn = TabPFNClassifier(random_state=1)

        # "Fit" TabPFN on all 800 training persons.
        # TabPFN is a table model: it needs one row per person, so each person's
        # 10 x 3 table (visits x codes) is flattened into one row of 30 columns
        # (column = visit * 3 + code). It doesn't know that the columns are visits in time.
        # Unlike the GRU, TabPFN does not train any weights here: it is a network that was
        # pre-trained on millions of simulated datasets, and .fit() only stores the training
        # data. When predicting, it reads the training data together with the new persons and
        # predicts from them in one pass ("in-context learning").
        # No validation persons are needed, because there is no training that could overfit,
        # so all 800 persons are used.
        tabpfn.fit(X_train.reshape(800, T * D), y_train)

        # How good is TabPFN? Predict the 200 test persons, also flattened to 30 columns.
        # predict_proba gives two columns per person: P(y = 0) and P(y = 1);
        # [:, 1] takes P(y = 1). The AUC is computed from these probabilities.
        print(f"TabPFN: test AUC {roc_auc_score(y_test, tabpfn.predict_proba(X_test.reshape(200, T * D))[:, 1]):.3f}")





        #...................................................................................
        # 9. KernelSHAP on all models
        # Each model function gets rows of 30 columns and returns a logit.
        # l1_reg=False: plain KernelSHAP (the shap default keeps only the 10 largest cells).
        # The persons are split into chunks that are explained on several cores at once.
        #...................................................................................

        # KernelSHAP only needs a function "rows of 30 numbers in -> one number per row out".
        # These two small functions wrap the GRU and TabPFN in that form, like true_model.

        # GRU as a function: turn the rows back into 10 x 3 tables (visits x codes), convert
        # them to a torch tensor, predict without gradients and return the logits as numpy
        def gru_model(X_flat):
            x = torch.tensor(X_flat.reshape(len(X_flat), T, D), dtype=torch.float32)
            with torch.no_grad():
                return out_layer(gru_layer(x)[1][0]).squeeze(1).numpy()

        # TabPFN as a function: TabPFN gives probabilities, but the other models give logits.
        # So P(y = 1) is turned into a logit with log(p / (1 - p)), so that all three
        # models are explained on the same (logit) scale.
        # np.clip keeps p between 0.000001 and 0.999999, so the log never becomes infinite.
        def tabpfn_model(X_flat):
            p = np.clip(tabpfn.predict_proba(X_flat)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        # The three models to explain, each with a name
        models = {"true logit": true_model, "GRU": gru_model, "TabPFN": tabpfn_model}

        # Empty dictionary for the KernelSHAP values: one entry per model
        shap_kernel = {}

        for name, model in models.items():

            # Start the stopwatch (time.time() = current time in seconds)
            start_time = time.time()

            # The explainer: the model function plus the background persons (flattened to
            # rows of 30). Switched-off cells are filled in from these background persons.
            explainer = shap.KernelExplainer(model, background.reshape(len(background), T * D))

            # How many processes to run at once: TabPFN runs on the GPU, so only one process
            # (several would fight over GPU memory); the other models use n_jobs CPU cores
            jobs = n_jobs_tabpfn if name == "TabPFN" else n_jobs

            # Split the persons to explain into as many chunks as there are processes,
            # e.g. 100 persons on 32 cores -> 32 chunks of 3-4 persons
            chunks = np.array_split(X_explain.reshape(n_explain, T * D), jobs)

            # Run KernelSHAP on all chunks at the same time, each chunk on its own core.
            # delayed(...)(...) means: "run this function with these arguments" in a process.
            # nsamples: number of coalitions per person; l1_reg=False: no feature selection;
            # silent=True: no progress bar
            results = Parallel(n_jobs=jobs)(
                delayed(explainer.shap_values)(chunk, nsamples=nsamples, l1_reg=False, silent=True)
                for chunk in chunks)

            # Put the chunks back together (in the original order) and turn each person's
            # 30 values back into a 10 x 3 table: shape persons x visits x codes
            shap_kernel[name] = np.concatenate(results).reshape(n_explain, T, D)

            # Stop the stopwatch: run time (wall-clock) divided by the number of explained
            # persons. Note: TabPFN runs with n_jobs_tabpfn processes (GPU), the other
            # models with n_jobs CPU cores, so report the numbers of processes with it.
            seconds = (time.time() - start_time) / n_explain
            all_results.append({"rep": rep, "pattern": pattern, "key number": "seconds per person",
                                "source": name, "value": seconds})
            print(f"KernelSHAP on {name}: done ({seconds:.2f} seconds per person)")

        #...................................................................................
        # 10. Example person for the heatmap
        # The person with the complete pattern and as few other codes as possible
        #...................................................................................

        # complete: True/False per explained person, True if the person has the full pattern
        if pattern == "P1":
            # code 1 at visit 10
            complete = X_explain[:, 9, 0] == 1
        if pattern == "P2":
            # code 1 at visit 8 AND at visit 10
            complete = (X_explain[:, 7, 0] == 1) & (X_explain[:, 9, 0] == 1)
        if pattern == "P3":
            # codes 1, 2, 3 together at exactly one visit (not at two visits)
            complete = (X_explain[:, :, 0] & X_explain[:, :, 1] & X_explain[:, :, 2]).sum(axis=1) == 1
        if pattern == "P4":
            # code 1 at all three of visits 1-3
            complete = X_explain[:, 0:3, 0].sum(axis=1) == 3
        if pattern == "P5":
            # code 1 exactly once and code 2 exactly once (so code 2 follows code 1 and
            # there is no extra code 1 from noise)
            complete = (X_explain[:, :, 0].sum(axis=1) == 1) & (X_explain[:, :, 1].sum(axis=1) == 1)

        # Choose the example person with a score (lower = better):
        #   ~complete is 1 if the pattern is NOT complete, so 1000 * ~complete puts these
        #   persons far behind; X_explain.sum(axis=(1, 2)) is the number of codes the person
        #   has, so among the complete persons the one with the fewest extra (noise) codes wins.
        # np.argmin gives the index of the person with the lowest score.
        e = np.argmin(1000 * ~complete + X_explain.sum(axis=(1, 2)))

        # Store the example person for the heatmap: their data, ground truth and the KernelSHAP
        # values of each model, each with a title for the plot
        examples[pattern] = [("Data", X_explain[e]), ("Ground truth", shap_true[e])] + \
                            [(f"KernelSHAP: {name}", shap_kernel[name][e]) for name in models]

        #...................................................................................
        # 11. Key numbers of the pattern, for the ground truth and each model
        # Mean Shapley value on the pattern cells over the explained persons who have the pattern
        #...................................................................................

        # All Shapley values to summarise: the ground truth plus KernelSHAP on each model.
        # The | joins two dictionaries into one.
        sources = {"ground truth": shap_true} | shap_kernel

        for name, values in sources.items():
            # values: Shapley values of all explained persons, shape persons x visits x codes

            if pattern == "P1":     # strength: full credit on one cell (expected 1.5)
                # values[:, 9, 0]: value of code 1 at visit 10 for every person;
                # [X_explain[:, 9, 0] == 1]: keep only the persons who have code 1 there;
                # .mean(): average over these persons
                all_results.append({"rep": rep, "pattern":"P1", "key number": "cell", "source": name,
                                    "value": values[:, 9, 0][X_explain[:, 9, 0] == 1].mean()})

            if pattern == "P2":     # strength: sign kept (expected +1.5 and -1.5)
                # Same as P1, once for code 1 at visit 8 and once for code 1 at visit 10
                all_results.append({"rep": rep, "pattern":"P2", "key number": "visit 8", "source": name,
                                    "value": values[:, 7, 0][X_explain[:, 7, 0] == 1].mean()})
                all_results.append({"rep": rep, "pattern":"P2", "key number": "visit 10", "source": name,
                                    "value": values[:, 9, 0][X_explain[:, 9, 0] == 1].mean()})

            if pattern == "P3":     # weakness: split in 3 (expected 0.5)
                # and_visit: True/False per person and visit, True where codes 1-3 occur together
                and_visit = (X_explain[:, :, 0] & X_explain[:, :, 1] & X_explain[:, :, 2]) == 1
                # values[and_visit]: the 3 code values at each of these visits;
                # .mean(): average value per AND cell
                all_results.append({"rep": rep, "pattern":"P3", "key number": "cell", "source": name,
                                    "value": values[and_visit].mean()})

            if pattern == "P4":     # weakness: diluted (expected 1.5, 0.75, 0.5)
                # n_occ: per person, how often code 1 occurs in visits 1-3 (0, 1, 2 or 3)
                n_occ = X_explain[:, 0:3, 0].sum(axis=1)
                # Separately for persons with 1, 2 and 3 occurrences:
                #   values[n_occ == m]: Shapley values of these persons;
                #   [:, 0:3, 0]: their values of code 1 at visits 1-3;
                #   [X_explain[n_occ == m][:, 0:3, 0] == 1]: only the cells where code 1 is present;
                #   .mean(): average value per occurrence
                for m in [1, 2, 3]:
                    all_results.append({"rep": rep, "pattern":"P4", "key number": f"{m} occ.", "source": name,
                                        "value": values[n_occ == m][:, 0:3, 0][X_explain[n_occ == m][:, 0:3, 0] == 1].mean()})

            if pattern == "P5":     # weakness: code 2 has no effect (expected share 0)
                # Total credit on all cells where code 1 is present, and on all cells where
                # code 2 is present, summed over all persons
                code_1 = values[:, :, 0][X_explain[:, :, 0] == 1].sum()
                code_2 = values[:, :, 1][X_explain[:, :, 1] == 1].sum()
                # Share of the credit that goes to code 2 (should be 0, since code 2 has no effect)
                all_results.append({"rep": rep, "pattern":"P5", "key number": "share code 2", "source": name,
                                    "value": code_2 / (code_1 + code_2)})

            # Errors against the ground truth (only for KernelSHAP), over all explained persons:
            #   error: absolute difference KernelSHAP - ground truth, for every cell of every person
            #   truth_zero: True for cells where the ground truth is 0 (cells without credit)
            #   MAE truth != 0: mean error on the cells where the ground truth gives credit
            #                   (the pattern cells): is the credit there correct?
            #   MAE truth = 0:  mean error on the cells where the ground truth is 0:
            #                   how much spurious credit lands on cells that don't matter?
            if name != "ground truth":
                error = np.abs(values - shap_true)
                truth_zero = np.abs(shap_true) < 1e-9
                all_results.append({"rep": rep, "pattern": pattern, "key number": "MAE truth != 0",
                                    "source": name, "value": error[~truth_zero].mean()})
                all_results.append({"rep": rep, "pattern": pattern, "key number": "MAE truth = 0",
                                    "source": name, "value": error[truth_zero].mean()})

#.......................................................................................
# 12. Results of all patterns: mean (SD) over the repetitions
# (no longer indented: runs once, after the loops over the repetitions and patterns)
#.......................................................................................

# Turn the list of results into a table (one row per repetition, pattern, key number and
# source) and save it as a csv file
all_results = pd.DataFrame(all_results)
all_results.to_csv("results_all_patterns.csv", index=False)

# Rearrange into wide tables: one row per pattern and key number, one column per source
# (ground truth, true logit, GRU, TabPFN). sort=False keeps the order P1, P2, ...
# means: mean over the repetitions; sds: standard deviation over the repetitions
means = all_results.pivot_table(index=["pattern", "key number"], columns="source",
                                values="value", aggfunc="mean", sort=False)
sds = all_results.pivot_table(index=["pattern", "key number"], columns="source",
                              values="value", aggfunc="std", sort=False)

# Write each cell as "mean (SD)", columns in a fixed order
table = means.round(2).astype(str) + " (" + sds.round(2).astype(str) + ")"
table = table[["ground truth", "true logit", "GRU", "TabPFN"]]

# Add a first column with the expected values (what the ground truth should show with a
# pattern-free background). The MAE rows have no expected value ("-").
expected = {("P1", "cell"): beta,
            ("P2", "visit 8"): beta, ("P2", "visit 10"): -beta,
            ("P3", "cell"): beta / 3,
            ("P4", "1 occ."): beta, ("P4", "2 occ."): beta / 2, ("P4", "3 occ."): beta / 3,
            ("P5", "share code 2"): 0}
table.insert(0, "expected", [expected.get(row, "-") for row in table.index])

# The ground truth has no MAE (it is the reference), so its MAE cells are empty: show "-"
table = table.replace("nan (nan)", "-")

# Print the table, wide enough that it doesn't wrap
pd.set_option("display.width", 200)
print(f"\nResults over {n_reps} repetitions: mean (SD)")
print(table.to_string())

#.......................................................................................
# 13. Heatmaps: one example person per pattern (visits on the x axis, codes on the y axis)
# (from the last repetition, because examples[pattern] is overwritten in every repetition)
#.......................................................................................

# A grid of 5 x 5 panels: one row per pattern; columns: data, ground truth and KernelSHAP
# on the true logit, GRU and TabPFN. layout="constrained" keeps the panels from overlapping.
fig, axes = plt.subplots(5, 5, figsize=(20, 9), layout="constrained")

for r, pattern in enumerate(["P1", "P2", "P3", "P4", "P5"]):         # r = row number
    for c, (title, values) in enumerate(examples[pattern]):          # c = column number
        ax = axes[r, c]                                              # the current panel

        # values is a 10 x 3 table (visits x codes); .T turns it so that the codes are the
        # rows (y axis) and the visits the columns (x axis)
        if c == 0:
            # First column: the data, grey = code present, white = code absent
            ax.imshow(values.T, cmap="Greys", vmin=0, vmax=1.4)
            ax.set_title(f"{pattern}: data", fontsize=9)
        else:
            # Other columns: Shapley values, red = raises the risk, blue = lowers it,
            # white = 0. The colour scale runs from -1.5 to +1.5 (= -beta to +beta) in all panels.
            image = ax.imshow(values.T, cmap="RdBu_r", vmin=-beta, vmax=beta)
            ax.set_title(title, fontsize=9)
            # Write the value into every cell where it is at least 0.05 away from 0
            for d in range(D):
                for t in range(T):
                    if abs(values[t, d]) >= 0.05:
                        ax.text(t, d, f"{values[t, d]:.2f}", ha="center", va="center", fontsize=7)

        # Axis labels: visits 1-10 on the x axis, code 1-3 on the y axis
        ax.set_xticks(range(T), range(1, T + 1))
        ax.set_yticks(range(D), [f"code {d + 1}" for d in range(D)])

# One shared colour bar for all Shapley-value panels, then save and show the figure
fig.colorbar(image, ax=axes, shrink=0.6, label="Shapley value (logit)")
plt.savefig("heatmaps_all_patterns.png", dpi=150, bbox_inches="tight")
plt.show()