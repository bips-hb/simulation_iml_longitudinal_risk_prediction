#.......................................................................................
# Simulation study: strengths and weaknesses of WindowSHAP (stationary, sliding, dynamic)
# One dataset per pattern, each with 10 visits and 3 codes
#
#   W1 single cell         code 1 at visit 10
#   W2 episode             code 1 at visits 6, 7 AND 8 (all three needed)
#   W3 episode across      code 1 at visits 4, 5 AND 6 (all three needed; crosses the
#                          boundary of the stationary windows 1-5 and 6-10)
#   W4 opposite signs      code 1 at visit 8 raises, at visit 10 lowers
#   W5 AND across codes    codes 1, 2, 3 together at the same visit
#
# Three versions of WindowSHAP, each a different game with its own ground truth:
#   Stationary: fixed windows of 5 visits (1-5, 6-10); one window of one code = one player
#   Sliding:    a window of 5 visits slides over the visits (step 1); at every position the
#               players are "inside the window" and "outside the window" (per code); a visit
#               gets the average of the inside values of all positions that cover it
#   Dynamic:    starts with one window per code (all 10 visits) and splits windows with
#               |value| > delta in half, up to n_w windows per code; one person at a time
#
# Models: true logit, GRU, TabPFN. Ground truth: shapiq.
# All five patterns are run one after the other, and the whole simulation is repeated
# n_reps times with new data. Heatmaps: last repetition.
#
# windowshap.py from https://github.com/vsubbian/WindowSHAP must be in the same folder,
# with two small changes (see the instructions that came with this script):
#   1. shap_values = np.moveaxis(np.array(shap_values), -1, 0)   (newer shap versions)
#   2. in DynamicWindowSHAP: start_ind = ... split_points[i][j-1] + 1, end_ind = split_points[i][j] + 1
#.......................................................................................

# New tmux session
# tmux new -s windowshap

# Reattach to the session
# tmux attach -t windowshap

# Run code
# source eICU_env/bin/activate
# CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -u simulation_WindowSHAP.py 2>&1 | tee run_log_windowshap.txt

# Leave session
# Press Crtl + b, then d

# Follow the log
# tail -f /home/hoepler/eICU_Real_World_Example/run_log_windowshap.txt


#.......................................................................................
# 1. Import packages
#.......................................................................................

import os
os.environ["OMP_NUM_THREADS"] = "1"        # one thread per process (WindowSHAP runs on many processes)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use only the first GPU (cuda:0)

import time
import numpy as np
import pandas as pd
import shapiq
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tabpfn import TabPFNClassifier
from windowshap.windowshap import StationaryWindowSHAP, SlidingWindowSHAP, DynamicWindowSHAP

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

window_len = 5        # window length (stationary and sliding), in visits
n_windows = T // window_len   # number of stationary windows per code (here 2)
stride = 1            # step size of the sliding window
n_positions = (T - window_len) // stride + 1   # number of sliding window positions (here 6)
delta = 0.1           # dynamic: split a window if |its Shapley value| > delta
n_w = 4               # dynamic: maximum number of windows per code
nsamples = 2048       # KernelSHAP budget inside sliding and dynamic WindowSHAP

n_reps = 2           # repetitions of the whole simulation
n_explain = 100       # number of test persons to explain (stationary and sliding)
n_explain_dynamic = 20  # number of test persons to explain with dynamic WindowSHAP (slow)
n_jobs = 32           # CPU cores
n_jobs_tabpfn = 1     # 1 if TabPFN runs on a GPU

versions = ["stationary", "sliding", "dynamic"]

# Note: Python counts from 0, so code 1 = index 0 and visit 1 = index 0

#.......................................................................................
# The true model:
# true_model takes one row of 30 columns per person (column = visit * 3 + code) and
# returns the true logit of the current pattern
# Z is a 3-dimensional array: Z[person, visit, code]
#.......................................................................................

# Note that "term" is always one number per person: 0 or 1; for W4 it also can be -1
def true_model(X_flat):
    Z = X_flat.reshape(len(X_flat), T, D).astype(int)
    if pattern == "W1":
        term = Z[:, 9, 0]                                          # for every person: 1 if code 1 at visit 10
    if pattern == "W2":
        term = Z[:, 5:8, 0].all(axis=1)                            # for every person: 1 if code 1 at visits 6, 7 AND 8
    if pattern == "W3":
        term = Z[:, 3:6, 0].all(axis=1)                            # for every person: 1 if code 1 at visits 4, 5 AND 6
    if pattern == "W4":
        term = Z[:, 7, 0] - Z[:, 9, 0]                             # for every person: +1 for code 1 at visit 8, -1 for code 1 at visit 10
    if pattern == "W5":
        term = (Z[:, :, 0] & Z[:, :, 1] & Z[:, :, 2]).any(axis=1)  # for every person: 1 if codes 1, 2 and 3 together at at least one visit
    return beta_0 + beta * term

#.......................................................................................
# Dynamic WindowSHAP explains one person at a time and chooses its windows itself.
# This small function runs it for one person and returns the Shapley values per visit
# AND the chosen windows (split points), which are needed for the ground truth.
#.......................................................................................

def run_dynamic(model, background, x_one):
    explainer = DynamicWindowSHAP(model, delta, n_w, background, x_one)
    values = explainer.shap_values(nsamples_in_loop=nsamples, nsamples_final=nsamples)
    return values[0], explainer.split_points

#.......................................................................................
# Ground truth for a game with grouped players.
# groups: list of True/False tables (visits x codes), one per player.
# A coalition = the cells of these players from person x, all other cells from the
# background persons; its value = true logit averaged over the background persons.
# shapiq computes the exact Shapley values of all players.
#.......................................................................................

def exact_shapley(x, background, groups):
    def value_function(coalitions):
        values = []
        for coalition in coalitions:
            Z = background.copy()
            for j in range(len(groups)):
                if coalition[j]:
                    Z[:, groups[j]] = x[groups[j]]
            values.append(true_model(Z.reshape(len(Z), T * D)).mean())
        return np.array(values)

    sv = shapiq.ExactComputer(value_function, n_players=len(groups))(index="SV", order=1)
    return np.array([sv[(j,)] for j in range(len(groups))])

#.......................................................................................
# The players of stationary WindowSHAP: one window of one code = one player
# Same order as in WindowSHAP: window 1 code 1, window 1 code 2, ..., window 2 code 3
#.......................................................................................

groups_stationary = []
for w in range(n_windows):
    for d in range(D):
        cells = np.zeros((T, D), dtype=bool)
        cells[w * window_len:(w + 1) * window_len, d] = True
        groups_stationary.append(cells)

all_results = []                          # key numbers of all repetitions, patterns and versions
examples = {version: {} for version in versions}   # example person per version and pattern

# Repeat the whole simulation n_reps times; everything indented below runs in every repetition
for rep in range(n_reps):

    # New random numbers in every repetition, so every repetition gets new data
    rng = np.random.default_rng(seed + rep)

    for pattern in ["W1", "W2", "W3", "W4", "W5"]:
        print(f"\n----- Repetition {rep + 1} of {n_reps}, pattern {pattern} -----")

        #...................................................................................
        # 3. Generate the data and insert the pattern
        # X[i, t, d] = 1 if person i has code d at visit t
        #...................................................................................

        X = rng.binomial(n=1, p=p_noise, size=(n, T, D))

        if pattern == "W1":
            # code 1 at visit 10
            X[rng.random(n) < p_pattern, 9, 0] = 1

        if pattern == "W2":
            # code 1 at visits 6, 7 and 8
            X[rng.random(n) < p_pattern, 5:8, 0] = 1

        if pattern == "W3":
            # code 1 at visits 4, 5 and 6
            X[rng.random(n) < p_pattern, 3:6, 0] = 1

        if pattern == "W4":
            # code 1 at visit 8 and code 1 at visit 10, independently
            X[rng.random(n) < p_pattern, 7, 0] = 1
            X[rng.random(n) < p_pattern, 9, 0] = 1

        if pattern == "W5":
            # codes 1, 2, 3 together at one random visit
            for i in np.where(rng.random(n) < p_pattern)[0]:
                X[i, rng.integers(0, T), :] = 1

        #...................................................................................
        # 4. The outcome
        #...................................................................................

        logit = true_model(X.reshape(n, T * D))
        y = rng.binomial(n=1, p=1 / (1 + np.exp(-logit)))
        print(f"Prevalence of the outcome: {y.mean():.3f}")

        #...................................................................................
        # 5. Split the data, background persons, persons to explain
        # Persons 1-700 training, 701-800 validation (GRU), 801-1000 test
        # Background: training persons without the pattern
        # Explained: the example person (complete pattern, fewest other codes) comes first,
        # so that it is also among the persons explained with dynamic WindowSHAP
        #...................................................................................

        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]

        if pattern == "W4":
            no_pattern = np.where((X_train[:, 7, 0] == 0) & (X_train[:, 9, 0] == 0))[0]
        else:
            no_pattern = np.where(logit[:800] == beta_0)[0]
        background = X_train[rng.choice(no_pattern, size=min(100, len(no_pattern)), replace=False)].astype(float)
        print(f"Background: {len(background)} persons")

        # complete: True/False per test person, True if the person has the full pattern
        if pattern == "W1":
            complete = X_test[:, 9, 0] == 1
        if pattern == "W2":
            complete = X_test[:, 5:8, 0].all(axis=1)
        if pattern == "W3":
            complete = X_test[:, 3:6, 0].all(axis=1)
        if pattern == "W4":
            complete = (X_test[:, 7, 0] == 1) & (X_test[:, 9, 0] == 1)
        if pattern == "W5":
            complete = (X_test[:, :, 0] & X_test[:, :, 1] & X_test[:, :, 2]).sum(axis=1) == 1

        # Lowest score = complete pattern and fewest extra codes; this person is row 0
        example = np.argmin(1000 * ~complete + X_test.sum(axis=(1, 2)))
        others = [i for i in range(200) if i != example]
        X_explain = X_test[[example] + others[:n_explain - 1]].astype(float)

        #...................................................................................
        # 6. Fit the GRU with early stopping on the validation persons
        # (same as in the KernelSHAP script)
        #...................................................................................

        X_fit = torch.tensor(X_train[:700], dtype=torch.float32)
        y_fit = torch.tensor(y_train[:700], dtype=torch.float32)
        X_val = torch.tensor(X_train[700:], dtype=torch.float32)
        y_val = torch.tensor(y_train[700:], dtype=torch.float32)
        loss_function = torch.nn.BCEWithLogitsLoss()

        torch.manual_seed(1)
        gru_layer = torch.nn.GRU(input_size=D, hidden_size=16, batch_first=True)
        out_layer = torch.nn.Linear(16, 1)
        optimizer = torch.optim.Adam(list(gru_layer.parameters()) + list(out_layer.parameters()),
                                     lr=0.01, weight_decay=0.001)

        best_val_loss = np.inf
        for epoch in range(500):
            optimizer.zero_grad()
            loss = loss_function(out_layer(gru_layer(X_fit)[1][0]).squeeze(1), y_fit)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_loss = loss_function(out_layer(gru_layer(X_val)[1][0]).squeeze(1), y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_gru = {k: v.clone() for k, v in gru_layer.state_dict().items()}
                best_out = {k: v.clone() for k, v in out_layer.state_dict().items()}

        gru_layer.load_state_dict(best_gru)
        out_layer.load_state_dict(best_out)

        with torch.no_grad():
            test_logit = out_layer(gru_layer(torch.tensor(X_test, dtype=torch.float32))[1][0]).squeeze(1).numpy()
        print(f"GRU: test AUC {roc_auc_score(y_test, test_logit):.3f}")
        print(f"True logit: test AUC {roc_auc_score(y_test, logit[800:]):.3f}")

        #...................................................................................
        # 7. Fit TabPFN (one row of 30 columns per person)
        #...................................................................................

        tabpfn = TabPFNClassifier(random_state=1)
        tabpfn.fit(X_train.reshape(800, T * D), y_train)
        print(f"TabPFN: test AUC {roc_auc_score(y_test, tabpfn.predict_proba(X_test.reshape(200, T * D))[:, 1]):.3f}")

        #...................................................................................
        # 8. The models as WindowSHAP needs them
        # WindowSHAP needs a model object with a .predict() method that takes the data as
        # persons x visits x codes and returns one logit per person as a column (persons x 1).
        # SimpleNamespace(predict=...) makes such an object from a plain function.
        #...................................................................................

        def true_model_3d(X3):
            return true_model(X3.reshape(len(X3), T * D))[:, None]

        def gru_model_3d(X3):
            x = torch.tensor(X3, dtype=torch.float32)
            with torch.no_grad():
                return out_layer(gru_layer(x)[1][0]).numpy()

        def tabpfn_model_3d(X3):
            p = np.clip(tabpfn.predict_proba(X3.reshape(len(X3), T * D))[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))[:, None]

        models = {"true logit": SimpleNamespace(predict=true_model_3d),
                  "GRU": SimpleNamespace(predict=gru_model_3d),
                  "TabPFN": SimpleNamespace(predict=tabpfn_model_3d)}

        # values[version][source]: Shapley values per visit, shape persons x visits x codes
        values = {version: {} for version in versions}

        # seconds[version][model]: run time per explained person (wall-clock). Note: TabPFN
        # runs with n_jobs_tabpfn processes (GPU), the other models with n_jobs CPU cores.
        seconds = {version: {} for version in versions}

        #...................................................................................
        # 9. Stationary WindowSHAP
        # Ground truth: exact Shapley values of the 6 window players (2 windows x 3 codes);
        # each window value is spread equally over its 5 visits, as WindowSHAP reports it.
        # WindowSHAP evaluates all 64 coalitions, so it has no sampling error.
        #...................................................................................

        truth = np.zeros((n_explain, T, D))
        for e in range(n_explain):
            sv = exact_shapley(X_explain[e], background, groups_stationary)
            for j in range(len(groups_stationary)):
                truth[e][groups_stationary[j]] = sv[j] / window_len
        values["stationary"]["ground truth"] = truth

        for name, model in models.items():
            start_time = time.time()                     # start the stopwatch
            jobs = n_jobs_tabpfn if name == "TabPFN" else n_jobs
            chunks = np.array_split(X_explain, jobs)
            results = Parallel(n_jobs=jobs)(
                delayed(StationaryWindowSHAP(model, window_len, background, chunk).shap_values)()
                for chunk in chunks)
            values["stationary"][name] = np.concatenate(results)
            seconds["stationary"][name] = (time.time() - start_time) / n_explain
        print("Stationary WindowSHAP: done", {k: round(v, 2) for k, v in seconds["stationary"].items()})

        #...................................................................................
        # 10. Sliding WindowSHAP
        # Ground truth: for every window position, the exact Shapley values of the players
        # "inside the window" and "outside the window" (per code, so 6 players); every
        # visit gets the average of the inside values of all positions that cover it.
        #...................................................................................

        truth = np.zeros((n_explain, T, D))
        for e in range(n_explain):
            inside_values = np.full((n_positions, T, D), np.nan)   # inside value per position and visit
            for k in range(n_positions):
                start = k * stride
                inside = np.zeros((T, D), dtype=bool)
                inside[start:start + window_len, :] = True
                # Players in the order of WindowSHAP: inside code 1-3, then outside code 1-3
                groups = []
                for part in [inside, ~inside]:
                    for d in range(D):
                        cells = np.zeros((T, D), dtype=bool)
                        cells[:, d] = part[:, d]
                        groups.append(cells)
                sv = exact_shapley(X_explain[e], background, groups)
                inside_values[k, start:start + window_len, :] = sv[:D]   # inside values of codes 1-3
            truth[e] = np.nanmean(inside_values, axis=0)
        values["sliding"]["ground truth"] = truth

        for name, model in models.items():
            start_time = time.time()                     # start the stopwatch
            jobs = n_jobs_tabpfn if name == "TabPFN" else n_jobs
            chunks = np.array_split(X_explain, jobs)
            results = Parallel(n_jobs=jobs)(
                delayed(SlidingWindowSHAP(model, stride, window_len, background, chunk).shap_values)(nsamples=nsamples)
                for chunk in chunks)
            values["sliding"][name] = np.concatenate(results)
            seconds["sliding"][name] = (time.time() - start_time) / n_explain
        print("Sliding WindowSHAP: done", {k: round(v, 2) for k, v in seconds["sliding"].items()})

        #...................................................................................
        # 11. Dynamic WindowSHAP (only the first n_explain_dynamic persons, one at a time)
        # The windows are chosen by the method itself. Ground truth: exact Shapley values on
        # the windows that dynamic WindowSHAP chooses for the TRUE logit; each window value
        # is spread equally over its visits.
        #...................................................................................

        X_dynamic = X_explain[:n_explain_dynamic]
        split_points_true = None

        for name, model in models.items():
            start_time = time.time()                     # start the stopwatch
            jobs = n_jobs_tabpfn if name == "TabPFN" else n_jobs
            results = Parallel(n_jobs=jobs)(
                delayed(run_dynamic)(model, background, X_dynamic[e:e + 1])
                for e in range(n_explain_dynamic))
            values["dynamic"][name] = np.array([r[0] for r in results])
            seconds["dynamic"][name] = (time.time() - start_time) / n_explain_dynamic
            if name == "true logit":
                split_points_true = [r[1] for r in results]
        print("Dynamic WindowSHAP: done", {k: round(v, 2) for k, v in seconds["dynamic"].items()})

        truth = np.zeros((n_explain_dynamic, T, D))
        for e in range(n_explain_dynamic):
            # Players: per code, the windows chosen for the true logit. A visit t belongs to
            # window j if split point j-1 < t <= split point j (as in WindowSHAP)
            groups = []
            for d in range(D):
                window_of_visit = np.searchsorted(split_points_true[e][d], np.arange(T))
                for j in range(len(split_points_true[e][d])):
                    cells = np.zeros((T, D), dtype=bool)
                    cells[window_of_visit == j, d] = True
                    groups.append(cells)
            sv = exact_shapley(X_dynamic[e], background, groups)
            for j in range(len(groups)):
                truth[e][groups[j]] = sv[j] / groups[j].sum()
        values["dynamic"]["ground truth"] = truth

        #...................................................................................
        # 12. Example person (row 0) for the heatmaps; overwritten until the last repetition
        #...................................................................................

        for version in versions:
            examples[version][pattern] = [("Data", X_explain[0])] + \
                [(name if name == "ground truth" else f"WindowSHAP: {name}", values[version][name][0])
                 for name in ["ground truth", "true logit", "GRU", "TabPFN"]]

        #...................................................................................
        # 13. Key numbers per visit, for the ground truth and each model, for every version
        # Mean Shapley value per visit over the explained persons with the complete pattern:
        #   on the visits of the pattern, and on nearby visits of the same code without
        #   the code (credit that is spread there by the windows)
        #...................................................................................

        for version in versions:
            truth = values[version]["ground truth"]

            for name in ["ground truth", "true logit", "GRU", "TabPFN"]:
                v = values[version][name]
                Xv = X_explain[:len(v)]          # the persons explained in this version

                if pattern == "W1":
                    has = Xv[:, 9, 0] == 1
                    key = {"visit 10": v[has, 9, 0].mean(),
                           "visits 6-9": v[has, 5:9, 0].mean()}
                if pattern == "W2":
                    has = Xv[:, 5:8, 0].all(axis=1)
                    key = {"visits 6-8": v[has, 5:8, 0].mean(),
                           "visits 9-10": v[has, 8:10, 0].mean()}
                if pattern == "W3":
                    has = Xv[:, 3:6, 0].all(axis=1)
                    key = {"visits 4-5": v[has, 3:5, 0].mean(),
                           "visit 6": v[has, 5, 0].mean(),
                           "visits 1-3, 7-10": np.concatenate([v[has, 0:3, 0], v[has, 6:10, 0]], axis=1).mean()}
                if pattern == "W4":
                    has = (Xv[:, 7, 0] == 1) & (Xv[:, 9, 0] == 1)
                    key = {"visit 8 (both)": v[has, 7, 0].mean(),
                           "visit 10 (both)": v[has, 9, 0].mean()}
                if pattern == "W5":
                    and_visit = (Xv[:, :, 0] * Xv[:, :, 1] * Xv[:, :, 2]) == 1
                    key = {"AND cells": v[and_visit].mean()}

                # Errors against the ground truth of this version, and the run time
                # (not for the ground truth)
                if name != "ground truth":
                    error = np.abs(v - truth)
                    truth_zero = np.abs(truth) < 1e-9
                    key["MAE truth != 0"] = error[~truth_zero].mean()
                    key["MAE truth = 0"] = error[truth_zero].mean()
                    key["seconds per person"] = seconds[version][name]

                for key_number, value in key.items():
                    all_results.append({"rep": rep, "version": version, "pattern": pattern,
                                        "key number": key_number, "source": name, "value": value})

#.......................................................................................
# 14. Results: one table per version, mean (SD) over the repetitions
# (no longer indented: runs once, after the loops)
#.......................................................................................

all_results = pd.DataFrame(all_results)
all_results.to_csv("results_windowshap.csv", index=False)
pd.set_option("display.width", 200)

for version in versions:
    r = all_results[all_results["version"] == version]
    means = r.groupby(["pattern", "key number", "source"], sort=False)["value"].mean().unstack()
    sds = r.groupby(["pattern", "key number", "source"], sort=False)["value"].std().unstack()

    # "mean (SD)"; SD 0 if there is only one repetition; "-" where there is no value
    table = means.round(2).astype(str) + " (" + sds.fillna(0).round(2).astype(str) + ")"
    table = table.where(means.notna(), "-")
    table = table[["ground truth", "true logit", "GRU", "TabPFN"]]

    print(f"\n{version.capitalize()} WindowSHAP, {n_reps} repetitions: mean (SD) per visit")
    print(table.to_string())

#.......................................................................................
# 15. Heatmaps: one figure per version, one example person per pattern (last repetition)
# The dashed lines mark the stationary window boundary (between visit 5 and 6)
#.......................................................................................

for version in versions:
    limit = max(np.abs(examples[version][p][1][1]).max() for p in examples[version])
    fig, axes = plt.subplots(5, 5, figsize=(20, 9), layout="constrained")

    for r, pattern in enumerate(["W1", "W2", "W3", "W4", "W5"]):
        for c, (title, v) in enumerate(examples[version][pattern]):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(v.T, cmap="Greys", vmin=0, vmax=1.4)
                ax.set_title(f"{pattern}: data", fontsize=9)
            else:
                image = ax.imshow(v.T, cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_title(title, fontsize=9)
                for d in range(D):
                    for t in range(T):
                        if abs(v[t, d]) >= 0.05:
                            ax.text(t, d, f"{v[t, d]:.2f}", ha="center", va="center", fontsize=7,
                                    color="white" if abs(v[t, d]) > 0.6 * limit else "black")
            if version == "stationary":
                ax.axvline(window_len - 0.5, color="black", linestyle="--", linewidth=1)
            ax.set_xticks(range(T), range(1, T + 1))
            ax.set_yticks(range(D), [f"code {d + 1}" for d in range(D)])

    fig.colorbar(image, ax=axes, shrink=0.6, label="Shapley value per visit (logit)")
    fig.suptitle(f"{version.capitalize()} WindowSHAP", x=0.01, ha="left")
    plt.savefig(f"heatmaps_windowshap_{version}.png", dpi=150, bbox_inches="tight")
    plt.show()