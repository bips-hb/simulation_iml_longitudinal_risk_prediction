#.......................................................................................
# Simulation study: strengths and weaknesses of TimeSHAP (event level, feature level, pruning)
# One dataset per pattern, each with 20 visits and 3 codes
#
#   T1 single recent cell    code 1 at visit 20                                -> strength (all levels);
#                                                                                 pruning merges all older visits (fast)
#   T2 opposite signs        code 1 at visit 18 raises, at visit 20 lowers     -> event level: strength,
#                                                                                 feature level: weakness (cancel to 0)
#   T3 interaction in time   code 1 at visit 3 AND code 2 at visit 18          -> weakness (split in half)
#   T4 early opposite        code 1 at visit 1 raises, code 2 at visit 2 lowers -> weakness of pruning: together
#                                                                                 they cancel, are merged and hidden
#   T5 coupled codes         code 1 always followed by code 2, only code 1 matters -> weakness (marginal)
#
# TimeSHAP per person: 1) pruning: older visits whose combined Shapley value is below the
# tolerance are merged into one player; 2) event level: every remaining visit is a player
# (plus the merged older visits); 3) feature level: every code is a player (plus the merged
# older visits); 4) cell level: the cells of the important visits x important codes
# (|value| >= cell_threshold at event and feature level) are players on their own, the
# rest is grouped (other codes at an important visit, other visits of an important code,
# all other cells, the merged older visits).
# The ground truth plays exactly these games with the true model.
#
# Models: true logit, GRU, TabPFN. Ground truth: shapiq.
# All five patterns are run one after the other, and the whole simulation is repeated
# n_reps times with new data. Figure: last repetition.
#.......................................................................................

# tmux attach -t timeshap
# CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -u simulation_TimeSHAP.py 2>&1 | tee run_log_timeshap.txt
# tail -f /home/hoepler/eICU_Real_World_Example/run_log_timeshap.txt


#.......................................................................................
# 1. Import packages
#.......................................................................................

import os
os.environ["OMP_NUM_THREADS"] = "1"        # one thread per process (TimeSHAP runs on many processes)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use only the first GPU (cuda:0)

import io
import time
import contextlib
import numpy as np
import pandas as pd
import shapiq
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tabpfn import TabPFNClassifier
import shap.explainers._kernel     # needed for timeshap, see run_timeshap below

#.......................................................................................
# 2. Settings
#.......................................................................................

n = 1000              # number of persons
T = 20                # number of visits
D = 3                 # number of binary codes
p_noise = 0.02        # probability of a code at a visit (background noise)
p_pattern = 0.3       # share of persons who get the pattern
seed = 43756

beta_0 = -2.0         # baseline log odds
beta = 1.5            # effect of the pattern

tolerance = 0.1       # pruning: older visits are merged if their combined |Shapley value| <= tolerance
nsamples = 2048       # KernelSHAP budget inside TimeSHAP (event, feature and cell level)
cell_threshold = 0.1  # cell level: visits and codes with |Shapley value| >= cell_threshold
                      # are explained cell by cell (TimeSHAP's "threshold" option)

n_reps = 2            # repetitions of the whole simulation (test with 2, later 20)
n_explain = 100       # number of test persons to explain
n_jobs = 32           # CPU cores
n_jobs_tabpfn = 1     # 1 if TabPFN runs on a GPU

# Note: Python counts from 0, so code 1 = index 0 and visit 1 = index 0

#.......................................................................................
# The true model:
# true_model takes one row of 60 columns per person (column = visit * 3 + code) and
# returns the true logit of the current pattern
# Z is a 3-dimensional array: Z[person, visit, code]
#.......................................................................................

# Note that "term" is always one number per person: 0 or 1; for T2 and T4 it also can be -1
def true_model(X_flat):
    Z = X_flat.reshape(len(X_flat), T, D).astype(int)
    if pattern == "T1":
        term = Z[:, 19, 0]                                         # for every person: 1 if code 1 at visit 20
    if pattern == "T2":
        term = Z[:, 17, 0] - Z[:, 19, 0]                           # +1 for code 1 at visit 18, -1 for code 1 at visit 20
    if pattern == "T3":
        term = Z[:, 2, 0] & Z[:, 17, 1]                            # 1 if code 1 at visit 3 AND code 2 at visit 18
    if pattern == "T4":
        term = Z[:, 0, 0] - Z[:, 1, 1]                             # +1 for code 1 at visit 1, -1 for code 2 at visit 2
    if pattern == "T5":
        term = Z[:, :, 0].any(axis=1)                              # 1 if code 1 at at least one visit (code 2 has no effect)
    return beta_0 + beta * term

#.......................................................................................
# The players of TimeSHAP's cell level, as True/False tables (visits x codes).
# E: important visits, F: important codes, p: pruning index (as in timeshap's cell_level)
#   (t, d)                         one cell: visit t, code d (t in E, d in F)
#   ("other codes at visit", t)    the codes not in F at visit t (t in E)
#   ("other visits of code", d)    code d at the visits not in E (d in F)
#   "other cells"                  codes not in F at visits not in E
#   "pruned"                       the merged older visits 1..p
#.......................................................................................

def cell_groups(E, F, p):
    other_codes = [d for d in range(D) if d not in F]
    other_visits = [t for t in range(p, T) if t not in E]
    groups = {}
    for t in E:
        for d in F:
            groups[(t, d)] = np.zeros((T, D), dtype=bool)
            groups[(t, d)][t, d] = True
    if len(other_codes) > 0:
        for t in E:
            groups[("other codes at visit", t)] = np.zeros((T, D), dtype=bool)
            groups[("other codes at visit", t)][t, other_codes] = True
    if len(other_visits) > 0:
        for d in F:
            groups[("other visits of code", d)] = np.zeros((T, D), dtype=bool)
            groups[("other visits of code", d)][other_visits, d] = True
    if len(other_codes) > 0 and len(other_visits) > 0:
        groups["other cells"] = np.zeros((T, D), dtype=bool)
        groups["other cells"][np.ix_(other_visits, other_codes)] = True
    if p > 0:
        groups["pruned"] = np.zeros((T, D), dtype=bool)
        groups["pruned"][:p, :] = True
    return groups

#.......................................................................................
# TimeSHAP for one person: pruning, event level, feature level and cell level.
# Returns the pruning index p (visits 1..p are merged into one player; p = 0: no pruning),
# the event-level values (last visit first, then the merged visits), the feature-level
# values (codes 1-3, then the merged visits), the important visits E and codes F, and the
# cell-level values as a dictionary with the player names of cell_groups.
# The cell level only runs if at least one visit and one code are important
# (otherwise timeshap cannot run it); then cell = {}.
# TimeSHAP prints a lot (e.g. "Unable to prune sequence."), so its printing is switched off.
#
# timeshap was written for older shap versions (e.g. 0.42), which call their KernelSHAP class
# "Kernel"; newer versions call it "KernelExplainer". The first two lines give the class the
# old name if it is missing, so timeshap can be imported with old and new shap versions.
# It is done inside this function, because every parallel process imports timeshap again.
#.......................................................................................

def run_timeshap(model, background, x_one):
    if not hasattr(shap.explainers._kernel, "Kernel"):
        shap.explainers._kernel.Kernel = shap.explainers._kernel.KernelExplainer
    from timeshap.explainer import temp_coalition_pruning, event_level, feature_level, cell_level
    with contextlib.redirect_stdout(io.StringIO()):
        p = T + temp_coalition_pruning(model, x_one, background, tolerance)
        event_data = event_level(model, x_one, background, p, 42, nsamples)
        feature_data = feature_level(model, x_one, background, p, 42, nsamples)
        event = event_data["Shapley Value"].values.astype(float)
        feature = feature_data["Shapley Value"].values.astype(float)

        # Important visits and codes (event[0] is the last visit, so event[i] is visit T-1-i)
        E = sorted([T - 1 - i for i in range(T - p) if abs(event[i]) >= cell_threshold])
        F = [d for d in range(D) if abs(feature[d]) >= cell_threshold]

        cell = {}
        if len(E) > 0 and len(F) > 0:
            cell_data = cell_level(model, x_one, background, event_data, feature_data, 42, nsamples,
                                   {"threshold": cell_threshold}, p)
            # timeshap's names -> our names ("Event -1" = last visit, "Feature 0" = code 1)
            for event_name, feature_name, value in cell_data[["Event", "Feature", "Shapley Value"]].values:
                if event_name == "Pruned Events":
                    key = "pruned"
                elif event_name == "Other Events" and feature_name == "Other Features":
                    key = "other cells"
                elif event_name == "Other Events":
                    key = ("other visits of code", int(feature_name.split()[-1]))
                elif feature_name == "Other Features":
                    key = ("other codes at visit", T + int(event_name.split()[-1]))
                else:
                    key = (T + int(event_name.split()[-1]), int(feature_name.split()[-1]))
                cell[key] = float(value)
    return p, event, feature, E, F, cell

#.......................................................................................
# Ground truth for a game with grouped players (exact, with shapiq).
# groups: list of True/False tables (visits x codes), one per player.
# As in the KernelSHAP script, it is computed per background person b: only players that
# contain pattern cells where x and b differ can change the true logit, so only they are
# players (the others get 0); then averaged over the background persons.
#.......................................................................................

def exact_shapley(x, background, groups):
    sv_total = np.zeros(len(groups))
    for b in background:
        relevant = [j for j in range(len(groups)) if ((x != b) & pattern_cells & groups[j]).any()]
        if len(relevant) == 0:
            continue

        # Value of each coalition: start from b, take the cells of the players in the coalition from x
        def value_function(coalitions):
            Z = np.repeat(b[None], len(coalitions), axis=0)
            for i, j in enumerate(relevant):
                rows = coalitions[:, i].astype(bool)
                Z[rows] = np.where(groups[j][None], x[None], Z[rows])
            return true_model(Z.reshape(len(Z), T * D))

        sv = shapiq.ExactComputer(value_function, n_players=len(relevant))(index="SV", order=1)
        for i, j in enumerate(relevant):
            sv_total[j] += sv[(i,)] / len(background)
    return sv_total

all_results = []    # key numbers of all repetitions and patterns
examples = {}       # example person of each pattern (last repetition), for the figure

# Repeat the whole simulation n_reps times; everything indented below runs in every repetition
for rep in range(n_reps):

    # New random numbers in every repetition, so every repetition gets new data
    rng = np.random.default_rng(seed + rep)

    for pattern in ["T1", "T2", "T3", "T4", "T5"]:
        print(f"\n----- Repetition {rep + 1} of {n_reps}, pattern {pattern} -----")

        #...................................................................................
        # 3. Generate the data and insert the pattern
        # X[i, t, d] = 1 if person i has code d at visit t
        # pattern_cells: the cells the pattern depends on (for the ground truth)
        #...................................................................................

        X = rng.binomial(n=1, p=p_noise, size=(n, T, D))
        pattern_cells = np.zeros((T, D), dtype=bool)

        if pattern == "T1":
            # code 1 at visit 20
            X[rng.random(n) < p_pattern, 19, 0] = 1
            pattern_cells[19, 0] = True

        if pattern == "T2":
            # code 1 at visit 18 and code 1 at visit 20, independently
            X[rng.random(n) < p_pattern, 17, 0] = 1
            X[rng.random(n) < p_pattern, 19, 0] = 1
            pattern_cells[[17, 19], 0] = True

        if pattern == "T3":
            # code 1 at visit 3 and code 2 at visit 18, together
            gets = rng.random(n) < p_pattern
            X[gets, 2, 0] = 1
            X[gets, 17, 1] = 1
            pattern_cells[2, 0] = True
            pattern_cells[17, 1] = True

        if pattern == "T4":
            # code 1 at visit 1 and code 2 at visit 2, independently
            X[rng.random(n) < p_pattern, 0, 0] = 1
            X[rng.random(n) < p_pattern, 1, 1] = 1
            pattern_cells[0, 0] = True
            pattern_cells[1, 1] = True

        if pattern == "T5":
            # code 1 at a random visit 1-19; code 2 always at the visit after code 1
            for i in np.where(rng.random(n) < p_pattern)[0]:
                X[i, rng.integers(0, T - 1), 0] = 1
            X[:, :, 1] = 0
            X[:, 1:, 1] = X[:, :-1, 0]
            pattern_cells[:, 0] = True

        #...................................................................................
        # 4. The outcome
        #...................................................................................

        logit = true_model(X.reshape(n, T * D))
        y = rng.binomial(n=1, p=1 / (1 + np.exp(-logit)))
        print(f"Prevalence of the outcome: {y.mean():.3f}")

        #...................................................................................
        # 5. Split the data, background persons, persons to explain
        # Persons 1-700 training, 701-800 validation (GRU), 801-1000 test
        # Background: training persons without the pattern (no pattern cell present).
        # TimeSHAP's "baseline" can be a set of sequences, so it uses the same background
        # persons as KernelSHAP and WindowSHAP.
        # Explained: the example person (complete pattern, fewest other codes) comes first
        #...................................................................................

        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]

        if pattern == "T2":
            no_pattern = np.where((X_train[:, 17, 0] == 0) & (X_train[:, 19, 0] == 0))[0]
        elif pattern == "T4":
            no_pattern = np.where((X_train[:, 0, 0] == 0) & (X_train[:, 1, 1] == 0))[0]
        else:
            no_pattern = np.where(logit[:800] == beta_0)[0]
        background = X_train[rng.choice(no_pattern, size=min(100, len(no_pattern)), replace=False)].astype(float)
        print(f"Background: {len(background)} persons")

        # complete: True/False per test person, True if the person has the full pattern
        if pattern == "T1":
            complete = X_test[:, 19, 0] == 1
        if pattern == "T2":
            complete = (X_test[:, 17, 0] == 1) & (X_test[:, 19, 0] == 1)
        if pattern == "T3":
            complete = (X_test[:, 2, 0] == 1) & (X_test[:, 17, 1] == 1)
        if pattern == "T4":
            complete = (X_test[:, 0, 0] == 1) & (X_test[:, 1, 1] == 1)
        if pattern == "T5":
            complete = (X_test[:, :, 0].sum(axis=1) == 1) & (X_test[:, :, 1].sum(axis=1) == 1)

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
        # 7. Fit TabPFN (one row of 60 columns per person)
        #...................................................................................

        tabpfn = TabPFNClassifier(random_state=1)
        tabpfn.fit(X_train.reshape(800, T * D), y_train)
        print(f"TabPFN: test AUC {roc_auc_score(y_test, tabpfn.predict_proba(X_test.reshape(200, T * D))[:, 1]):.3f}")

        #...................................................................................
        # 8. The models as TimeSHAP needs them: data persons x visits x codes in,
        # one logit per person as a column (persons x 1) out
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

        models = {"true logit": true_model_3d, "GRU": gru_model_3d, "TabPFN": tabpfn_model_3d}

        #...................................................................................
        # 9. TimeSHAP on all models
        # For every person: pruning index, event-level and feature-level values.
        # Stored as:
        #   prune[source]:          pruning index per person (visits 1..p merged)
        #   events[source]:         value per visit (persons x visits); NaN for merged visits
        #   events_merged[source]:  value of the merged visits (0 if nothing merged)
        #   features[source]:       value per code (persons x codes)
        #   features_merged[source]: value of the merged visits at feature level
        #   cell_map[source]:       cell level as a map (persons x visits x codes): a cell on its
        #                           own gets its value, a group's value is spread equally over
        #                           its cells (all 0 if the cell level did not run)
        #   cells_alone[source]:    True for the cells that were players on their own
        #   cell_run[source]:       True if the cell level ran for the person
        #...................................................................................

        prune, events, events_merged, features, features_merged, seconds = {}, {}, {}, {}, {}, {}
        cell_map, cells_alone, cell_run = {}, {}, {}

        for name, model in models.items():
            start_time = time.time()
            jobs = n_jobs_tabpfn if name == "TabPFN" else n_jobs
            results = Parallel(n_jobs=jobs)(
                delayed(run_timeshap)(model, background, X_explain[e:e + 1])
                for e in range(n_explain))
            seconds[name] = (time.time() - start_time) / n_explain

            prune[name] = np.array([r[0] for r in results])
            events[name] = np.full((n_explain, T), np.nan)
            events_merged[name] = np.zeros(n_explain)
            features[name] = np.zeros((n_explain, D))
            features_merged[name] = np.zeros(n_explain)
            cell_map[name] = np.zeros((n_explain, T, D))
            cells_alone[name] = np.zeros((n_explain, T, D), dtype=bool)
            cell_run[name] = np.zeros(n_explain, dtype=bool)
            for e, (p, event, feature, E, F, cell) in enumerate(results):
                events[name][e, p:] = event[:T - p][::-1]        # TimeSHAP lists the last visit first
                features[name][e] = feature[:D]
                if p > 0:
                    events_merged[name][e] = event[T - p]
                    features_merged[name][e] = feature[D]
                if len(cell) > 0:
                    cell_run[name][e] = True
                    for key, g in cell_groups(E, F, p).items():
                        cell_map[name][e][g] += cell[key] / g.sum()
                        if not isinstance(key[0], str):     # a single cell (t, d)
                            cells_alone[name][e][g] = True
            print(f"TimeSHAP on {name}: done ({seconds[name]:.2f} seconds per person)")

        #...................................................................................
        # 10. Ground truth with shapiq: the same three steps with the true model
        # Pruning: for p = 19, 18, ..., 1: two players, "visits p+1..20" and "visits 1..p";
        # the first (largest) p where |value of visits 1..p| <= tolerance is the pruning index
        # (0 if there is none). Then the event and feature levels with the same pruning.
        #...................................................................................

        prune["ground truth"] = np.zeros(n_explain, dtype=int)
        events["ground truth"] = np.full((n_explain, T), np.nan)
        events_merged["ground truth"] = np.zeros(n_explain)
        features["ground truth"] = np.zeros((n_explain, D))
        features_merged["ground truth"] = np.zeros(n_explain)
        cell_map["ground truth"] = np.zeros((n_explain, T, D))
        cells_alone["ground truth"] = np.zeros((n_explain, T, D), dtype=bool)
        cell_run["ground truth"] = np.zeros(n_explain, dtype=bool)

        for e in range(n_explain):
            x = X_explain[e]

            # Pruning
            p = 0
            for seq_len in range(T - 1, 0, -1):
                older = np.zeros((T, D), dtype=bool)
                older[:seq_len, :] = True
                sv = exact_shapley(x, background, [~older, older])
                if abs(sv[1]) <= tolerance:
                    p = seq_len
                    break
            prune["ground truth"][e] = p
            merged = np.zeros((T, D), dtype=bool)
            merged[:p, :] = True

            # Event level: every visit from p+1 on is a player, plus the merged visits
            groups = []
            for t in range(p, T):
                cells = np.zeros((T, D), dtype=bool)
                cells[t, :] = True
                groups.append(cells)
            if p > 0:
                groups.append(merged)
            sv = exact_shapley(x, background, groups)
            events["ground truth"][e, p:] = sv[:T - p]
            if p > 0:
                events_merged["ground truth"][e] = sv[T - p]

            # Feature level: every code (at the visits from p+1 on) is a player, plus the merged visits
            groups = []
            for d in range(D):
                cells = np.zeros((T, D), dtype=bool)
                cells[p:, d] = True
                groups.append(cells)
            if p > 0:
                groups.append(merged)
            sv = exact_shapley(x, background, groups)
            features["ground truth"][e] = sv[:D]
            if p > 0:
                features_merged["ground truth"][e] = sv[D]

            # Cell level: important visits E and codes F (|value| >= cell_threshold), then the
            # same players as timeshap (see cell_groups); only if E and F are not empty
            E = [t for t in range(p, T) if abs(events["ground truth"][e, t]) >= cell_threshold]
            F = [d for d in range(D) if abs(features["ground truth"][e, d]) >= cell_threshold]
            if len(E) > 0 and len(F) > 0:
                cell_run["ground truth"][e] = True
                groups = cell_groups(E, F, p)
                sv = exact_shapley(x, background, list(groups.values()))
                for (key, g), value in zip(groups.items(), sv):
                    cell_map["ground truth"][e][g] += value / g.sum()
                    if not isinstance(key[0], str):     # a single cell (t, d)
                        cells_alone["ground truth"][e][g] = True

        #...................................................................................
        # 11. Example person (row 0) for the figure; overwritten until the last repetition
        #...................................................................................

        examples[pattern] = {"data": X_explain[0],
                             "events": {s: events[s][0] for s in events},
                             "events_merged": {s: events_merged[s][0] for s in events},
                             "features": {s: features[s][0] for s in events},
                             "features_merged": {s: features_merged[s][0] for s in events},
                             "prune": {s: prune[s][0] for s in events},
                             "cell_map": {s: cell_map[s][0] for s in events},
                             "cells_alone": {s: cells_alone[s][0] for s in events},
                             "cell_run": {s: cell_run[s][0] for s in events}}

        #...................................................................................
        # 12. Key numbers, for the ground truth and each model
        # Mean over the explained persons with the complete pattern. A visit that was merged
        # by pruning is not shown on its own, so its own value counts as 0.
        # Cell level: the value of the cell in the map (if the cell was in a group: its share
        # of the group's value; 0 if the cell level did not run).
        #...................................................................................

        for name in ["ground truth", "true logit", "GRU", "TabPFN"]:
            ev = np.nan_to_num(events[name])        # merged visits: 0 (not shown on their own)
            fe = features[name]
            ce = cell_map[name]
            Xv = X_explain

            if pattern == "T1":
                has = Xv[:, 19, 0] == 1
                key = {"event: visit 20": ev[has, 19].mean(),
                       "feature: code 1": fe[has, 0].mean(),
                       "cell: visit 20, code 1": ce[has, 19, 0].mean()}
            if pattern == "T2":
                has = (Xv[:, 17, 0] == 1) & (Xv[:, 19, 0] == 1)
                key = {"event: visit 18": ev[has, 17].mean(),
                       "event: visit 20": ev[has, 19].mean(),
                       "feature: code 1": fe[has, 0].mean(),
                       "cell: visit 18, code 1": ce[has, 17, 0].mean(),
                       "cell: visit 20, code 1": ce[has, 19, 0].mean()}
            if pattern == "T3":
                has = (Xv[:, 2, 0] == 1) & (Xv[:, 17, 1] == 1)
                key = {"event: visit 3": ev[has, 2].mean(),
                       "event: visit 18": ev[has, 17].mean(),
                       "feature: code 1": fe[has, 0].mean(),
                       "feature: code 2": fe[has, 1].mean(),
                       "cell: visit 3, code 1": ce[has, 2, 0].mean(),
                       "cell: visit 18, code 2": ce[has, 17, 1].mean()}
            if pattern == "T4":
                has = (Xv[:, 0, 0] == 1) & (Xv[:, 1, 1] == 1)
                key = {"visit 1 shown on its own (%)": 100 * (prune[name][has] == 0).mean(),
                       "event: visit 1": ev[has, 0].mean(),
                       "event: visit 2": ev[has, 1].mean(),
                       "feature: code 1": fe[has, 0].mean(),
                       "feature: code 2": fe[has, 1].mean(),
                       "cell: visit 1, code 1": ce[has, 0, 0].mean(),
                       "cell: visit 2, code 2": ce[has, 1, 1].mean()}
            if pattern == "T5":
                code_1 = fe[:, 0][Xv[:, :, 0].any(axis=1)].sum()
                code_2 = fe[:, 1][Xv[:, :, 1].any(axis=1)].sum()
                key = {"feature: share code 2": code_2 / (code_1 + code_2)}
                has = Xv[:, :, 0].any(axis=1)
                cell_1 = ce[has, :, 0].sum()
                cell_2 = ce[has, :, 1].sum()
                key["cell: share code 2"] = cell_2 / (cell_1 + cell_2)

            # Number of players at event level (visits on their own + 1 for the merged visits)
            key["players (event level)"] = (T - prune[name] + (prune[name] > 0)).mean()

            # Share of persons with the complete pattern for whom the cell level ran (%)
            key["cell level ran (%)"] = 100 * cell_run[name][has].mean()

            # Errors against the ground truth and run time (not for the ground truth).
            # Event level: per visit, with the value of the merged visits spread equally over them.
            if name != "ground truth":
                per_visit = {}
                for s in [name, "ground truth"]:
                    v = np.nan_to_num(events[s])
                    for e in range(n_explain):
                        if prune[s][e] > 0:
                            v[e, :prune[s][e]] = events_merged[s][e] / prune[s][e]
                    per_visit[s] = v
                key["MAE event level"] = np.abs(per_visit[name] - per_visit["ground truth"]).mean()
                key["MAE feature level"] = np.abs(features[name] - features["ground truth"]).mean()
                key["MAE cell level"] = np.abs(cell_map[name] - cell_map["ground truth"]).mean()
                key["seconds per person"] = seconds[name]

            for key_number, value in key.items():
                all_results.append({"rep": rep, "pattern": pattern, "key number": key_number,
                                    "source": name, "value": value})

#.......................................................................................
# 13. Results: mean (SD) over the repetitions
# (no longer indented: runs once, after the loops)
#.......................................................................................

all_results = pd.DataFrame(all_results)
all_results.to_csv("results_timeshap.csv", index=False)

means = all_results.groupby(["pattern", "key number", "source"], sort=False)["value"].mean().unstack()
sds = all_results.groupby(["pattern", "key number", "source"], sort=False)["value"].std().unstack()

# "mean (SD)"; SD 0 if there is only one repetition; "-" where there is no value
table = means.round(2).astype(str) + " (" + sds.fillna(0).round(2).astype(str) + ")"
table = table.where(means.notna(), "-")
table = table[["ground truth", "true logit", "GRU", "TabPFN"]]

# Rows in the order in which they were computed (pattern by pattern)
order = list(all_results[["pattern", "key number"]].drop_duplicates().itertuples(index=False, name=None))
table = table.loc[order]

# Expected values WITHOUT pruning (what TimeSHAP would show if nothing were merged)
expected = {("T1", "event: visit 20"): beta, ("T1", "feature: code 1"): beta,
            ("T2", "event: visit 18"): beta, ("T2", "event: visit 20"): -beta, ("T2", "feature: code 1"): 0,
            ("T3", "event: visit 3"): beta / 2, ("T3", "event: visit 18"): beta / 2,
            ("T3", "feature: code 1"): beta / 2, ("T3", "feature: code 2"): beta / 2,
            ("T4", "visit 1 shown on its own (%)"): 100, ("T4", "event: visit 1"): beta,
            ("T4", "event: visit 2"): -beta, ("T4", "feature: code 1"): beta, ("T4", "feature: code 2"): -beta,
            ("T5", "feature: share code 2"): 0,
            ("T1", "cell: visit 20, code 1"): beta,
            ("T2", "cell: visit 18, code 1"): beta, ("T2", "cell: visit 20, code 1"): -beta,
            ("T3", "cell: visit 3, code 1"): beta / 2, ("T3", "cell: visit 18, code 2"): beta / 2,
            ("T4", "cell: visit 1, code 1"): beta, ("T4", "cell: visit 2, code 2"): -beta,
            ("T5", "cell: share code 2"): 0}
table.insert(0, "expected (no pruning)", [expected.get(row, "-") for row in table.index])

pd.set_option("display.width", 220)
print(f"\nTimeSHAP, {n_reps} repetitions: mean (SD)")
print(table.to_string())

#.......................................................................................
# 14. Figure: one example person per pattern (last repetition)
# Left: the data. Middle: event level (value per visit; "M" = merged older visits).
# Right: feature level (value per code; "M" = merged older visits).
#.......................................................................................

colors = {"ground truth": "black", "true logit": "tab:grey", "GRU": "tab:blue", "TabPFN": "tab:orange"}
fig, axes = plt.subplots(5, 3, figsize=(20, 14), layout="constrained",
                         gridspec_kw={"width_ratios": [3, 4, 1.5]})

for r, pattern in enumerate(["T1", "T2", "T3", "T4", "T5"]):
    ex = examples[pattern]

    # Data
    ax = axes[r, 0]
    ax.imshow(ex["data"].T, cmap="Greys", vmin=0, vmax=1.4)
    ax.set_xticks(range(T), range(1, T + 1), fontsize=7)
    ax.set_yticks(range(D), [f"code {d + 1}" for d in range(D)])
    ax.set_title(f"{pattern}: data", fontsize=9, loc="left")

    # Event level: position 0 = merged visits, positions 1-20 = visits 1-20
    ax = axes[r, 1]
    ax.axhline(0, color="lightgrey")
    for s, name in enumerate(colors):
        offset = -0.3 + 0.2 * s
        ax.scatter(np.arange(1, T + 1) + offset, ex["events"][name], color=colors[name], s=18,
                   label=f"{name} (merged: visits 1-{ex['prune'][name]})" if ex["prune"][name] > 0 else name)
        if ex["prune"][name] > 0:
            ax.scatter([offset], [ex["events_merged"][name]], color=colors[name], s=18, marker="s")
    ax.set_xticks(range(T + 1), ["M"] + [str(t) for t in range(1, T + 1)], fontsize=7)
    ax.set_title("Event level (Shapley value per visit)", fontsize=9, loc="left")
    ax.legend(fontsize=7)

    # Feature level: positions 0-2 = codes 1-3, position 3 = merged visits
    ax = axes[r, 2]
    ax.axhline(0, color="lightgrey")
    for s, name in enumerate(colors):
        offset = -0.3 + 0.2 * s
        values = list(ex["features"][name]) + [ex["features_merged"][name]]
        ax.scatter(np.arange(D + 1) + offset, values, color=colors[name], s=18)
    ax.set_xticks(range(D + 1), [f"code {d + 1}" for d in range(D)] + ["M"], fontsize=7)
    ax.set_title("Feature level (Shapley value per code)", fontsize=9, loc="left")

plt.savefig("figure_timeshap.png", dpi=150, bbox_inches="tight")
plt.show()

#.......................................................................................
# 15. Figure: cell level of the example person per pattern (last repetition)
# Left: the data. Then the cell-level map for the ground truth and each model.
# Numbers: cells that were players on their own; the colour of the other cells is their
# share of a group's value. Dashed line: visits 1..p left of it were merged by pruning.
#.......................................................................................

sources = ["ground truth", "true logit", "GRU", "TabPFN"]
fig, axes = plt.subplots(5, 5, figsize=(26, 9), layout="constrained")

for r, pattern in enumerate(["T1", "T2", "T3", "T4", "T5"]):
    ex = examples[pattern]

    # Data
    ax = axes[r, 0]
    ax.imshow(ex["data"].T, cmap="Greys", vmin=0, vmax=1.4, aspect="auto")
    ax.set_title(f"{pattern}: data", fontsize=9)

    # Cell level of each source
    for c, name in enumerate(sources):
        ax = axes[r, c + 1]
        im = ax.imshow(ex["cell_map"][name].T, cmap="RdBu_r", vmin=-beta, vmax=beta, aspect="auto")
        for t, d in zip(*np.nonzero(ex["cells_alone"][name])):
            ax.text(t, d, f"{ex['cell_map'][name][t, d]:.2f}", ha="center", va="center", fontsize=6)
        if ex["prune"][name] > 0:
            ax.axvline(ex["prune"][name] - 0.5, color="black", linestyle="--", linewidth=1)
        title = f"cell level: {name}"
        if not ex["cell_run"][name]:
            title += " (did not run)"
        ax.set_title(title, fontsize=9)

    for ax in axes[r]:
        ax.set_xticks(range(T), range(1, T + 1), fontsize=7)
        ax.set_yticks(range(D), [f"code {d + 1}" for d in range(D)])

fig.colorbar(im, ax=axes, shrink=0.6, label="Shapley value (logit)")
plt.savefig("figure_timeshap_cells.png", dpi=150, bbox_inches="tight")
plt.show()