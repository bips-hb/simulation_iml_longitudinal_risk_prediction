"""
compute_one_task.py
===================
Entry point for one SLURM array task.
Handles training all models and computing all attribution methods for one
(repetition, scenario) combination.

Called by submit_array.sh as:
    python compute_one_task.py --rep 7 --scenario high_noise

The script is intentionally self-contained: all paths, constants, and helper
functions are defined here so a single task can be run interactively on a
login node for debugging without importing from a notebook.
"""

import argparse
import json
import os
import pickle
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ── Project imports (adjust to your actual module layout) ──────────────────
import utils
from utils import (
    GRUModel, LSTMModel, AttentionGRUModel, AttentionLSTMModel, CNNModel,
    set_seed, run_training_pipeline,
)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = Path('/home/hoepler/eICU_Real_World_Example')
SIM_ROOT     = BASE_DIR / 'sim_outputs_repeated'
RESULT_ROOT  = BASE_DIR / 'simulation_results_repeated'

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'baseline':         'N3000_V20_trig0.15_esc0.10_bg0.010_corr0.00_ac0.00',
    'high_noise':       'N3000_V20_trig0.15_esc0.10_bg0.050_corr0.00_ac0.00',
    'high_correlation': 'N3000_V20_trig0.15_esc0.10_bg0.010_corr0.30_ac0.40',
    'short_history':    'N3000_V10_trig0.15_esc0.10_bg0.010_corr0.00_ac0.00',
}

N_STATIC    = 2
HIDDEN_SIZE = 32
NUM_FILTERS = 64
TEST_SIZE   = 0.20
SEED_OFFSET = 3105     # base seed; actual seed = base + rep index

# Only save model weights for these audit repetitions (disk budget management)
AUDIT_REPS = {1, 10, 25, 50, 75, 100}

MODEL_KEYS   = ['gru', 'lstm', 'attn_gru', 'attn_lstm', 'cnn']
MODEL_LABELS = ['GRU', 'LSTM', 'AttnGRU', 'AttnLSTM', 'CNN']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, rep: int, scenario: str, tag: str,
                   sim_dir: Path, extra: dict = None):
    """
    Write a manifest.json recording the exact run parameters, seed, and
    content hashes of the simulation CSVs. Lets you detect stale stores
    or corrupted input files without re-running anything.
    """
    manifest = {
        'rep':       rep,
        'scenario':  scenario,
        'tag':       tag,
        'seed':      SEED_OFFSET + rep,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device':    str(device),
        'sim_hashes': {},
    }
    for fname in [f'ehr_events_{tag}.csv', f'static_data_{tag}.csv',
                  f'phenotype_labels_gt_{tag}.csv']:
        p = sim_dir / fname
        if p.exists():
            manifest['sim_hashes'][fname] = md5_of_file(p)
    if extra:
        manifest.update(extra)
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  manifest.json written')


def load_simulation_data(sim_dir: Path, tag: str):
    """Load the R-generated simulation CSVs for one scenario/repetition."""
    df_events = pd.read_csv(sim_dir / f'ehr_events_{tag}.csv')
    df_static = pd.read_csv(sim_dir / f'static_data_{tag}.csv')
    labels    = pd.read_csv(sim_dir / f'phenotype_labels_gt_{tag}.csv')
    return df_events, df_static, labels


def build_tensor(df_events, df_static, feature_names, n_visits, n_tokens,
                 n_static, n_patients):
    """
    Convert sparse event log + static features into (N, T, P) tensor
    and (N, n_static) static feature matrix. Same logic as in your
    baseline_compute.py.
    """
    tokens = feature_names[:n_tokens]
    tok_idx = {t: i for i, t in enumerate(tokens)}

    X_seq = np.zeros((n_patients, n_visits, n_tokens), dtype=np.float32)
    for _, row in df_events.iterrows():
        pid = int(row['patient_id']) - 1
        t   = int(row['time']) - 1
        c   = tok_idx.get(row['token'])
        if c is not None and t < n_visits and pid < n_patients:
            X_seq[pid, t, c] = 1.0

    X_static = df_static[['static_bin', 'static_cont']].values.astype(np.float32)
    y        = df_static['outcome'].values.astype(np.float32)
    return X_seq, X_static, y


def load_ground_truth_ntf(sim_dir: Path, tag: str, n_patients: int,
                           n_visits: int, n_tokens: int) -> np.ndarray:
    """
    Load ground-truth Shapley weights from the flat CSV files produced by R
    and reassemble into (N_patients, T, n_tokens) tensor.
    """
    icd_df = pd.read_csv(sim_dir / f'truth_icd_flat_{tag}.csv')
    atc_df = pd.read_csv(sim_dir / f'truth_atc_flat_{tag}.csv')

    n_icd = n_tokens - 40  # 100 ICD codes
    n_atc = 40             # 40 ATC codes

    gt = np.zeros((n_patients, n_visits, n_tokens), dtype=np.float32)

    icd_cols = [c for c in icd_df.columns if c.startswith('ICD_')]
    atc_cols = [c for c in atc_df.columns if c.startswith('ATC_')]

    for _, row in icd_df.iterrows():
        pid = int(row['patient_id']) - 1
        t   = int(row['visit']) - 1
        for j, col in enumerate(icd_cols):
            gt[pid, t, j] = row[col]

    for _, row in atc_df.iterrows():
        pid = int(row['patient_id']) - 1
        t   = int(row['visit']) - 1
        for j, col in enumerate(atc_cols):
            gt[pid, t, n_icd + j] = row[col]

    return gt


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X_seq_train, X_static_train, y_train,
                     X_seq_val, X_static_val, y_val,
                     input_size, out_dir, seed):
    models = {}
    hists  = {}
    perfs  = {}

    model_specs = [
        ('gru',       GRUModel(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                output_size=1, static_size=N_STATIC)),
        ('lstm',      LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                 output_size=1, static_size=N_STATIC)),
        ('attn_gru',  AttentionGRUModel(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                         output_size=1, static_size=N_STATIC)),
        ('attn_lstm', AttentionLSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                          output_size=1, static_size=N_STATIC)),
        ('cnn',       CNNModel(input_size=input_size, output_size=1,
                               static_size=N_STATIC, num_filters=NUM_FILTERS)),
    ]

    for mkey, model in model_specs:
        print(f'  Training {mkey}...')
        set_seed(seed)
        model, hist, perf = run_training_pipeline(
            model=model, device=device,
            X_seq_train=X_seq_train, X_static_train=X_static_train, y_train=y_train,
            X_seq_val=X_seq_val,   X_static_val=X_static_val,   y_val=y_val,
        )
        torch.save(model.state_dict(), out_dir / f'best_model_{mkey}.pth')
        models[mkey] = model
        hists[mkey]  = hist
        perfs[mkey]  = perf
        print(f'    AUROC={perf["auroc"]:.3f}')

    return models, hists, perfs


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

def compute_ig(models, X_seq_test, X_static_test, out_dir, n_steps=64):
    """Integrated Gradients — full test set."""
    from captum.attr import IntegratedGradients
    arrays = {}
    for mkey in MODEL_KEYS:
        m = models[mkey].to(device).eval()
        ig = IntegratedGradients(lambda x: m(x, torch.zeros(x.shape[0], N_STATIC).to(device)))
        X_t = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        attr = ig.attribute(X_t, n_steps=n_steps)
        arrays[f'ig_{mkey}_ntf'] = attr.detach().cpu().numpy()
    np.savez_compressed(out_dir / 'ig_arrays.npz', **arrays)
    print(f'  IG saved: {out_dir / "ig_arrays.npz"}')


def compute_eg(models, X_seq_test, X_static_test, X_seq_train, out_dir,
               n_samples=200, bg_size=128, seed=42):
    """Expected Gradients — full test set, training-data background."""
    from captum.attr import GradientShap
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_seq_train), size=bg_size, replace=False)
    bg     = torch.tensor(X_seq_train[bg_idx], dtype=torch.float32).to(device)

    arrays = {}
    for mkey in MODEL_KEYS:
        m = models[mkey].to(device).eval()
        gs = GradientShap(lambda x: m(x, torch.zeros(x.shape[0], N_STATIC).to(device)))
        X_t = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        attr = gs.attribute(X_t, baselines=bg, n_samples=n_samples,
                            stdevs=0.0, return_convergence_delta=False)
        arrays[f'eg_{mkey}_ntf'] = attr.detach().cpu().numpy()
    np.savez_compressed(out_dir / 'eg_arrays.npz', **arrays)
    print(f'  EG saved: {out_dir / "eg_arrays.npz"}')


def compute_kernelshap(models, X_seq_test, X_static_test, X_seq_train,
                       case_indices, out_dir, n_bg=300, n_samples=2048):
    """KernelSHAP — case patients only (computationally expensive)."""
    import shap
    X_explain = X_seq_test[case_indices]
    bg_idx    = np.random.choice(len(X_seq_train), size=n_bg, replace=False)
    bg        = X_seq_train[bg_idx]

    arrays = {'case_indices': np.array(case_indices)}
    for mkey in MODEL_KEYS:
        m = models[mkey].to(device).eval()

        def predict(x_flat):
            N  = x_flat.shape[0]
            T  = X_explain.shape[1]
            P  = X_explain.shape[2]
            xt = torch.tensor(x_flat.reshape(N, T, P), dtype=torch.float32).to(device)
            xs = torch.zeros(N, N_STATIC).to(device)
            with torch.no_grad():
                return m(xt, xs).cpu().numpy()

        bg_flat = bg.reshape(n_bg, -1)
        ex      = shap.KernelExplainer(predict, bg_flat)
        shap_vals = ex.shap_values(X_explain.reshape(len(case_indices), -1),
                                   nsamples=n_samples, silent=True)
        T, P = X_explain.shape[1], X_explain.shape[2]
        arrays[f'shap_ntf_{mkey}'] = shap_vals.reshape(-1, T, P)
    np.savez_compressed(out_dir / 'kernelshap_results.npz', **arrays)
    print(f'  KernelSHAP saved')


def compute_timeshap(models, X_seq_test, X_seq_train, case_indices, out_dir,
                     n_samples=32000, pruning_tol=0.025):
    """TimeSHAP cell-level — case patients only."""
    import timeshap
    X_explain = X_seq_test[case_indices]
    bg_idx    = np.random.choice(len(X_seq_train), size=300, replace=False)
    bg        = X_seq_train[bg_idx].mean(axis=0, keepdims=True)  # mean background

    for mkey in MODEL_KEYS:
        m = models[mkey].to(device).eval()
        arrays = {}

        ntf = np.zeros_like(X_explain)
        for i, x in enumerate(X_explain):
            try:
                cell_attr = timeshap.explain.local_report(
                    model=lambda s: m(torch.tensor(s, dtype=torch.float32).to(device),
                                      torch.zeros(s.shape[0], N_STATIC).to(device)).cpu().detach().numpy(),
                    data=x[np.newaxis],
                    baseline=bg,
                    pruning_dict={'tol': pruning_tol},
                    cell_dict={'nsamples': n_samples},
                )
                # map back to (T, P) array
                for _, row in cell_attr.iterrows():
                    t = int(row['Feature timestep'])
                    c = int(row['Feature index'])
                    ntf[i, t, c] = row['Shapley Value']
            except Exception as e:
                print(f'    TimeSHAP failed for case {i}: {e}')
        arrays[f'ts_{mkey}_ntf'] = ntf
        np.savez_compressed(out_dir / f'timeshap_cell_ntf_{mkey}.npz', **arrays)
    print(f'  TimeSHAP saved')


def compute_windowshap(models, X_seq_test, X_seq_train, case_indices,
                        out_dir, window_len=5):
    """WindowSHAP (Stationary, Sliding, Dynamic) — case patients only."""
    from windowshap import StationaryWindowSHAP, SlidingWindowSHAP, DynamicWindowSHAP
    X_explain = X_seq_test[case_indices]
    bg_idx    = np.random.choice(len(X_seq_train), size=300, replace=False)
    bg        = X_seq_train[bg_idx]

    for mkey in MODEL_KEYS:
        m = models[mkey].to(device).eval()

        def predict(x):
            xt = torch.tensor(x, dtype=torch.float32).to(device)
            xs = torch.zeros(x.shape[0], N_STATIC).to(device)
            with torch.no_grad():
                return m(xt, xs).cpu().numpy()

        stat = StationaryWindowSHAP(predict, bg, window_len=window_len)
        slid = SlidingWindowSHAP(predict, bg, window_len=window_len)
        dyn  = DynamicWindowSHAP(predict, bg, window_len=window_len)

        stat_ntf = stat.shap_values(X_explain)
        slid_ntf = slid.shap_values(X_explain)
        dyn_ntf  = dyn.shap_values(X_explain)

        np.savez_compressed(out_dir / f'windowshap_{mkey}.npz',
                            stat_ntf=stat_ntf, slid_ntf=slid_ntf,
                            dyn_ntf=dyn_ntf, tp_ws=np.array(case_indices))
    print(f'  WindowSHAP saved')


# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_case_indices(labels, gt_ntf, pid_test):
    """
    Select one representative test patient per phenotype group by signal
    strength (not predicted probability), so case selection is independent
    of model quality. Returns a flat list of test-set indices.
    """
    case_indices = []
    group_priorities = {
        'trigger_only':      lambda df: df['has_trigger_gt'],
        'chronic_only':      lambda df: df['has_chronic_gt'] & ~df['has_trigger_gt'] & ~df['has_escalation_gt'],
        'escalation_only':   lambda df: df['has_escalation_gt'] & ~df['has_trigger_gt'] & ~df['has_chronic_gt'],
        'trigger+chronic':   lambda df: df['has_trigger_gt'] & df['has_chronic_gt'],
        'trigger+escalation':lambda df: df['has_trigger_gt'] & df['has_escalation_gt'],
        'escalation+chronic':lambda df: df['has_escalation_gt'] & df['has_chronic_gt'] & ~df['has_trigger_gt'],
        'all_three':         lambda df: df['has_trigger_gt'] & df['has_escalation_gt'] & df['has_chronic_gt'],
    }
    for group, mask_fn in group_priorities.items():
        pool = labels[mask_fn(labels) & labels['patient_id'].isin(pid_test)]
        if len(pool) == 0:
            continue
        # pick patient with highest total ground-truth weight (signal strength)
        pool = pool.copy()
        pool['gt_sum'] = pool['patient_id'].map(
            lambda pid: float(gt_ntf[int(pid) - 1].sum()))
        best_pid = pool.nlargest(1, 'gt_sum')['patient_id'].values[0]
        idx      = np.where(pid_test == best_pid)[0]
        if len(idx) > 0:
            case_indices.append(int(idx[0]))

    return sorted(set(case_indices))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep',      type=int,   required=True)
    parser.add_argument('--scenario', type=str,   required=True)
    args = parser.parse_args()

    rep      = args.rep
    scenario = args.scenario
    tag      = SCENARIOS[scenario]
    seed     = SEED_OFFSET + rep

    rep_tag  = f'rep_{rep:03d}'
    sim_dir  = SIM_ROOT / rep_tag
    out_dir  = RESULT_ROOT / rep_tag / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'rep={rep}  scenario={scenario}  seed={seed}  device={device}')
    print(f'sim_dir : {sim_dir}')
    print(f'out_dir : {out_dir}')
    print(f'{"="*60}')

    # ── Check if this task has already completed (idempotent) ──────────────
    store_path = out_dir / f'scenario_store_{scenario}.pkl'
    if store_path.exists() and (out_dir / 'manifest.json').exists():
        print('Store and manifest already exist — skipping (delete to rerun).')
        return

    # ── Load simulation data ───────────────────────────────────────────────
    print('\n[1] Loading simulation data...')
    df_events, df_static, labels = load_simulation_data(sim_dir, tag)

    n_patients = len(df_static)
    tokens     = sorted(df_events['token'].unique())
    n_tokens   = len(tokens)
    n_visits   = int(tag.split('_')[1][1:])
    feature_names = tokens + ['static_bin', 'static_cont']

    X_seq, X_static, y = build_tensor(
        df_events, df_static, feature_names, n_visits, n_tokens, N_STATIC, n_patients)

    gt_ntf = load_ground_truth_ntf(sim_dir, tag, n_patients, n_visits, n_tokens)
    print(f'  {n_patients} patients, {n_visits} visits, {n_tokens} tokens')
    print(f'  Outcome prevalence: {y.mean():.3f}')

    # ── Train/test split ───────────────────────────────────────────────────
    print('\n[2] Train/test split (80/20, stratified)...')
    idx_all = np.arange(n_patients)
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y, random_state=seed)

    X_seq_train  = X_seq[idx_train];   X_seq_test  = X_seq[idx_test]
    X_static_trn = X_static[idx_train]; X_static_tst = X_static[idx_test]
    y_train      = y[idx_train];        y_test      = y[idx_test]
    pid_train    = idx_train + 1        # 1-indexed to match R patient IDs
    pid_test     = idx_test  + 1

    # 10% of training as validation for early stopping
    idx_tr2, idx_val = train_test_split(
        np.arange(len(idx_train)), test_size=0.125, random_state=seed)
    X_seq_tr2  = X_seq_train[idx_tr2];    X_seq_val  = X_seq_train[idx_val]
    X_static_t2 = X_static_trn[idx_tr2];  X_static_v = X_static_trn[idx_val]
    y_tr2      = y_train[idx_tr2];        y_val      = y_train[idx_val]

    # ── Train models ───────────────────────────────────────────────────────
    print('\n[3] Training models...')
    models, hists, perfs = train_all_models(
        X_seq_tr2, X_static_t2, y_tr2,
        X_seq_val, X_static_v, y_val,
        input_size=n_tokens, out_dir=out_dir, seed=seed)

    # Remove weights for non-audit repetitions to save disk space
    if rep not in AUDIT_REPS:
        for mkey in MODEL_KEYS:
            wpath = out_dir / f'best_model_{mkey}.pth'
            if wpath.exists():
                wpath.unlink()
        print(f'  Model weights deleted (rep {rep} not in audit set).')

    # ── Case study indices (needed by KernelSHAP/TimeSHAP/WindowSHAP) ─────
    print('\n[4] Selecting case study patients...')
    case_indices = build_case_indices(labels, gt_ntf, pid_test)
    print(f'  Case indices: {case_indices}')

    # ── Attribution methods ────────────────────────────────────────────────
    print('\n[5] Computing attributions...')

    print('  IG...')
    compute_ig(models, X_seq_test, X_static_tst, out_dir)

    print('  EG...')
    compute_eg(models, X_seq_test, X_static_tst, X_seq_train, out_dir, seed=seed)

    print('  KernelSHAP...')
    compute_kernelshap(models, X_seq_test, X_static_tst, X_seq_train,
                       case_indices, out_dir)

    print('  TimeSHAP...')
    compute_timeshap(models, X_seq_test, X_seq_train, case_indices, out_dir)

    print('  WindowSHAP...')
    compute_windowshap(models, X_seq_test, X_seq_train, case_indices, out_dir)

    # ── Save scenario store ────────────────────────────────────────────────
    print('\n[6] Saving scenario store...')

    # Predicted probabilities per model for the test set
    y_pred = {}
    for mkey in MODEL_KEYS:
        m = models[mkey].eval()
        xt = torch.tensor(X_seq_test,  dtype=torch.float32).to(device)
        xs = torch.tensor(X_static_tst, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred[mkey] = m(xt, xs).cpu().numpy().ravel()

    store = {
        'feature_names':   feature_names,
        'n_visits':        n_visits,
        'n_tokens':        n_tokens,
        'n_static':        N_STATIC,
        'X_explain':       X_seq_test,
        'X_static_test':   X_static_tst,
        'X_train':         X_seq_train,
        'X_static_train':  X_static_trn,
        'y_test':          y_test,
        'y_train':         y_train,
        'y_pred':          y_pred,       # dict of mkey -> (N_test,) probabilities
        'pid_test':        pid_test,
        'pid_train':       pid_train,
        'gt_ntf':          gt_ntf,       # (N_patients, T, n_tokens) — the key array
        'case_indices':    case_indices,
        'perf':            perfs,
        'hist_gru':        hists.get('gru'),
        'hist_lstm':       hists.get('lstm'),
        'hist_attn_gru':   hists.get('attn_gru'),
        'hist_attn_lstm':  hists.get('attn_lstm'),
        'hist_cnn':        hists.get('cnn'),
    }
    with open(store_path, 'wb') as f:
        pickle.dump(store, f, protocol=4)
    print(f'  Store saved: {store_path}  ({store_path.stat().st_size / 1e6:.1f} MB)')

    # ── Write manifest ─────────────────────────────────────────────────────
    write_manifest(out_dir, rep, scenario, tag, sim_dir, extra={
        'n_test':    int(len(pid_test)),
        'n_train':   int(len(pid_train)),
        'perf':      {k: {m: round(float(v), 4) for m, v in p.items()}
                     for k, p in perfs.items()},
        'case_indices': case_indices,
    })

    print(f'\nTask complete: rep={rep}  scenario={scenario}')


if __name__ == '__main__':
    main()
