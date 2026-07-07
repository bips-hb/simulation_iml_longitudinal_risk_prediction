"""
baseline_compute.py
===================

Run in tmux:
    python3 baseline_compute.py 2>&1 | tee baseline_compute.log

Tail the log file in another terminal to monitor progress (even when being not attached to tmux):
    tail -f baseline_compute.log

    or: tail -100 -f baseline_compute.log (tail with last N lines)

Detach from tmux session (leave it running in the background): Ctrl+b, then d

Reattach to tmux session:
    tmux attach -t simulation_baseline
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import time as _time
from datetime import datetime as _datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import utils
from utils import (
    GRUModel, LSTMModel, LogitWrapper,
    AttentionGRUModel, AttentionLSTMModel,
    CNNModel, set_seed, pick_best,
    make_timeshap_predict_fn, make_windowshap_predict_fn, ModelWrapper,
)

set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
_section_times = {}

def log(msg=''):
    ts = _datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def log_section(name):
    now = _time.time()
    if name in _section_times:
        elapsed = now - _section_times.pop(name)
        m, s = divmod(int(elapsed), 60)
        log(f'{"─"*60}')
        log(f'  ✓  {name}  ({m}m {s}s)')
        log(f'{"─"*60}')
    else:
        _section_times[name] = now
        log(f'{"═"*60}')
        log(f'  ▶  {name}')
        log(f'{"═"*60}')

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SIM_DIR       = "/home/hoepler/eICU_Real_World_Example/sim_outputs_advanced/"
OUTPUT_DIR    = "/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/"
SCENARIO_NAME = "baseline"

# Updated tag format to match new simulation naming convention:
# N{patients}_V{visits}_trig{p}_esc{p}_bg{p}_corr{r}_ac{r}
TAG = "N3000_V20_trig0.15_esc0.10_bg0.010_corr0.00_ac0.00"

STORE_PATH = os.path.join(OUTPUT_DIR, 'scenario_store_baseline.pkl')

out = os.path.join(OUTPUT_DIR, SCENARIO_NAME)
os.makedirs(out, exist_ok=True)

MODEL_WEIGHTS = [
    'best_model_gru.pth',
    'best_model_lstm.pth',
    'best_model_attn_gru.pth',
    'best_model_attn_lstm.pth',
    'best_model_cnn.pth',
]

def _has_all_model_weights() -> bool:
    return all(os.path.exists(os.path.join(out, fname)) for fname in MODEL_WEIGHTS)

device = torch.device('cpu')

# ══════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth(sim_dir: str, tag: str) -> dict:
    """
    Load flat ground-truth Shapley weight matrices produced by the R simulation.

    The simulation stores analytical Shapley values at (patient, visit, code) level:
      - Trigger ICD_10 at t_trig       : 5.0  (10.0 / 2 by symmetry)
      - Trigger ATC_5  at t_trig + 1   : 5.0
      - Chronic ICD_1/2/3 per occurrence: 0.5  (each occurrence independent)
      - Escalation-window ICDs          : 2.5 / n_active_late (patient-specific)
      - All other triples               : 0.0

    Returns a dict with:
      icd_weights : np.ndarray (N_patients, N_visits, N_ICD)  float32
      atc_weights : np.ndarray (N_patients, N_visits, N_ATC)  float32
      pheno_labels: pd.DataFrame with phenotype group labels per patient
    """
    icd_flat = pd.read_csv(
        os.path.join(sim_dir, f"truth_icd_flat_{tag}.csv"))
    atc_flat = pd.read_csv(
        os.path.join(sim_dir, f"truth_atc_flat_{tag}.csv"))
    pheno    = pd.read_csv(
        os.path.join(sim_dir, f"phenotype_labels_gt_{tag}.csv"))

    n_patients = icd_flat['patient_id'].nunique()
    n_visits   = icd_flat['visit'].nunique()

    icd_cols = [c for c in icd_flat.columns if c.startswith('ICD_')]
    atc_cols = [c for c in atc_flat.columns if c.startswith('ATC_')]
    n_icd    = len(icd_cols)
    n_atc    = len(atc_cols)

    # Reshape to (N_patients, N_visits, N_codes)
    # Rows in the flat CSV are ordered by (patient_id, visit) — same as R output
    icd_weights = (icd_flat[icd_cols]
                   .values
                   .reshape(n_patients, n_visits, n_icd)
                   .astype(np.float32))
    atc_weights = (atc_flat[atc_cols]
                   .values
                   .reshape(n_patients, n_visits, n_atc)
                   .astype(np.float32))

    log(f"Ground truth loaded: ICD {icd_weights.shape}, ATC {atc_weights.shape}")
    log(f"  ICD non-zero entries : {(icd_weights != 0).sum():,}")
    log(f"  ATC non-zero entries : {(atc_weights != 0).sum():,}")
    log(f"  Unique ICD weights   : {np.unique(icd_weights[icd_weights > 0]).round(4)}")

    return dict(
        icd_weights  = icd_weights,
        atc_weights  = atc_weights,
        pheno_labels = pheno,
        n_patients   = n_patients,
        n_visits     = n_visits,
        n_icd        = n_icd,
        n_atc        = n_atc,
    )


def build_gt_tensor(gt: dict, feature_names: list, n_tokens: int) -> np.ndarray:
    """
    Combine ICD and ATC ground-truth weight arrays into a single
    (N_patients, N_visits, n_tokens) tensor aligned to feature_names.

    feature_names is the token vocabulary from the event log (sorted),
    containing entries like 'ICD_10', 'ATC_5', etc.

    Returns:
        gt_ntf : np.ndarray (N_patients, N_visits, n_tokens) float32
                 Zero for static features and any token not in the ground truth.
    """
    n_patients = gt['n_patients']
    n_visits   = gt['n_visits']
    gt_ntf     = np.zeros((n_patients, n_visits, n_tokens), dtype=np.float32)

    for j, fname in enumerate(feature_names[:n_tokens]):
        if fname.startswith('ICD_'):
            icd_idx = int(fname.split('_')[1]) - 1  # R is 1-indexed
            if icd_idx < gt['n_icd']:
                gt_ntf[:, :, j] = gt['icd_weights'][:, :, icd_idx]
        elif fname.startswith('ATC_'):
            atc_idx = int(fname.split('_')[1]) - 1
            if atc_idx < gt['n_atc']:
                gt_ntf[:, :, j] = gt['atc_weights'][:, :, atc_idx]

    log(f"GT tensor built: shape={gt_ntf.shape}, "
        f"non-zero={( gt_ntf != 0).sum():,}")
    return gt_ntf


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS AGAINST GROUND-TRUTH SHAPLEY WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_attribution(
    attr_ntf:    np.ndarray,   # (N_test, T, F) predicted attributions
    gt_ntf:      np.ndarray,   # (N_patients, T, F) ground-truth Shapley weights
    pid_test:    np.ndarray,   # patient IDs in the test set (1-indexed)
    pheno_labels: pd.DataFrame,
    feature_names: list,
    n_tokens:    int,
) -> pd.DataFrame:
    """
    Compute attribution recovery metrics against the analytical ground-truth
    Shapley weights for each patient in the test set.

    Metrics (computed per patient, then aggregated):

      spearman_signal : Spearman rho between |attr| and gt_weight, over
                        signal positions (gt > 0) only.
      spearman_noise  : Spearman rho over noise positions (gt == 0) only.
                        High values here indicate false attribution mass.
      auprc           : Area under precision-recall curve treating binarised
                        gt (signal vs noise) as labels and |attr| as scores.
                        Preferred over AUROC because signal positions are sparse.
      nmae            : Normalised MAE between normalised |attr| and normalised
                        gt weights (both normalised to sum to 1 within patient).
                        Scale-invariant: removes differences in attribution scale
                        across methods.
      top_k_precision : Proportion of top-k attributed positions (by |attr|)
                        that coincide with ground-truth signal positions, where
                        k = number of non-zero gt positions for that patient.

    All metrics are returned as a long-format DataFrame with columns:
      patient_id, group_gt, metric, value

    Aggregation (mean + 95% bootstrap CI) should be done downstream,
    stratified by group_gt, to allow phenotype-level comparisons.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    records = []

    for i, pid in enumerate(pid_test):
        # Ground truth for this patient (0-indexed: pid is 1-indexed)
        gt_pat = gt_ntf[int(pid) - 1, :, :n_tokens].ravel()   # (T*F,)
        at_pat = np.abs(attr_ntf[i, :, :n_tokens]).ravel()     # (T*F,)

        signal_mask = gt_pat > 0
        noise_mask  = gt_pat == 0
        n_signal    = signal_mask.sum()
        n_noise     = noise_mask.sum()

        # phenotype group for this patient
        row = pheno_labels.loc[pheno_labels['patient_id'] == pid]
        group = row['group_gt'].values[0] if len(row) > 0 else 'unknown'

        result = {'patient_id': int(pid), 'group_gt': group}

        # --- Spearman on signal positions ---
        if n_signal >= 2:
            rho_sig, _ = spearmanr(at_pat[signal_mask], gt_pat[signal_mask])
            result['spearman_signal'] = float(rho_sig) if not np.isnan(rho_sig) else np.nan
        else:
            result['spearman_signal'] = np.nan

        # --- Spearman on noise positions ---
        if n_noise >= 2:
            rho_noi, _ = spearmanr(at_pat[noise_mask], gt_pat[noise_mask])
            result['spearman_noise'] = float(rho_noi) if not np.isnan(rho_noi) else np.nan
        else:
            result['spearman_noise'] = np.nan

        # --- AUPRC (binarised gt: signal=1, noise=0) ---
        if n_signal > 0 and n_noise > 0:
            gt_binary = signal_mask.astype(int)
            result['auprc'] = float(average_precision_score(gt_binary, at_pat))
        else:
            result['auprc'] = np.nan

        # --- Normalised MAE ---
        # Normalise both vectors to sum to 1 to remove scale differences
        at_sum = at_pat.sum()
        gt_sum = gt_pat.sum()
        if at_sum > 0 and gt_sum > 0:
            at_norm = at_pat / at_sum
            gt_norm = gt_pat / gt_sum
            result['nmae'] = float(np.mean(np.abs(at_norm - gt_norm)))
        else:
            result['nmae'] = np.nan

        # --- Top-k precision ---
        # k = number of non-zero gt positions for this patient
        if n_signal > 0:
            k        = int(n_signal)
            top_k    = np.argsort(at_pat)[::-1][:k]
            result['top_k_precision'] = float(signal_mask[top_k].mean())
        else:
            result['top_k_precision'] = np.nan

        records.append(result)

    return pd.DataFrame(records)


def aggregate_metrics(
    eval_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Aggregate patient-level metrics by phenotype group.
    Returns mean + 95% bootstrap CI per (group_gt, metric).
    """
    rng     = np.random.default_rng(seed)
    metrics = ['spearman_signal', 'spearman_noise', 'auprc', 'nmae', 'top_k_precision']
    rows    = []

    for group in sorted(eval_df['group_gt'].unique()):
        sub = eval_df[eval_df['group_gt'] == group]
        for metric in metrics:
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            mean_val = float(np.mean(vals))
            # Bootstrap 95% CI
            boot = np.array([
                np.mean(rng.choice(vals, size=len(vals), replace=True))
                for _ in range(n_bootstrap)
            ])
            rows.append(dict(
                group_gt = group,
                metric   = metric,
                n        = len(vals),
                mean     = round(mean_val, 4),
                ci_lower = round(float(np.percentile(boot, 2.5)),  4),
                ci_upper = round(float(np.percentile(boot, 97.5)), 4),
            ))

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — build case_studies from labels, y_probs, y_test, pid_test
# ══════════════════════════════════════════════════════════════════════════════
def _build_case_studies(labels, y_test, pid_test, y_probs_ref,
                        gt_ntf, feature_names, n_tokens):

    pid_to_test_index = {pid: i for i, pid in enumerate(pid_test)}
    labels_test = labels[labels['patient_id'].isin(pid_test)].copy()
    labels_test['test_idx'] = labels_test['patient_id'].map(pid_to_test_index)

    # Pre-compute signal strength per patient for signal-rich selection
    trigger_ids  = {feature_names.index('ICD_10'),
                    feature_names.index('ATC_5')}
    chronic_ids  = {feature_names.index(f)
                    for f in ['ICD_1', 'ICD_2', 'ICD_3']}
    escalation_ids_set = set(range(n_tokens)) - trigger_ids - chronic_ids

    def _signal_strength(test_idx, mode):
        pid    = int(pid_test[test_idx])
        gt_pat = gt_ntf[pid - 1]   # (T, n_tokens)
        if mode == 'trigger':
            # Both ICD_10 and ATC_5 must be non-zero — all trigger patients
            # have exactly 2 signal positions so use predicted prob as tiebreak
            return y_probs_ref[test_idx]
        elif mode == 'chronic':
            return sum((gt_pat[:, j] != 0).sum() for j in chronic_ids)
        elif mode == 'escalation':
            return sum((gt_pat[:, j] != 0).sum() for j in escalation_ids_set)
        else:
            # Mixed: total non-zero weight positions
            return (gt_pat != 0).sum()

    case_studies = {}
    for group, title, mode in [
        ('trigger_only',       'Pure Trigger',            'trigger'),
        ('chronic_only',       'Pure Chronic',            'chronic'),
        ('escalation_only',    'Pure Late Escalation',    'escalation'),
        ('trigger+chronic',    'Mixed: Trigger + Chronic','mixed'),
        ('trigger+escalation', 'Mixed: Trigger + Escalation', 'mixed'),
        ('escalation+chronic', 'Mixed: Chronic + Escalation', 'mixed'),
        ('all_three',          'Mixed: All Three',        'mixed'),
    ]:
        cand = labels_test.loc[
            labels_test['group_gt'] == group, 'test_idx'
        ].dropna().astype(int).values
        cand = cand[y_test[cand] == 1]
        if len(cand) == 0:
            continue

        # Select candidate with strongest signal, break ties by predicted prob
        scores = np.array([_signal_strength(i, mode) for i in cand])
        best   = cand[np.argmax(scores)]
        case_studies[title] = best

    # False positive: highest predicted prob among true negatives
    fp = np.where((y_test == 0) & (y_probs_ref > 0.6))[0]
    p  = pick_best(fp, y_probs_ref)
    if p is not None:
        case_studies['False Positive'] = p

    return case_studies


# ══════════════════════════════════════════════════════════════════════════════
# PART A — DATA LOADING & MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(STORE_PATH) or not _has_all_model_weights():
    if os.path.exists(STORE_PATH) and not _has_all_model_weights():
        log("WARNING: persisted store found but one or more model checkpoint files are missing.")
        log("         Re-running Part A to regenerate model weights and refresh the store.")

    log_section('Part A: Data loading & training')

    df_events = pd.read_csv(os.path.join(SIM_DIR, f"ehr_events_{TAG}.csv"))
    df_static = pd.read_csv(os.path.join(SIM_DIR, f"static_data_{TAG}.csv"))
    labels    = pd.read_csv(os.path.join(SIM_DIR, f"phenotype_labels_gt_{TAG}.csv"))

    n_patients   = df_static['patient_id'].nunique()
    n_visits     = int(df_events['time'].max())
    tokens       = sorted(df_events['token'].unique())
    token_to_idx = {t: i for i, t in enumerate(tokens)}
    n_tokens     = len(tokens)

    X_full = np.zeros((n_patients, n_visits, n_tokens + 2))
    for _, row in df_events.iterrows():
        X_full[int(row['patient_id'])-1, int(row['time'])-1,
               token_to_idx[row['token']]] = 1
    for i in range(n_patients):
        X_full[i, :, n_tokens]     = df_static.loc[i, 'static_bin']
        X_full[i, :, n_tokens + 1] = df_static.loc[i, 'static_cont']

    y             = df_static['outcome'].values
    patient_ids   = np.arange(1, n_patients + 1)
    feature_names = tokens + ['Static_Bin', 'Static_Cont']
    X_seq         = X_full[:, :, :n_tokens].astype(np.float32)
    X_static      = X_full[:, 0, n_tokens:].astype(np.float32)
    n_static      = X_static.shape[1]

    (X_seq_train, X_seq_test,
     X_static_train, X_static_test,
     y_train, y_test,
     pid_train, pid_test) = train_test_split(
        X_seq, X_static, y, patient_ids,
        test_size=0.2, random_state=3105, stratify=y)

    X_train_flat = np.concatenate([X_seq_train.sum(axis=1), X_static_train], axis=1)
    X_test_flat  = np.concatenate([X_seq_test.sum(axis=1),  X_static_test],  axis=1)

    log(f"Patients: {n_patients}  Visits: {n_visits}  Tokens: {n_tokens}")

    # Load ground truth and build aligned tensor
    gt            = load_ground_truth(SIM_DIR, TAG)
    gt_ntf        = build_gt_tensor(gt, feature_names, n_tokens)

    input_size      = n_tokens
    model_gru       = GRUModel(          input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_lstm      = LSTMModel(         input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_attn_gru  = AttentionGRUModel( input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_attn_lstm = AttentionLSTMModel(input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_cnn       = CNNModel(          input_size=input_size, output_size=1,  static_size=n_static, num_filters=64)

    hist_gru = hist_lstm = hist_attn_gru = hist_attn_lstm = hist_cnn = None
    for label, model, path, hkey in [
        ("GRU",      model_gru,       os.path.join(out, 'best_model_gru.pth'),       'hist_gru'),
        ("LSTM",     model_lstm,      os.path.join(out, 'best_model_lstm.pth'),      'hist_lstm'),
        ("AttnGRU",  model_attn_gru,  os.path.join(out, 'best_model_attn_gru.pth'),  'hist_attn_gru'),
        ("AttnLSTM", model_attn_lstm, os.path.join(out, 'best_model_attn_lstm.pth'), 'hist_attn_lstm'),
        ("CNN",      model_cnn,       os.path.join(out, 'best_model_cnn.pth'),       'hist_cnn'),
    ]:
        log(f"  Training {label}...")
        hist = utils.run_training_pipeline(
            model, X_seq_train, y_train, X_seq_test, y_test, save_path=path,
            X_static_train=X_static_train, X_static_val=X_static_test)
        if   hkey == 'hist_gru':       hist_gru       = hist
        elif hkey == 'hist_lstm':      hist_lstm      = hist
        elif hkey == 'hist_attn_gru':  hist_attn_gru  = hist
        elif hkey == 'hist_attn_lstm': hist_attn_lstm = hist
        elif hkey == 'hist_cnn':       hist_cnn       = hist
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        log(f"  {label} done.")

    lr_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=3105)
    lr_model.fit(X_train_flat, y_train)
    rf_model.fit(X_train_flat, y_train)

    def _auc(model, X_seq, y, X_static=None, flat=False):
        if flat:
            return roc_auc_score(y, model.predict_proba(X_seq)[:, 1])
        s_t = torch.tensor(X_static, dtype=torch.float32).to(device) \
              if X_static is not None else None
        with torch.no_grad():
            p = model(torch.tensor(X_seq, dtype=torch.float32).to(device),
                      s_t).cpu().numpy().flatten()
        return roc_auc_score(y, p)

    perf = {
        'GRU_test':      _auc(model_gru,       X_seq_test,  y_test,  X_static=X_static_test),
        'LSTM_test':     _auc(model_lstm,       X_seq_test,  y_test,  X_static=X_static_test),
        'AttnGRU_test':  _auc(model_attn_gru,  X_seq_test,  y_test,  X_static=X_static_test),
        'AttnLSTM_test': _auc(model_attn_lstm,  X_seq_test,  y_test,  X_static=X_static_test),
        'CNN_test':      _auc(model_cnn,        X_seq_test,  y_test,  X_static=X_static_test),
        'LR_test':       _auc(lr_model, X_test_flat,  y_test,  flat=True),
        'RF_test':       _auc(rf_model, X_test_flat,  y_test,  flat=True),
        'LR_train':      _auc(lr_model, X_train_flat, y_train, flat=True),
        'RF_train':      _auc(rf_model, X_train_flat, y_train, flat=True),
    }
    log(f"Test AUC: { {k: round(v, 4) for k, v in perf.items() if 'test' in k} }")

    X_test_tensor_a = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_static_t_a    = torch.tensor(X_static_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_probs_ref_a = model_gru(X_test_tensor_a,
                                  X_static_t_a).cpu().numpy().flatten()

    case_studies = _build_case_studies(labels, y_test, pid_test, y_probs_ref_a, gt_ntf, feature_names, n_tokens)
    log(f"Case studies selected: { {t: int(i) for t, i in case_studies.items()} }")

    with open(STORE_PATH, 'wb') as f:
        pickle.dump({
            'hist_gru':      hist_gru,      'hist_lstm':      hist_lstm,
            'hist_attn_gru': hist_attn_gru, 'hist_attn_lstm': hist_attn_lstm,
            'hist_cnn':      hist_cnn,
            'perf':          perf,
            'feature_names': feature_names,
            'n_tokens':      n_tokens,
            'n_visits':      n_visits,
            'n_static':      n_static,
            'X_explain':     X_seq_test,    'X_static_test':  X_static_test,
            'X_train':       X_seq_train,   'X_static_train': X_static_train,
            'y_test':        y_test,        'y_train':        y_train,
            'pid_test':      pid_test,
            # Ground truth
            'gt_ntf':        gt_ntf,
            # Case selection
            'case_studies':  case_studies,
            'case_indices':  list(case_studies.values()),
        }, f)
    log(f"Store saved to: {STORE_PATH}")
    log_section('Part A: Data loading & training')
    log("Part A complete.")

else:
    log("Store already exists — loading from disk, skipping Part A.")
    log_section('Part A: Data loading & training')
    log_section('Part A: Data loading & training')

    with open(STORE_PATH, 'rb') as f:
        _store = pickle.load(f)

    feature_names  = _store['feature_names']
    n_visits       = _store['n_visits']
    n_static       = _store.get('n_static', 2)
    X_seq_test     = _store['X_explain']
    X_static_test  = _store.get('X_static_test', None)
    X_seq_train    = _store['X_train']
    X_static_train = _store.get('X_static_train', None)
    y_test         = _store['y_test']
    y_train        = _store['y_train']
    pid_test       = _store['pid_test']
    hist_gru       = _store['hist_gru']
    hist_lstm      = _store['hist_lstm']
    hist_attn_gru  = _store['hist_attn_gru']
    hist_attn_lstm = _store['hist_attn_lstm']
    hist_cnn       = _store.get('hist_cnn', None)
    perf           = _store['perf']

    # Load ground truth — rebuild from CSV if store predates this change
    if 'gt_ntf' in _store:
        gt_ntf = _store['gt_ntf']
        log("Ground truth loaded from store.")
    else:
        log("WARNING: store predates gt_ntf. Loading from CSV and re-saving.")
        n_tokens_tmp   = _store.get('n_tokens', len(feature_names) - n_static)
        gt             = load_ground_truth(SIM_DIR, TAG)
        gt_ntf         = build_gt_tensor(gt, feature_names, n_tokens_tmp)
        _store['gt_ntf'] = gt_ntf
        with open(STORE_PATH, 'wb') as f:
            pickle.dump(_store, f)
        log("Store updated with gt_ntf.")

    n_patients = int(TAG.split('_')[0][1:])
    n_tokens   = _store.get('n_tokens', len(feature_names) - n_static)
    input_size = n_tokens

    model_gru       = GRUModel(          input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_lstm      = LSTMModel(         input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_attn_gru  = AttentionGRUModel( input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_attn_lstm = AttentionLSTMModel(input_size=input_size, hidden_size=32, output_size=1, static_size=n_static)
    model_cnn       = CNNModel(          input_size=input_size, output_size=1,  static_size=n_static, num_filters=64)

    for m, fname in [
        (model_gru,       'best_model_gru.pth'),
        (model_lstm,      'best_model_lstm.pth'),
        (model_attn_gru,  'best_model_attn_gru.pth'),
        (model_attn_lstm, 'best_model_attn_lstm.pth'),
        (model_cnn,       'best_model_cnn.pth'),
    ]:
        m.load_state_dict(torch.load(os.path.join(out, fname), map_location=device))
        m.to(device).eval()
    log("Models loaded from disk.")

    if X_static_train is not None:
        X_train_flat = np.concatenate([X_seq_train.sum(axis=1), X_static_train], axis=1)
        X_test_flat  = np.concatenate([X_seq_test.sum(axis=1),  X_static_test],  axis=1)
    else:
        X_train_flat = X_seq_train.sum(axis=1)
        X_test_flat  = X_seq_test.sum(axis=1)

    lr_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=3105)
    lr_model.fit(X_train_flat, y_train)
    rf_model.fit(X_train_flat, y_train)
    log("LR and RF refitted.")

    df_events = pd.read_csv(os.path.join(SIM_DIR, f"ehr_events_{TAG}.csv"))
    df_static = pd.read_csv(os.path.join(SIM_DIR, f"static_data_{TAG}.csv"))
    labels    = pd.read_csv(os.path.join(SIM_DIR, f"phenotype_labels_gt_{TAG}.csv"))
    tokens    = sorted(df_events['token'].unique())
    log("CSV data loaded.")

    if 'case_studies' in _store:
        case_studies = _store['case_studies']
        log("case_studies loaded from store.")
    else:
        log("WARNING: store predates case_studies persistence. Rebuilding.")
        X_test_tensor_fb = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        X_static_t_fb    = torch.tensor(X_static_test, dtype=torch.float32).to(device) \
                           if X_static_test is not None else None
        with torch.no_grad():
            y_probs_ref_fb = model_gru(X_test_tensor_fb,
                                       X_static_t_fb).cpu().numpy().flatten()
        case_studies = _build_case_studies(labels, y_test, pid_test, y_probs_ref_fb,gt_ntf, feature_names, n_tokens)
        _store['case_studies'] = case_studies
        _store['case_indices'] = list(case_studies.values())
        with open(STORE_PATH, 'wb') as f:
            pickle.dump(_store, f)
        log("Store updated with case_studies.")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DERIVED VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
log_section('Part B: XAI Computation')

T          = n_visits
F          = len(feature_names)
fnames_seq = feature_names[:n_tokens]

X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
X_static_t    = torch.tensor(X_static_test, dtype=torch.float32).to(device) \
                if X_static_test is not None else None

pid_to_test_index = {pid: i for i, pid in enumerate(pid_test)}
labels_test = labels[labels['patient_id'].isin(pid_test)].copy()
labels_test['test_idx'] = labels_test['patient_id'].map(pid_to_test_index)

log(f"Case studies (fixed): { {t: int(i) for t, i in case_studies.items()} }")

# Ground-truth feature indices — used for targeted verification
idx_icd10        = feature_names.index('ICD_10')
idx_atc5         = feature_names.index('ATC_5')
chronic_feat_ids = [feature_names.index(n) for n in ['ICD_1', 'ICD_2', 'ICD_3']]
gt_indices       = [idx_icd10, idx_atc5] + chronic_feat_ids

case_indices = list(case_studies.values())
case_titles  = list(case_studies.keys())
MODEL_KEYS   = ['gru', 'lstm', 'attn_gru', 'attn_lstm', 'cnn']
MODEL_LABELS = ['GRU', 'LSTM', 'AttnGRU', 'AttnLSTM', 'CNN']

log(f"Ready: {n_tokens} token features + {n_static} static = {F} total, "
    f"{n_visits} visits")
log(f"GT signal features: {[feature_names[i] for i in gt_indices]}")



# ══════════════════════════════════════════════════════════════════════════════
# B1 — KERNELSHAP
# ══════════════════════════════════════════════════════════════════════════════
log_section('B1: KernelSHAP')

ks_save_path = os.path.join(out, 'kernelshap_results.npz')
N            = len(X_seq_test)

# Stale-file guard
if os.path.exists(ks_save_path):
    _ks = np.load(ks_save_path, allow_pickle=True)
    saved_ks_set   = set(_ks['case_indices'].tolist())
    current_ks_set = set(case_indices)
    if saved_ks_set != current_ks_set:
        log(f"  KernelSHAP: case_indices changed — deleting and recomputing.")
        os.remove(ks_save_path)
    else:
        log("KernelSHAP: coverage OK — skipping.")

if not os.path.exists(ks_save_path):

    # Case patients flattened to (N_cases, T * n_tokens)
    X_cases = X_seq_test[case_indices]
    X_flat  = X_cases.reshape(len(case_indices), T * n_tokens)

    # Background: 300 raw samples from the test set, no kmeans compression.
    # Raw binary samples are used rather than cluster centroids because
    # centroids produce non-integer reference values that are semantically
    # meaningless for binary codes and can distort marginalisation.
    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(N, size=min(300, N), replace=False)
    bg_flat = X_seq_test[bg_idx].reshape(len(bg_idx), T * n_tokens)
    log(f"  Background size: {bg_flat.shape[0]} raw samples "
        f"(shape {bg_flat.shape})")

    shap_ntf_dict = {}
    for name, model in [('gru',       model_gru),
                        ('lstm',      model_lstm),
                        ('cnn',       model_cnn),
                        ('attn_gru',  model_attn_gru),
                        ('attn_lstm', model_attn_lstm)]:
        log(f"  Running KernelSHAP — {name.upper()}...")

        # Build a fresh explainer per model with the raw background
        explainer = shap.KernelExplainer(
            utils.make_predict_fn(model, T, n_tokens, device,
                                  X_static_np=X_static_test),
            bg_flat, link='identity')

        # nsamples=2048 for better convergence over the high-dimensional
        # flattened input (T * n_tokens features)
        sv = explainer.shap_values(X_flat, nsamples=2048, l1_reg=0,
                                   silent=False)
        if isinstance(sv, list):
            sv = sv[0]

        shap_ntf_dict[name] = sv.reshape(
            len(case_indices), T, n_tokens).astype(np.float32)
        log(f"    Done — shape {shap_ntf_dict[name].shape}")

        # Evaluate against ground truth
        eval_df = evaluate_attribution(
            attr_ntf      = shap_ntf_dict[name],
            gt_ntf        = gt_ntf,
            pid_test      = np.array([pid_test[i] for i in case_indices]),
            pheno_labels  = labels,
            feature_names = feature_names,
            n_tokens      = n_tokens,
        )
        agg_df = aggregate_metrics(eval_df)
        log(f"    KernelSHAP {name.upper()} — aggregated metrics:")
        log(agg_df.to_string(index=False))

    np.savez(ks_save_path,
             shap_ntf_gru=shap_ntf_dict['gru'],
             shap_ntf_lstm=shap_ntf_dict['lstm'],
             shap_ntf_cnn=shap_ntf_dict['cnn'],
             shap_ntf_attn_gru=shap_ntf_dict['attn_gru'],
             shap_ntf_attn_lstm=shap_ntf_dict['attn_lstm'],
             case_indices=np.array(case_indices),
             case_titles=np.array(case_titles))
    log(f"Saved: kernelshap_results.npz")

log_section('B1: KernelSHAP')



# ══════════════════════════════════════════════════════════════════════════════
# B2 — TIMESHAP  
# ══════════════════════════════════════════════════════════════════════════════
log_section('B2: TimeSHAP')

from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level

seq_feature_names = feature_names[:n_tokens]
average_event     = pd.DataFrame(
    X_seq_test.reshape(-1, n_tokens).mean(axis=0).reshape(1, -1),
    columns=seq_feature_names)

pruning_dict     = {'tol': 0.025}
event_dict_local = {'rs': 42, 'nsamples': 32000}
feat_dict_local  = {'rs': 42, 'nsamples': 32000}
cell_dict        = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 15, 'top_x_events': T}

for model_name, model in [('gru', model_gru), ('lstm', model_lstm),
                           ('attn_gru', model_attn_gru), ('attn_lstm', model_attn_lstm),
                           ('cnn', model_cnn)]:
    ts_save_path = os.path.join(out, f'timeshap_cell_ntf_{model_name}.npz')

    if os.path.exists(ts_save_path):
        _ts      = np.load(ts_save_path)
        _ts_ntf  = _ts[f'ts_{model_name}_ntf']
        _missing = {i for i in case_indices if _ts_ntf[i].sum() == 0}
        if _missing:
            log(f"  TimeSHAP {model_name}: missing entries — recomputing.")
            os.remove(ts_save_path)
        else:
            log(f"  TimeSHAP {model_name}: coverage OK — skipping.")
            continue

    log(f"  TimeSHAP — {model_name}")
    model.eval()
    predict_fn  = make_timeshap_predict_fn(model, X_static_test, device)
    ts_cell_ntf = {}

    for case_title, p_idx in case_studies.items():
        test_idx    = int(p_idx)
        instance_3d = X_seq_test[test_idx:test_idx+1]
        log(f"    Computing — {case_title} (test_idx {test_idx})")
        pruning_data, pruning_idx = local_pruning(
            predict_fn, instance_3d, pruning_dict, average_event,
            entity_uuid=test_idx, entity_col='patient_id', verbose=False)
        event_data = local_event(predict_fn, instance_3d, event_dict_local,
            entity_uuid=test_idx, entity_col='patient_id',
            baseline=average_event, pruned_idx=pruning_idx)
        feat_data  = local_feat(predict_fn, instance_3d, feat_dict_local,
            entity_uuid=test_idx, entity_col='patient_id',
            baseline=average_event, pruned_idx=pruning_idx)
        cell_data  = local_cell_level(predict_fn, instance_3d, cell_dict,
            event_data=event_data, feat_data=feat_data,
            entity_uuid=test_idx, entity_col='patient_id',
            baseline=average_event, pruned_idx=pruning_idx)
        cell_ntf = np.zeros((T, n_tokens), dtype=np.float32)
        for _, row in cell_data.iterrows():
            fname = row['Feature']; event = row['Event']
            if event == 'Other Events' or fname == 'Other Features':
                continue
            try:
                f_idx = (int(fname.replace('Feature ', ''))
                         if isinstance(fname, str) and fname.startswith('Feature ')
                         else seq_feature_names.index(fname)
                         if fname in seq_feature_names else None)
                if f_idx is None: continue
                t_abs = T + int(str(event).replace('Event ', ''))
                if 0 <= t_abs < T and 0 <= f_idx < n_tokens:
                    cell_ntf[t_abs, f_idx] = row['Shapley Value']
            except (ValueError, IndexError):
                continue
        ts_cell_ntf[test_idx] = cell_ntf
        log(f"      Non-zero entries: {(cell_ntf != 0).sum()}")

    ts_model_ntf = np.zeros((len(X_seq_test), T, n_tokens), dtype=np.float32)
    for test_idx, cell_arr in ts_cell_ntf.items():
        ts_model_ntf[test_idx] = cell_arr

    # Evaluate TimeSHAP against ground truth (case patients only)
    eval_df = evaluate_attribution(
        attr_ntf      = ts_model_ntf[case_indices],
        gt_ntf        = gt_ntf,
        pid_test      = np.array([pid_test[i] for i in case_indices]),
        pheno_labels  = labels,
        feature_names = feature_names,
        n_tokens      = n_tokens,
    )
    agg_df = aggregate_metrics(eval_df)
    log(f"  TimeSHAP {model_name.upper()} — aggregated metrics:")
    log(agg_df.to_string(index=False))

    np.savez(ts_save_path, **{f'ts_{model_name}_ntf': ts_model_ntf})
    log(f"    Saved: timeshap_cell_ntf_{model_name}.npz  shape={ts_model_ntf.shape}")

log_section('B2: TimeSHAP')

# ══════════════════════════════════════════════════════════════════════════════
# B3 — WINDOWSHAP  
# ══════════════════════════════════════════════════════════════════════════════
log_section('B3: WindowSHAP')

import windowshap.windowshap as ws
import shap.explainers._kernel as _shap_kernel
if not getattr(_shap_kernel.Kernel.solve, '_is_patched', False):
    _original_solve = _shap_kernel.Kernel.solve
    def _patched_solve(self, fraction_evaluated, dim):
        saved = self.l1_reg; self.l1_reg = 0
        result = _original_solve(self, fraction_evaluated, dim)
        self.l1_reg = saved; return result
    _patched_solve._is_patched = True
    _shap_kernel.Kernel.solve  = _patched_solve
    log("  Kernel.solve patched")

all_case_patients = [int(p) for p in case_studies.values()]
tp_sample  = np.array(list(dict.fromkeys(all_case_patients)))
tp_pos_map = {int(p): i for i, p in enumerate(tp_sample)}
n_test_ws  = len(tp_sample)
rng        = np.random.default_rng(42)
bg_idx_ws  = rng.choice(len(X_seq_test), size=100, replace=False)
B_ts_small    = X_seq_test[bg_idx_ws[:20]]
test_ts_small = X_seq_test[tp_sample]

for model_name, model in [('gru', model_gru), ('lstm', model_lstm),
                           ('attn_gru', model_attn_gru), ('attn_lstm', model_attn_lstm),
                           ('cnn', model_cnn)]:
    ws_save_path = os.path.join(out, f'windowshap_{model_name}.npz')
    if os.path.exists(ws_save_path):
        log(f"  WindowSHAP {model_name}: already on disk — skipping.")
        continue
    log(f"  WindowSHAP — {model_name}")
    model.eval()
    predict_fn_ws = make_windowshap_predict_fn(model, X_static_test, device)
    model_ws      = ModelWrapper(predict_fn_ws)
    log("    StationaryWindowSHAP...")
    stat_ntf = np.array(ws.StationaryWindowSHAP(
        model=model_ws, window_len=5, B_ts=B_ts_small,
        test_ts=test_ts_small, model_type='lstm').shap_values(num_output=1))
    log("    SlidingWindowSHAP...")
    slid_ntf = np.array(ws.SlidingWindowSHAP(
        model=model_ws, stride=1, window_len=5, B_ts=B_ts_small,
        test_ts=test_ts_small, model_type='lstm').shap_values(num_output=1))
    log("    DynamicWindowSHAP...")
    dyn_list = []
    for i in range(test_ts_small.shape[0]):
        dyn_list.append(np.array(ws.DynamicWindowSHAP(
            model=model_ws, delta=0.1, n_w=5, B_ts=B_ts_small,
            test_ts=test_ts_small[i:i+1], model_type='lstm').shap_values(num_output=1)))
        log(f"      {i+1}/{n_test_ws} done")
    dyn_ntf = np.concatenate(dyn_list, axis=0)

    # Evaluate WindowSHAP variants against ground truth
    for variant_name, variant_ntf in [('stationary', stat_ntf),
                                       ('sliding',    slid_ntf),
                                       ('dynamic',    dyn_ntf)]:
        eval_df = evaluate_attribution(
            attr_ntf      = variant_ntf,
            gt_ntf        = gt_ntf,
            pid_test      = np.array([pid_test[i] for i in tp_sample]),
            pheno_labels  = labels,
            feature_names = feature_names,
            n_tokens      = n_tokens,
        )
        agg_df = aggregate_metrics(eval_df)
        log(f"  WindowSHAP {variant_name} {model_name.upper()} — aggregated metrics:")
        log(agg_df.to_string(index=False))

    np.savez(ws_save_path,
             stat_ntf=stat_ntf, slid_ntf=slid_ntf, dyn_ntf=dyn_ntf,
             tp_ws=tp_sample,
             tp_pos_map_keys=np.array(list(tp_pos_map.keys())),
             tp_pos_map_values=np.array(list(tp_pos_map.values())))
    log(f"    Saved: windowshap_{model_name}.npz  stat={stat_ntf.shape}")

log_section('B3: WindowSHAP')

# ══════════════════════════════════════════════════════════════════════════════
# B4 — INTEGRATED GRADIENTS
# ══════════════════════════════════════════════════════════════════════════════
log_section('B4: Integrated Gradients')

ig_save_path = os.path.join(out, 'ig_arrays.npz')
if not os.path.exists(ig_save_path):
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        log("  Computing IG (GRU)...")
        ig_gru_ntf       = utils.compute_ig_ntf(model_gru,       X_seq_test, n_steps=64, X_static_np=X_static_test)
        log("  Computing IG (LSTM)...")
        ig_lstm_ntf      = utils.compute_ig_ntf(model_lstm,      X_seq_test, n_steps=64, X_static_np=X_static_test)
        log("  Computing IG (AttnGRU)...")
        ig_attn_gru_ntf  = utils.compute_ig_ntf(model_attn_gru,  X_seq_test, n_steps=64, X_static_np=X_static_test)
        log("  Computing IG (AttnLSTM)...")
        ig_attn_lstm_ntf = utils.compute_ig_ntf(model_attn_lstm, X_seq_test, n_steps=64, X_static_np=X_static_test)
        log("  Computing IG (CNN)...")
        ig_cnn_ntf       = utils.compute_ig_ntf(model_cnn,       X_seq_test, n_steps=64, X_static_np=X_static_test)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    # Evaluate IG against ground truth
    for name, ig_ntf in [('gru',       ig_gru_ntf),
                          ('lstm',      ig_lstm_ntf),
                          ('attn_gru',  ig_attn_gru_ntf),
                          ('attn_lstm', ig_attn_lstm_ntf),
                          ('cnn',       ig_cnn_ntf)]:
        eval_df = evaluate_attribution(
            attr_ntf      = ig_ntf,
            gt_ntf        = gt_ntf,
            pid_test      = pid_test,
            pheno_labels  = labels,
            feature_names = feature_names,
            n_tokens      = n_tokens,
        )
        agg_df = aggregate_metrics(eval_df)
        log(f"  IG {name.upper()} — aggregated metrics:")
        log(agg_df.to_string(index=False))

    np.savez(ig_save_path,
             ig_gru_ntf=ig_gru_ntf,           ig_lstm_ntf=ig_lstm_ntf,
             ig_attn_gru_ntf=ig_attn_gru_ntf, ig_attn_lstm_ntf=ig_attn_lstm_ntf,
             ig_cnn_ntf=ig_cnn_ntf)
    log("Saved: ig_arrays.npz")
else:
    log("IG already on disk — skipping.")

log_section('B4: Integrated Gradients')

# ══════════════════════════════════════════════════════════════════════════════
# B5 — EXPECTED GRADIENTS
# ══════════════════════════════════════════════════════════════════════════════
log_section('B5: Expected Gradients')

eg_save_path = os.path.join(out, 'eg_arrays.npz')
if not os.path.exists(eg_save_path):
    log("  Computing EG (GRU)...")
    _, eg_gru_ntf,       base_gru       = utils.compute_eg_rnn(model_gru,       X_seq_train, X_seq_test, device=device, background_size=128, nsamples=200, X_static_np=X_static_test)
    log("  Computing EG (LSTM)...")
    _, eg_lstm_ntf,      base_lstm      = utils.compute_eg_rnn(model_lstm,      X_seq_train, X_seq_test, device=device, background_size=128, nsamples=200, X_static_np=X_static_test)
    log("  Computing EG (AttnGRU)...")
    _, eg_attn_gru_ntf,  base_attn_gru  = utils.compute_eg_rnn(model_attn_gru,  X_seq_train, X_seq_test, device=device, background_size=128, nsamples=200, X_static_np=X_static_test)
    log("  Computing EG (AttnLSTM)...")
    _, eg_attn_lstm_ntf, base_attn_lstm = utils.compute_eg_rnn(model_attn_lstm, X_seq_train, X_seq_test, device=device, background_size=128, nsamples=200, X_static_np=X_static_test)
    log("  Computing EG (CNN)...")
    _, eg_cnn_ntf,       base_cnn       = utils.compute_eg_rnn(model_cnn,       X_seq_train, X_seq_test, device=device, background_size=128, nsamples=200, X_static_np=X_static_test)

    # Evaluate EG against ground truth
    for name, eg_ntf in [('gru',       eg_gru_ntf),
                          ('lstm',      eg_lstm_ntf),
                          ('attn_gru',  eg_attn_gru_ntf),
                          ('attn_lstm', eg_attn_lstm_ntf),
                          ('cnn',       eg_cnn_ntf)]:
        eval_df = evaluate_attribution(
            attr_ntf      = eg_ntf,
            gt_ntf        = gt_ntf,
            pid_test      = pid_test,
            pheno_labels  = labels,
            feature_names = feature_names,
            n_tokens      = n_tokens,
        )
        agg_df = aggregate_metrics(eval_df)
        log(f"  EG {name.upper()} — aggregated metrics:")
        log(agg_df.to_string(index=False))

    base_val_gru       = utils._scalar_base_value(base_gru)
    base_val_lstm      = utils._scalar_base_value(base_lstm)
    base_val_attn_gru  = utils._scalar_base_value(base_attn_gru)
    base_val_attn_lstm = utils._scalar_base_value(base_attn_lstm)
    base_val_cnn       = utils._scalar_base_value(base_cnn)

    np.savez(eg_save_path,
             eg_gru_ntf=eg_gru_ntf,           eg_lstm_ntf=eg_lstm_ntf,
             eg_attn_gru_ntf=eg_attn_gru_ntf, eg_attn_lstm_ntf=eg_attn_lstm_ntf,
             eg_cnn_ntf=eg_cnn_ntf,
             base_vals=np.array([base_val_gru, base_val_lstm,
                                 base_val_attn_gru, base_val_attn_lstm,
                                 base_val_cnn]))
    log("Saved: eg_arrays.npz")
else:
    log("EG already on disk — skipping.")

log_section('B5: Expected Gradients')
