"""
non_baseline_compute.py
=======================
Runs the full compute pipeline (Part A + Part B XAI) for all non-baseline
scenarios, one after the other.

Each scenario gets its own output directory:
    OUTPUT_DIR/{scenario_name}/

All save/load guards are preserved — already-computed files are skipped,
so the script is safe to resume after interruption.

Run in tmux:
    python3 non_baseline_compute.py 2>&1 | tee non_baseline_compute.log

Tail the log:
    tail -f non_baseline_compute.log

Detach: Ctrl+b, then d
Reattach: tmux attach -t non_baseline
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
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

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
# PATHS & SCENARIOS
# Updated tags match new R simulation naming convention:
# N{patients}_V{visits}_trig{p}_esc{p}_bg{p}_corr{r}_ac{r}
# ══════════════════════════════════════════════════════════════════════════════
SIM_DIR    = "/home/hoepler/eICU_Real_World_Example/sim_outputs_advanced/"
OUTPUT_DIR = "/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/"

NON_BASELINE_SCENARIOS = [
    ("high_noise",       "N3000_V20_trig0.15_esc0.10_bg0.050_corr0.00_ac0.00"),
    ("high_correlation", "N3000_V20_trig0.15_esc0.10_bg0.010_corr0.30_ac0.40"),
    ("short_history",    "N3000_V10_trig0.15_esc0.10_bg0.010_corr0.00_ac0.00"),
]

device = torch.device('cpu')

# ══════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH HELPERS (shared with baseline_compute.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth(sim_dir: str, tag: str) -> dict:
    """
    Load flat ground-truth Shapley weight matrices from the R simulation.

    Analytical Shapley weights:
      - Trigger ICD_10 at t_trig / ATC_5 at t_trig+1 : 5.0 each
      - Chronic ICD_1/2/3 per active occurrence       : 0.5 each
      - Escalation-window active ICDs                 : 2.5 / n_active_late
      - All other triples                             : 0.0
    """
    icd_flat = pd.read_csv(
        os.path.join(sim_dir, f"truth_icd_flat_{tag}.csv"))
    atc_flat = pd.read_csv(
        os.path.join(sim_dir, f"truth_atc_flat_{tag}.csv"))
    pheno    = pd.read_csv(
        os.path.join(sim_dir, f"phenotype_labels_gt_{tag}.csv"))

    n_patients = icd_flat['patient_id'].nunique()
    n_visits   = icd_flat['visit'].nunique()
    icd_cols   = [c for c in icd_flat.columns if c.startswith('ICD_')]
    atc_cols   = [c for c in atc_flat.columns if c.startswith('ATC_')]

    icd_weights = (icd_flat[icd_cols].values
                   .reshape(n_patients, n_visits, len(icd_cols))
                   .astype(np.float32))
    atc_weights = (atc_flat[atc_cols].values
                   .reshape(n_patients, n_visits, len(atc_cols))
                   .astype(np.float32))

    log(f"  GT loaded: ICD {icd_weights.shape}, ATC {atc_weights.shape}")
    return dict(
        icd_weights  = icd_weights,
        atc_weights  = atc_weights,
        pheno_labels = pheno,
        n_patients   = n_patients,
        n_visits     = n_visits,
        n_icd        = len(icd_cols),
        n_atc        = len(atc_cols),
    )


def build_gt_tensor(gt: dict, feature_names: list, n_tokens: int) -> np.ndarray:
    """
    Align ICD and ATC weight arrays to the token vocabulary used by the models.
    Returns (N_patients, N_visits, n_tokens) float32 array.
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

    log(f"  GT tensor: shape={gt_ntf.shape}, "
        f"non-zero={( gt_ntf != 0).sum():,}")
    return gt_ntf


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_attribution(
    attr_ntf:      np.ndarray,
    gt_ntf:        np.ndarray,
    pid_test:      np.ndarray,
    pheno_labels:  pd.DataFrame,
    feature_names: list,
    n_tokens:      int,
) -> pd.DataFrame:
    """
    Compute attribution recovery metrics per patient against analytical
    ground-truth Shapley weights.

    Metrics:
      spearman_signal  : Spearman rho over signal positions (gt > 0)
      spearman_noise   : Spearman rho over noise positions  (gt == 0)
      auprc            : AUPRC with binarised gt as labels
      nmae             : Normalised MAE (both vectors normalised to sum=1)
      top_k_precision  : Precision at k=n_signal_positions
    """
    records = []
    for i, pid in enumerate(pid_test):
        gt_pat = gt_ntf[int(pid) - 1, :, :n_tokens].ravel()
        at_pat = np.abs(attr_ntf[i,   :, :n_tokens]).ravel()

        signal_mask = gt_pat > 0
        noise_mask  = ~signal_mask
        n_signal    = signal_mask.sum()
        n_noise     = noise_mask.sum()

        row = pheno_labels.loc[pheno_labels['patient_id'] == pid]
        group = row['group_gt'].values[0] if len(row) > 0 else 'unknown'
        result = {'patient_id': int(pid), 'group_gt': group}

        if n_signal >= 2:
            rho, _ = spearmanr(at_pat[signal_mask], gt_pat[signal_mask])
            result['spearman_signal'] = float(rho) if not np.isnan(rho) else np.nan
        else:
            result['spearman_signal'] = np.nan

        if n_noise >= 2:
            rho, _ = spearmanr(at_pat[noise_mask], gt_pat[noise_mask])
            result['spearman_noise'] = float(rho) if not np.isnan(rho) else np.nan
        else:
            result['spearman_noise'] = np.nan

        if n_signal > 0 and n_noise > 0:
            result['auprc'] = float(
                average_precision_score(signal_mask.astype(int), at_pat))
        else:
            result['auprc'] = np.nan

        at_sum = at_pat.sum()
        gt_sum = gt_pat.sum()
        if at_sum > 0 and gt_sum > 0:
            result['nmae'] = float(
                np.mean(np.abs(at_pat / at_sum - gt_pat / gt_sum)))
        else:
            result['nmae'] = np.nan

        if n_signal > 0:
            top_k = np.argsort(at_pat)[::-1][:int(n_signal)]
            result['top_k_precision'] = float(signal_mask[top_k].mean())
        else:
            result['top_k_precision'] = np.nan

        records.append(result)
    return pd.DataFrame(records)


def aggregate_metrics(
    eval_df:     pd.DataFrame,
    n_bootstrap: int = 1000,
    seed:        int = 42,
) -> pd.DataFrame:
    """Mean + 95% bootstrap CI per (group_gt, metric)."""
    rng     = np.random.default_rng(seed)
    metrics = ['spearman_signal', 'spearman_noise', 'auprc',
               'nmae', 'top_k_precision']
    rows    = []
    for group in sorted(eval_df['group_gt'].unique()):
        sub = eval_df[eval_df['group_gt'] == group]
        for metric in metrics:
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            boot = np.array([
                np.mean(rng.choice(vals, size=len(vals), replace=True))
                for _ in range(n_bootstrap)
            ])
            rows.append(dict(
                group_gt = group, metric = metric, n = len(vals),
                mean     = round(float(np.mean(vals)), 4),
                ci_lower = round(float(np.percentile(boot, 2.5)),  4),
                ci_upper = round(float(np.percentile(boot, 97.5)), 4),
            ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — build case_studies
# ══════════════════════════════════════════════════════════════════════════════
def _build_case_studies(labels, y_test, pid_test, y_probs_ref,
                        gt_ntf, feature_names, n_tokens):

    pid_to_test_index = {pid: i for i, pid in enumerate(pid_test)}
    labels_test = labels[labels['patient_id'].isin(pid_test)].copy()
    labels_test['test_idx'] = labels_test['patient_id'].map(pid_to_test_index)

    trigger_ids    = {feature_names.index('ICD_10'),
                      feature_names.index('ATC_5')}
    chronic_ids    = {feature_names.index(f)
                      for f in ['ICD_1', 'ICD_2', 'ICD_3']}
    escalation_set = set(range(n_tokens)) - trigger_ids - chronic_ids

    def _signal_strength(test_idx, mode):
        pid    = int(pid_test[test_idx])
        gt_pat = gt_ntf[pid - 1]
        if mode == 'trigger':
            return y_probs_ref[test_idx]
        elif mode == 'chronic':
            return sum((gt_pat[:, j] != 0).sum() for j in chronic_ids)
        elif mode == 'escalation':
            return sum((gt_pat[:, j] != 0).sum() for j in escalation_set)
        else:
            return (gt_pat != 0).sum()

    case_studies = {}
    for group, title, mode in [
        ('trigger_only',       'Pure Trigger',               'trigger'),
        ('chronic_only',       'Pure Chronic',               'chronic'),
        ('escalation_only',    'Pure Late Escalation',       'escalation'),
        ('trigger+chronic',    'Mixed: Trigger + Chronic',   'mixed'),
        ('trigger+escalation', 'Mixed: Trigger + Escalation','mixed'),
        ('escalation+chronic', 'Mixed: Chronic + Escalation','mixed'),
        ('all_three',          'Mixed: All Three',           'mixed'),
    ]:
        cand = labels_test.loc[
            labels_test['group_gt'] == group, 'test_idx'
        ].dropna().astype(int).values
        cand = cand[y_test[cand] == 1]
        if len(cand) == 0:
            continue
        scores = np.array([_signal_strength(i, mode) for i in cand])
        best   = cand[np.argmax(scores)]
        case_studies[title] = best

    fp = np.where((y_test == 0) & (y_probs_ref > 0.6))[0]
    p  = pick_best(fp, y_probs_ref)
    if p is not None:
        case_studies['False Positive'] = p

    return case_studies

# ══════════════════════════════════════════════════════════════════════════════
# PER-SCENARIO FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(scenario_name, tag):

    store_path = os.path.join(OUTPUT_DIR, f'scenario_store_{scenario_name}.pkl')
    out        = os.path.join(OUTPUT_DIR, scenario_name)
    os.makedirs(out, exist_ok=True)
    log(f"Output dir: {out}")

    # ── PART A — DATA LOADING & MODEL TRAINING ────────────────────────────────
    if not os.path.exists(store_path):
        log_section(f'[{scenario_name}] Part A: Training')

        df_events = pd.read_csv(os.path.join(SIM_DIR, f"ehr_events_{tag}.csv"))
        df_static = pd.read_csv(os.path.join(SIM_DIR, f"static_data_{tag}.csv"))
        labels    = pd.read_csv(os.path.join(SIM_DIR, f"phenotype_labels_gt_{tag}.csv"))

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

        X_train_flat = np.concatenate(
            [X_seq_train.sum(axis=1), X_static_train], axis=1)
        X_test_flat  = np.concatenate(
            [X_seq_test.sum(axis=1),  X_static_test],  axis=1)

        log(f"Patients: {n_patients}  Visits: {n_visits}  Tokens: {n_tokens}")

        # Load ground truth and build aligned tensor
        gt     = load_ground_truth(SIM_DIR, tag)
        gt_ntf = build_gt_tensor(gt, feature_names, n_tokens)

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
                model, X_seq_train, y_train, X_seq_test, y_test,
                save_path=path,
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
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=3105)
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
        log(f"Test AUC: { {k: round(v,4) for k,v in perf.items() if 'test' in k} }")

        X_test_tensor_a = torch.tensor(
            X_seq_test, dtype=torch.float32).to(device)
        X_static_t_a    = torch.tensor(
            X_static_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_probs_ref_a = model_gru(
                X_test_tensor_a, X_static_t_a).cpu().numpy().flatten()

        case_studies = _build_case_studies(labels, y_test, pid_test, y_probs_ref_a,gt_ntf, feature_names, n_tokens)
        log(f"Case studies: { {t: int(i) for t, i in case_studies.items()} }")

        with open(store_path, 'wb') as f:
            pickle.dump({
                'hist_gru':      hist_gru,      'hist_lstm':      hist_lstm,
                'hist_attn_gru': hist_attn_gru, 'hist_attn_lstm': hist_attn_lstm,
                'hist_cnn':      hist_cnn,
                'perf':          perf,
                'feature_names': feature_names, 'n_tokens':  n_tokens,
                'n_visits':      n_visits,      'n_static':  n_static,
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
        log(f"Store saved: {store_path}")
        log_section(f'[{scenario_name}] Part A: Training')

    else:
        log("Store already exists — loading from disk, skipping Part A.")
        log_section(f'[{scenario_name}] Part A: Training')
        log_section(f'[{scenario_name}] Part A: Training')

        with open(store_path, 'rb') as f:
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
        perf           = _store['perf']

        # Load ground truth — rebuild from CSV if store predates this change
        if 'gt_ntf' in _store:
            gt_ntf = _store['gt_ntf']
            log("Ground truth loaded from store.")
        else:
            log("WARNING: store predates gt_ntf — loading from CSV and re-saving.")
            n_tokens_tmp = _store.get('n_tokens', len(feature_names) - n_static)
            gt     = load_ground_truth(SIM_DIR, tag)
            gt_ntf = build_gt_tensor(gt, feature_names, n_tokens_tmp)
            _store['gt_ntf'] = gt_ntf
            with open(store_path, 'wb') as f:
                pickle.dump(_store, f)
            log("Store updated with gt_ntf.")

        n_patients = int(tag.split('_')[0][1:])
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
            m.load_state_dict(
                torch.load(os.path.join(out, fname), map_location=device))
            m.to(device).eval()
        log("Models loaded from disk.")

        if X_static_train is not None:
            X_train_flat = np.concatenate(
                [X_seq_train.sum(axis=1), X_static_train], axis=1)
            X_test_flat  = np.concatenate(
                [X_seq_test.sum(axis=1),  X_static_test],  axis=1)
        else:
            X_train_flat = X_seq_train.sum(axis=1)
            X_test_flat  = X_seq_test.sum(axis=1)

        lr_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=3105)
        lr_model.fit(X_train_flat, y_train)
        rf_model.fit(X_train_flat, y_train)
        log("LR and RF refitted.")

        df_events = pd.read_csv(os.path.join(SIM_DIR, f"ehr_events_{tag}.csv"))
        df_static = pd.read_csv(os.path.join(SIM_DIR, f"static_data_{tag}.csv"))
        labels    = pd.read_csv(
            os.path.join(SIM_DIR, f"phenotype_labels_gt_{tag}.csv"))
        tokens    = sorted(df_events['token'].unique())
        log("CSV data loaded.")

        if 'case_studies' in _store:
            case_studies = _store['case_studies']
            log("case_studies loaded from store.")
        else:
            log("WARNING: store predates case_studies — rebuilding and re-saving.")
            X_test_tensor_fb = torch.tensor(
                X_seq_test, dtype=torch.float32).to(device)
            X_static_t_fb    = torch.tensor(
                X_static_test, dtype=torch.float32).to(device) \
                               if X_static_test is not None else None
            with torch.no_grad():
                y_probs_ref_fb = model_gru(
                    X_test_tensor_fb,
                    X_static_t_fb).cpu().numpy().flatten()
            case_studies = _build_case_studies(labels, y_test, pid_test, y_probs_ref_fb,gt_ntf, feature_names, n_tokens)
            _store['case_studies'] = case_studies
            _store['case_indices'] = list(case_studies.values())
            with open(store_path, 'wb') as f:
                pickle.dump(_store, f)
            log("Store updated with case_studies.")

    # ── SHARED DERIVED VARIABLES ──────────────────────────────────────────────
    log_section(f'[{scenario_name}] Part B: XAI Computation')

    T          = n_visits
    F          = len(feature_names)
    fnames_seq = feature_names[:n_tokens]

    X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_static_t    = torch.tensor(X_static_test, dtype=torch.float32).to(device) \
                    if X_static_test is not None else None

    log(f"Case studies (fixed): { {t: int(i) for t, i in case_studies.items()} }")

    case_indices = list(case_studies.values())
    case_titles  = list(case_studies.keys())

    log(f"Ready: {n_tokens} tokens + {n_static} static = {F} total, "
        f"{n_visits} visits")

    # ── B1: KERNELSHAP ────────────────────────────────────────────────────────
    log_section(f'[{scenario_name}] B1: KernelSHAP')

    ks_save_path = os.path.join(out, 'kernelshap_results.npz')
    N            = len(X_seq_test)

    if os.path.exists(ks_save_path):
        ks_check = np.load(ks_save_path, allow_pickle=True)
        if set(ks_check['case_indices'].tolist()) != set(case_indices):
            log("  KernelSHAP: case_indices changed — deleting and recomputing.")
            os.remove(ks_save_path)
        else:
            log("  KernelSHAP: coverage OK — skipping.")

    if not os.path.exists(ks_save_path):
        X_cases = X_seq_test[case_indices]
        X_flat  = X_cases.reshape(len(case_indices), T * n_tokens)
        rng     = np.random.default_rng(42)
        bg_idx  = rng.choice(N, size=min(300, N), replace=False)
        bg_flat = X_seq_test[bg_idx].reshape(len(bg_idx), T * n_tokens)
        log(f"  Background: {bg_flat.shape[0]} raw samples")

        shap_ntf_dict = {}
        for name, model in [('gru',       model_gru),
                             ('lstm',      model_lstm),
                             ('cnn',       model_cnn),
                             ('attn_gru',  model_attn_gru),
                             ('attn_lstm', model_attn_lstm)]:
            log(f"  Running KernelSHAP — {name.upper()}...")
            explainer = shap.KernelExplainer(
                utils.make_predict_fn(model, T, n_tokens, device,
                                      X_static_np=X_static_test),
                bg_flat, link='identity')
            sv = explainer.shap_values(
                X_flat, nsamples=2048, l1_reg=0, silent=False)
            if isinstance(sv, list): sv = sv[0]
            shap_ntf_dict[name] = sv.reshape(
                len(case_indices), T, n_tokens).astype(np.float32)
            log(f"    Done — shape {shap_ntf_dict[name].shape}")

            eval_df = evaluate_attribution(
                attr_ntf      = shap_ntf_dict[name],
                gt_ntf        = gt_ntf,
                pid_test      = np.array([pid_test[i] for i in case_indices]),
                pheno_labels  = labels,
                feature_names = feature_names,
                n_tokens      = n_tokens,
            )
            log(aggregate_metrics(eval_df).to_string(index=False))

        np.savez(ks_save_path,
                 shap_ntf_gru=shap_ntf_dict['gru'],
                 shap_ntf_lstm=shap_ntf_dict['lstm'],
                 shap_ntf_cnn=shap_ntf_dict['cnn'],
                 shap_ntf_attn_gru=shap_ntf_dict['attn_gru'],
                 shap_ntf_attn_lstm=shap_ntf_dict['attn_lstm'],
                 case_indices=np.array(case_indices),
                 case_titles=np.array(case_titles))
        log("Saved: kernelshap_results.npz")

    log_section(f'[{scenario_name}] B1: KernelSHAP')

    # ── B2: TIMESHAP ──────────────────────────────────────────────────────────
    log_section(f'[{scenario_name}] B2: TimeSHAP')

    from timeshap.explainer import (local_pruning, local_event,
                                    local_feat, local_cell_level)

    seq_feature_names = feature_names[:n_tokens]
    average_event     = pd.DataFrame(
        X_seq_test.reshape(-1, n_tokens).mean(axis=0).reshape(1, -1),
        columns=seq_feature_names)

    pruning_dict     = {'tol': 0.025}
    event_dict_local = {'rs': 42, 'nsamples': 32000}
    feat_dict_local  = {'rs': 42, 'nsamples': 32000}
    cell_dict        = {'rs': 42, 'nsamples': 32000,
                        'top_x_feats': 15, 'top_x_events': T}

    for model_name, model in [('gru',       model_gru),
                               ('lstm',      model_lstm),
                               ('attn_gru',  model_attn_gru),
                               ('attn_lstm', model_attn_lstm),
                               ('cnn',       model_cnn)]:
        ts_save_path = os.path.join(
            out, f'timeshap_cell_ntf_{model_name}.npz')

        if os.path.exists(ts_save_path):
            ts_check = np.load(ts_save_path)
            ts_ntf   = ts_check[f'ts_{model_name}_ntf']
            missing  = {i for i in case_indices if ts_ntf[i].sum() == 0}
            if missing:
                log(f"  TimeSHAP {model_name}: missing {missing} — recomputing.")
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
            log(f"    {case_title} (test_idx {test_idx})")
            pruning_data, pruning_idx = local_pruning(
                predict_fn, instance_3d, pruning_dict, average_event,
                entity_uuid=test_idx, entity_col='patient_id', verbose=False)
            event_data = local_event(
                predict_fn, instance_3d, event_dict_local,
                entity_uuid=test_idx, entity_col='patient_id',
                baseline=average_event, pruned_idx=pruning_idx)
            feat_data  = local_feat(
                predict_fn, instance_3d, feat_dict_local,
                entity_uuid=test_idx, entity_col='patient_id',
                baseline=average_event, pruned_idx=pruning_idx)
            cell_data  = local_cell_level(
                predict_fn, instance_3d, cell_dict,
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
                             if isinstance(fname, str)
                             and fname.startswith('Feature ')
                             else seq_feature_names.index(fname)
                             if fname in seq_feature_names else None)
                    if f_idx is None: continue
                    t_abs = T + int(str(event).replace('Event ', ''))
                    if 0 <= t_abs < T and 0 <= f_idx < n_tokens:
                        cell_ntf[t_abs, f_idx] = row['Shapley Value']
                except (ValueError, IndexError):
                    continue
            ts_cell_ntf[test_idx] = cell_ntf
            log(f"      Non-zero: {(cell_ntf != 0).sum()}")

        ts_model_ntf = np.zeros(
            (len(X_seq_test), T, n_tokens), dtype=np.float32)
        for test_idx, cell_arr in ts_cell_ntf.items():
            ts_model_ntf[test_idx] = cell_arr

        eval_df = evaluate_attribution(
            attr_ntf      = ts_model_ntf[case_indices],
            gt_ntf        = gt_ntf,
            pid_test      = np.array([pid_test[i] for i in case_indices]),
            pheno_labels  = labels,
            feature_names = feature_names,
            n_tokens      = n_tokens,
        )
        log(f"  TimeSHAP {model_name.upper()} metrics:")
        log(aggregate_metrics(eval_df).to_string(index=False))

        np.savez(ts_save_path, **{f'ts_{model_name}_ntf': ts_model_ntf})
        log(f"    Saved: timeshap_cell_ntf_{model_name}.npz")

    log_section(f'[{scenario_name}] B2: TimeSHAP')

    # ── B3: WINDOWSHAP ────────────────────────────────────────────────────────
    log_section(f'[{scenario_name}] B3: WindowSHAP')

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

    current_patients = set(tp_pos_map.keys())
    for model_name in ['gru', 'lstm', 'attn_gru', 'attn_lstm', 'cnn']:
        ws_save_path = os.path.join(out, f'windowshap_{model_name}.npz')
        if os.path.exists(ws_save_path):
            d = np.load(ws_save_path)
            saved_patients = set(d['tp_pos_map_keys'].tolist())
            if not current_patients.issubset(saved_patients):
                log(f"  WindowSHAP {model_name}: stale — deleting and recomputing.")
                os.remove(ws_save_path)
            else:
                log(f"  WindowSHAP {model_name}: coverage OK — skipping.")

    for model_name, model in [('gru',       model_gru),
                               ('lstm',      model_lstm),
                               ('attn_gru',  model_attn_gru),
                               ('attn_lstm', model_attn_lstm),
                               ('cnn',       model_cnn)]:
        ws_save_path = os.path.join(out, f'windowshap_{model_name}.npz')
        if os.path.exists(ws_save_path):
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
                test_ts=test_ts_small[i:i+1],
                model_type='lstm').shap_values(num_output=1)))
            log(f"      {i+1}/{n_test_ws} done")
        dyn_ntf = np.concatenate(dyn_list, axis=0)

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
            log(f"  WindowSHAP {variant_name} {model_name.upper()} metrics:")
            log(aggregate_metrics(eval_df).to_string(index=False))

        np.savez(ws_save_path,
                 stat_ntf=stat_ntf, slid_ntf=slid_ntf, dyn_ntf=dyn_ntf,
                 tp_ws=tp_sample,
                 tp_pos_map_keys=np.array(list(tp_pos_map.keys())),
                 tp_pos_map_values=np.array(list(tp_pos_map.values())))
        log(f"    Saved: windowshap_{model_name}.npz")

    log_section(f'[{scenario_name}] B3: WindowSHAP')

    # ── B4: INTEGRATED GRADIENTS ──────────────────────────────────────────────
    log_section(f'[{scenario_name}] B4: Integrated Gradients')

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
            log(f"  IG {name.upper()} metrics:")
            log(aggregate_metrics(eval_df).to_string(index=False))

        np.savez(ig_save_path,
                 ig_gru_ntf=ig_gru_ntf,           ig_lstm_ntf=ig_lstm_ntf,
                 ig_attn_gru_ntf=ig_attn_gru_ntf, ig_attn_lstm_ntf=ig_attn_lstm_ntf,
                 ig_cnn_ntf=ig_cnn_ntf)
        log("Saved: ig_arrays.npz")
    else:
        log("IG already on disk — skipping.")

    log_section(f'[{scenario_name}] B4: Integrated Gradients')

    # ── B5: EXPECTED GRADIENTS ────────────────────────────────────────────────
    log_section(f'[{scenario_name}] B5: Expected Gradients')

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
            log(f"  EG {name.upper()} metrics:")
            log(aggregate_metrics(eval_df).to_string(index=False))

        np.savez(eg_save_path,
                 eg_gru_ntf=eg_gru_ntf,           eg_lstm_ntf=eg_lstm_ntf,
                 eg_attn_gru_ntf=eg_attn_gru_ntf, eg_attn_lstm_ntf=eg_attn_lstm_ntf,
                 eg_cnn_ntf=eg_cnn_ntf,
                 base_vals=np.array([
                     utils._scalar_base_value(base_gru),
                     utils._scalar_base_value(base_lstm),
                     utils._scalar_base_value(base_attn_gru),
                     utils._scalar_base_value(base_attn_lstm),
                     utils._scalar_base_value(base_cnn),
                 ]))
        log("Saved: eg_arrays.npz")
    else:
        log("EG already on disk — skipping.")

    log_section(f'[{scenario_name}] B5: Expected Gradients')

    # ── B6: COUNTERFACTUALS ───────────────────────────────────────────────────
    log_section(f'[{scenario_name}] B6: Counterfactuals')

    def _run_cf(model):
        from utils import run_cf_case_studies
        def dispatcher(x_np):
            s   = torch.tensor(
                X_static_test.mean(axis=0, keepdims=True).astype(np.float32),
                device=device)
            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                return float(
                    model(x_t, s.expand(x_t.shape[0], -1)).reshape(-1).cpu()[0])
        return run_cf_case_studies(
            case_studies=case_studies, X_test=X_seq_test, y_test=y_test,
            pid_test=pid_test, feature_names=feature_names, tokens=tokens,
            predict_proba=dispatcher, threshold=0.5, max_edits=8, mode='token')

    for model_name, model in [('gru',       model_gru),
                               ('lstm',      model_lstm),
                               ('attn_gru',  model_attn_gru),
                               ('attn_lstm', model_attn_lstm),
                               ('cnn',       model_cnn)]:
        cf_save = os.path.join(out, f'cf_edits_{model_name}.csv')
        if os.path.exists(cf_save):
            log(f"  CF {model_name}: already on disk — skipping.")
            continue
        log(f"  Running CF — {model_name}...")
        cf_summary, cf_details = _run_cf(model)
        cf_table = utils.build_cf_edits_table(
            cf_summary, cf_details, threshold=0.5)
        cf_table.to_csv(cf_save, index=False)
        log(f"  Saved: cf_edits_{model_name}.csv  ({len(cf_table)} rows)")

    log_section(f'[{scenario_name}] B6: Counterfactuals')
    log_section(f'[{scenario_name}] Part B: XAI Computation')
    log(f"Scenario {scenario_name} complete. Output: {out}")
    log("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
log(f"Starting non-baseline compute — {len(NON_BASELINE_SCENARIOS)} scenarios")
log(f"Scenarios: {[s for s, _ in NON_BASELINE_SCENARIOS]}")
log("="*60)

for scenario_name, tag in NON_BASELINE_SCENARIOS:
    log("")
    log(f"{'#'*60}")
    log(f"  SCENARIO: {scenario_name}  |  TAG: {tag}")
    log(f"{'#'*60}")
    run_scenario(scenario_name, tag)

log("="*60)
log(f"All {len(NON_BASELINE_SCENARIOS)} scenarios complete.")
log("="*60)