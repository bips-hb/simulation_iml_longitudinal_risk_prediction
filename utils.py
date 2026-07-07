"""
utils.py  –  EHR Prediction & XAI Benchmark
============================================
Single utility module covering:
  1.  Reproducibility
  2.  Model definitions  (GRU, LSTM, LogitWrapper)
  3.  Training & evaluation
  4.  Plotting helpers
  5.  Attention model definitions
  6.  Occlusion-based attribution
  7.  Additional helpers
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim

from captum.attr import IntegratedGradients
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model_zoo import (
    GRUModel, LSTMModel, LogitWrapper,
    AttentionGRUModel, AttentionLSTMModel,
    CNNModel, run_training_pipeline,
)

# ==============================================================================
# 1.  REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def savefig(path: str) -> None:
    """Save the current matplotlib figure to *path* and close it."""
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def pick_best(test_indices, probs):
    """Return the test-row index with the highest predicted probability.

    Returns None if test_indices is empty.
    """
    test_indices = np.array(test_indices, dtype=int)
    if len(test_indices) == 0:
        return None
    return int(test_indices[np.argmax(probs[test_indices])])



# ==============================================================================
# 3.  TRAINING & EVALUATION
# ==============================================================================

def _scalar_base_value(base_vals) -> float:
    """Convert EG base value (from shap.GradientExplainer.expected_value — scalar / list / array) to a single float."""
    if base_vals is None:
        return 0.0
    try:
        return float(np.mean(base_vals))
    except Exception:
        return 0.0


def ensure_eg_NTF(shap_vals, T: int, F: int) -> np.ndarray:
    """Coerce EG output (from shap.GradientExplainer) into shape (N, T, F) regardless of SHAP version."""
    raw = np.array(shap_vals)
    if raw.ndim == 4 and raw.shape[1] == 1:   raw = raw[:, 0, :, :]
    if raw.ndim == 3 and raw.shape[1] == T:    return raw
    if raw.ndim == 2 and raw.shape[1] == T*F:  return raw.reshape(raw.shape[0], T, F)
    if raw.ndim == 4 and raw.shape[-1] == 1:   return raw[:, :, :, 0]
    raise ValueError(f"Unexpected EG output shape {raw.shape}")


def compute_eg_rnn(model, X_train_np, X_explain_np,
                   device=None, background_size=128,
                   nsamples=200, seed=42,
                   X_static_np=None):
    """Run Expected Gradients (EG) via shap.GradientExplainer for a single model.

    Parameters
    ----------
    X_static_np : np.ndarray (N, n_static) or None
        When provided, static is fixed at its test-set mean and passed via
        LogitWrapper so GradientExplainer only sees the sequence input.

    Returns
    -------
    explainer : shap.GradientExplainer
    eg_ntf    : np.ndarray  (N, T, F)  EG attribution values
    base_vals : float  scalar base value
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap model with fixed static mean if static features are present
    if X_static_np is not None:
        s_mean = torch.tensor(
            X_static_np.mean(axis=0, keepdims=True), dtype=torch.float32
        )
        wrapped = LogitWrapper(model, x_static=s_mean).to(device)
    else:
        wrapped = LogitWrapper(model).to(device)
    wrapped.eval()

    rng    = np.random.default_rng(seed)
    idx_bg = rng.choice(len(X_train_np),
                        size=min(background_size, len(X_train_np)),
                        replace=False)

    background = torch.tensor(X_train_np[idx_bg], dtype=torch.float32, device=device)
    X_e        = torch.tensor(X_explain_np,        dtype=torch.float32, device=device)

    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        explainer = shap.GradientExplainer(wrapped, background)
        raw_sv    = explainer.shap_values(X_e, nsamples=nsamples)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    if isinstance(raw_sv, list):
        raw_sv = raw_sv[0]

    base_vals = getattr(explainer, "expected_value", None)
    eg_ntf    = ensure_eg_NTF(raw_sv, T=X_explain_np.shape[1], F=X_explain_np.shape[2])
    return explainer, eg_ntf, base_vals


def global_importance_per_feature(eg_ntf: np.ndarray) -> np.ndarray:
    """Mean absolute EG value per feature, collapsed over patients and time."""
    return np.mean(np.abs(eg_ntf), axis=(0, 1))


def compute_ig_ntf(model, X_explain_np, n_steps=64, batch_size=32,
                    X_static_np=None):
    """Compute Integrated Gradients attributions for sequence features.

    Parameters
    ----------
    X_static_np : np.ndarray (N, n_static) or None
        Static features held fixed during IG computation. When provided,
        the model receives (x_seq, x_static) on every forward call.
    """
    device  = next(model.parameters()).device
    model.eval()

    # IG is computed only w.r.t. sequence input, with static held fixed.
    # We wrap forward_logits so the IG target function only takes x_seq.
    if X_static_np is not None:
        s_mean = torch.tensor(
            X_static_np.mean(axis=0, keepdims=True), dtype=torch.float32, device=device
        )
        def _fwd_logits(x):
            return model.forward_logits(x, s_mean.expand(x.shape[0], -1))
    else:
        def _fwd_logits(x):
            return model.forward_logits(x)

    ig      = IntegratedGradients(_fwd_logits)
    N, T, F = X_explain_np.shape
    ig_ntf  = np.zeros((N, T, F), dtype=np.float32)

    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            inp  = torch.tensor(X_explain_np[start:end],
                                dtype=torch.float32,
                                device=device).requires_grad_(True)
            attr = ig.attribute(inp,
                                baselines=torch.zeros_like(inp),
                                target=0,
                                n_steps=n_steps)
            ig_ntf[start:end] = attr.detach().cpu().numpy()
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    norms  = np.sum(np.abs(ig_ntf), axis=(1, 2), keepdims=True) + 1e-8
    ig_ntf = ig_ntf / norms
    return ig_ntf


def _greedy_cf_remove(x: np.ndarray,
                      token_names: list,
                      predict_proba,
                      threshold: float = 0.5,
                      max_edits: int = 10,
                      mode: str = "token",
                      static_idx=None,
                      base_token_cost: float = 1.0,
                      base_timestep_cost: float = 5.0,
                      time_weight: float = 0.0,
                      allowed_timesteps=None):
    """Greedy cost-aware counterfactual search (remove evidence).

    Selects edits by benefit-per-cost  =  (p_current - p_candidate) / cost(edit).

    Parameters
    ----------
    x                  : (1, T, F) patient sequence
    token_names        : feature name list of length F
    predict_proba      : callable (1,T,F) -> float
    threshold          : target probability to drop below
    max_edits          : greedy budget
    mode               : 'token' (remove one feature) or 'timestep' (zero full visit)
    static_idx         : feature indices that must not be edited
    base_token_cost    : base cost of a single token removal
    base_timestep_cost : base cost of zeroing an entire timestep
    time_weight        : >0 penalises edits to earlier timesteps
    allowed_timesteps  : restrict edits to a subset of timestep indices
    """
    x_cf       = x.copy()
    T, F       = x_cf.shape[1], x_cf.shape[2]
    static_idx = set(static_idx or [])
    allowed_ts = list(allowed_timesteps) if allowed_timesteps is not None else list(range(T))
    p_cur      = predict_proba(x_cf)
    p_before   = p_cur
    edits      = []

    if p_cur < threshold:
        return x_cf, edits, p_before, p_cur

    for step in range(1, max_edits + 1):
        best, best_score, best_p = None, 0.0, p_cur

        if mode == "timestep":
            candidates = [(t, None) for t in allowed_ts]
        else:
            candidates = [(t, j)
                          for t in allowed_ts
                          for j in np.where(x_cf[0, t, :] > 0.5)[0]
                          if j not in static_idx]

        for t, j in candidates:
            x_try = x_cf.copy()
            if mode == "timestep":
                x_try[0, t, :] = 0.0
                for jj in static_idx:
                    x_try[0, t, jj] = x_cf[0, t, jj]
            else:
                x_try[0, t, j] = 0.0

            p_new = predict_proba(x_try)
            gain  = p_cur - p_new
            if gain <= 0:
                continue

            c = base_timestep_cost if mode == "timestep" else base_token_cost
            if time_weight > 0 and T > 1:
                c *= 1.0 + time_weight * (T - 1 - t) / (T - 1)
            score = gain / c

            if score > best_score:
                best_score, best_p, best = score, p_new, (t, j)

        if best is None:
            break

        t, j = best
        if mode == "timestep":
            x_cf[0, t, :] = 0.0
            for jj in static_idx:
                x_cf[0, t, jj] = x[0, t, jj]
            edits.append({"step": step, "type": "timestep_zero",
                           "t": t, "feature": None, "new_p": best_p})
        else:
            x_cf[0, t, j] = 0.0
            edits.append({"step": step, "type": "token_remove",
                           "t": t, "feature": token_names[j],
                           "j": j, "new_p": best_p})
        p_cur = best_p
        if p_cur < threshold:
            break

    return x_cf, edits, p_before, p_cur


def run_cf_case_studies(case_studies: dict,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        pid_test: np.ndarray,
                        feature_names: list,
                        tokens: list,
                        predict_proba,
                        threshold: float = 0.5,
                        max_edits: int = 8,
                        mode: str = "token",
                        time_weight: float = 0.0,
                        allowed_timesteps=None):
    """Run counterfactual search for every case in case_studies.

    Returns
    -------
    summary_df : pd.DataFrame  one row per case
    details    : dict  {case_title: pd.DataFrame of edits}
    """
    static_idx = [len(tokens), len(tokens) + 1]
    rows, details = [], {}

    for title, test_idx in case_studies.items():
        test_idx = int(test_idx)
        x = np.asarray(X_test[test_idx:test_idx+1], dtype=float)

        x_cf, edits, p_before, p_after = _greedy_cf_remove(
            x, feature_names, predict_proba,
            threshold=threshold, max_edits=max_edits, mode=mode,
            static_idx=static_idx, time_weight=time_weight,
            allowed_timesteps=allowed_timesteps,
        )
        details[title] = pd.DataFrame(edits)

        first_edit = None
        if edits:
            e = edits[0]
            first_edit = (f"{e['feature']} @ t={e['t']}"
                          if e.get("type") == "token_remove"
                          else f"timestep t={e['t']}")

        rows.append({"Case":       title,
                     "test_idx":   test_idx,
                     "patient_id": int(pid_test[test_idx]),
                     "y_true":     int(y_test[test_idx]),
                     "p_before":   float(p_before),
                     "p_after":    float(p_after),
                     "delta":      float(p_before - p_after),
                     "crossed":    bool(p_after < threshold),
                     "n_edits":    len(edits),
                     "first_edit": first_edit})

    return pd.DataFrame(rows).sort_values("Case").reset_index(drop=True), details


def build_cf_edits_table(cf_summary: pd.DataFrame,
                          cf_details: dict,
                          threshold: float = 0.5) -> pd.DataFrame:
    """Flatten edit logs into a long DataFrame with step-level probability paths."""
    rows = []
    for _, r in cf_summary.iterrows():
        case     = r["Case"]
        p_prev   = float(r["p_before"])
        edits_df = cf_details.get(case, pd.DataFrame())

        if edits_df is None or edits_df.empty:
            rows.append({"Case":             case,
                          "patient_id":      int(r["patient_id"]),
                          "test_idx":        int(r["test_idx"]),
                          "y_true":          int(r["y_true"]),
                          "edit_step":       None,
                          "timestep":        None,
                          "feature":         None,
                          "p_before_step":   p_prev,
                          "p_after_step":    p_prev,
                          "delta_step":      0.0,
                          "crossed_this_step": False})
            continue

        for _, e in edits_df.iterrows():
            p_after = float(e["new_p"])
            delta   = p_prev - p_after
            crossed = (p_prev >= threshold) and (p_after < threshold)
            rows.append({"Case":             case,
                          "patient_id":      int(r["patient_id"]),
                          "test_idx":        int(r["test_idx"]),
                          "y_true":          int(r["y_true"]),
                          "edit_step":       int(e["step"]),
                          "timestep":        int(e["t"]) if pd.notnull(e.get("t")) else None,
                          "feature":         e.get("feature"),
                          "p_before_step":   p_prev,
                          "p_after_step":    p_after,
                          "delta_step":      delta,
                          "crossed_this_step": bool(crossed)})
            p_prev = p_after

    return (pd.DataFrame(rows)
              .sort_values(["Case", "patient_id", "edit_step"], na_position="last")
              .reset_index(drop=True))


def style_cf_table(df: pd.DataFrame):
    """Pandas Styler: alternating row colours + bold at threshold crossing."""
    case_codes = df["Case"].astype("category").cat.codes

    def _style(row):
        bg   = "#f5f5f5" if case_codes[row.name] % 2 == 0 else "#ffffff"
        bold = "; font-weight: bold" if row.get("crossed_this_step") else ""
        return [f"background-color: {bg}{bold}"] * len(row)

    return df.style.hide(axis="index").apply(_style, axis=1)


# ==============================================================================
# 4.  PLOTTING HELPERS
# ==============================================================================

def plot_global_bar(imp: np.ndarray, feature_names: list,
                    title: str, top_k: int = 20) -> pd.DataFrame:
    """Horizontal bar chart of global mean |EG| per feature."""
    df = (pd.DataFrame({"feature": feature_names, "mean_abs_eg": imp})
            .sort_values("mean_abs_eg", ascending=False)
            .head(top_k)
            .iloc[::-1])

    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"], df["mean_abs_eg"])
    plt.title(title)
    plt.xlabel("Mean |EG|")
    plt.tight_layout()
    plt.show()
    plt.close() 
    return df.iloc[::-1]


def plot_eg_over_time(eg_ntf: np.ndarray, feature_names: list,
                      top_k: int = 8, title: str = "EG over time") -> None:
    """Line plot of mean |EG| per visit for the top-k features."""
    mean_abs   = np.mean(np.abs(eg_ntf), axis=0)
    global_imp = np.mean(np.abs(eg_ntf), axis=(0, 1))
    top_idx    = np.argsort(global_imp)[-top_k:]

    plt.figure(figsize=(10, 6))
    for j in top_idx:
        plt.plot(range(1, mean_abs.shape[0] + 1), mean_abs[:, j],
                 label=feature_names[j])
    plt.xlabel("Visit (1..T)")
    plt.ylabel("Mean |EG|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_eg_beeswarm_all_visits(eg_ntf: np.ndarray, X_np: np.ndarray,
                                feature_names: list, max_display: int = 20) -> None:
    """EG beeswarm treating each (patient, visit) as one observation."""
    N, T, F  = eg_ntf.shape
    eg_flat  = eg_ntf.reshape(N * T, F)
    X_flat   = X_np.reshape(N * T, F)
    exp = shap.Explanation(values=eg_flat, data=X_flat,
                            feature_names=feature_names)
    shap.plots.beeswarm(exp, max_display=max_display)


def plot_patient_aggregated_waterfall(eg_ntf, X_np, feature_names,
                                       patient_idx=2, base_value=0.0,
                                       top_k_agg=20, figsize=(12, 7),
                                       patient_id=None) -> None:
    """Waterfall of EG values summed across all visits."""
    exp = shap.Explanation(
        values=eg_ntf[patient_idx].sum(axis=0),
        base_values=base_value,
        data=X_np[patient_idx].sum(axis=0),
        feature_names=feature_names,
    )
    pid_str = f"  |  Patient ID: {patient_id}" if patient_id is not None else ""
    plt.figure(figsize=figsize)
    shap.plots.waterfall(exp, max_display=top_k_agg, show=False)
    plt.title(f"Test idx: {patient_idx}{pid_str} | Aggregated over all visits",
              fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_patient_signed_eg_heatmap(eg_ntf, feature_names,
                                    patient_idx=2, top_k=20,
                                    top_mode="mean_abs", vmax=None,
                                    figsize=(14, 8), patient_id=None) -> None:
    """Signed EG heatmap (features × visits) for one patient."""
    sv = eg_ntf[patient_idx]
    score_fns = {
        "mean_abs":   lambda s: np.mean(np.abs(s), axis=0),
        "sum_abs":    lambda s: np.sum( np.abs(s), axis=0),
        "sum_signed": lambda s: np.abs(np.sum(s,   axis=0)),
    }
    if top_mode not in score_fns:
        raise ValueError(f"top_mode must be one of {list(score_fns)}")

    top_idx = np.argsort(score_fns[top_mode](sv))[::-1][:top_k]
    mat     = sv[:, top_idx].T
    vmax    = vmax or (np.max(np.abs(mat)) + 1e-12)
    T       = eg_ntf.shape[1]

    pid_str = f"  |  Patient ID: {patient_id}" if patient_id is not None else ""
    plt.figure(figsize=figsize)
    sns.heatmap(
        mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=[f"V{t+1}" for t in range(T)],
        yticklabels=[feature_names[j] for j in top_idx],
        cbar_kws={"label": "EG value"},
    )
    plt.title(f"Test idx: {patient_idx}{pid_str} | Signed EG heatmap (top {top_k})",
              fontsize=16, fontweight="bold")
    plt.xlabel("Visit")
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_patient_cumulative_prediction(eg_ntf, patient_idx=2,
                                        base_value=0.0,
                                        convert_to_probability=True,
                                        patient_id=None) -> None:
    """Cumulative predicted score/probability over visits for one patient."""
    T             = eg_ntf.shape[1]
    visit_contrib = eg_ntf[patient_idx].sum(axis=1)
    cumul_logit   = base_value + np.cumsum(visit_contrib)
    cumul         = expit(cumul_logit) if convert_to_probability else cumul_logit
    ylabel        = "Predicted probability" if convert_to_probability else "Logit"
    baseline_y    = expit(base_value) if convert_to_probability else base_value

    pid_str = f"  |  Patient ID: {patient_id}" if patient_id is not None else ""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, T + 1), cumul, marker="o")
    plt.axhline(baseline_y, linestyle="--", color="gray", label="Baseline")
    plt.xlabel("Visit")
    plt.ylabel(ylabel)
    plt.title(f"Test idx: {patient_idx}{pid_str} | Cumulative prediction over visits",
              fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_feature_cumulative(eg_ntf, feature_names, patient_idx,
                             feature_name: str, base_value: float = 0.0,
                             patient_id=None) -> None:
    """Cumulative EG contribution of a single feature over visits."""
    idx        = feature_names.index(feature_name)
    T          = eg_ntf.shape[1]
    cumulative = np.cumsum(eg_ntf[patient_idx, :, idx])

    pid_str = f"  |  Patient ID: {patient_id}" if patient_id is not None else ""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, T + 1), cumulative, marker="o")
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(f"Cumulative EG over visits\nTest idx: {patient_idx}{pid_str} | {feature_name}",
              fontsize=13, fontweight="bold")
    plt.xlabel("Visit")
    plt.ylabel("Cumulative EG (logit)")
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_eg_case_heatmaps_side_by_side(eg_by_model: dict,
                                        feature_names: list,
                                        case_studies: dict,
                                        model_dict: dict,
                                        X_test_tensor: torch.Tensor,
                                        y_test: np.ndarray,
                                        top_k: int = 15,
                                        normalize: bool = False,
                                        pid_test=None) -> None:
    """Side-by-side EG heatmaps for every case study, all models in one row."""
    n_models = len(model_dict)
    for title, p_idx in case_studies.items():
        fig, axes = plt.subplots(1, n_models,
                                  figsize=(12 * n_models, 8), sharey=False)
        if n_models == 1:
            axes = [axes]
        actual = int(y_test[p_idx])
        pid = int(pid_test[p_idx]) if pid_test is not None else None
        pid_str = f"  |  Patient ID: {pid}" if pid is not None else ""

        mats, vmaxs = {}, {}
        for name in model_dict:
            eg = eg_by_model[name][p_idx].copy()
            if normalize:
                eg /= (np.sum(np.abs(eg)) + 1e-8)
            mats[name]  = eg
            vmaxs[name] = np.max(np.abs(eg))

        shared_vmax = max(vmaxs.values())

        for ax, name in zip(axes, model_dict):
            eg      = mats[name]
            top_idx = np.argsort(np.sum(np.abs(eg), axis=0))[-top_k:][::-1]
            prob    = model_dict[name](X_test_tensor[p_idx:p_idx+1]).item()

            sns.heatmap(
                eg[:, top_idx].T,
                cmap="RdBu_r", center=0,
                vmin=-shared_vmax, vmax=shared_vmax, ax=ax,
                xticklabels=[f"V{t+1}" for t in range(eg.shape[0])],
                yticklabels=[feature_names[j] for j in top_idx],
                cbar_kws={"label": "EG value"},
            )
            ax.set_title(
                f"MODEL: {name}\nActual: {actual}  |  Pred: {prob:.4f}",
                fontsize=14, fontweight="bold",
            )

        plt.suptitle(f"{title}  (Test idx: {p_idx}{pid_str})",
                     fontsize=18, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.show()
        plt.close() 


# ==============================================================================
# 6.  OCCLUSION-BASED ATTRIBUTION
# ==============================================================================

def compute_occlusion_ntf(model, X_np, batch_size=64, device=None,
                           X_static_np=None):
    """Visit-level occlusion: zero out one visit at a time, measure prediction drop.

    Parameters
    ----------
    X_static_np : np.ndarray (N, n_static) or None
        Static features passed alongside sequence input on every forward call.

    Returns
    -------
    occ_ntf : np.ndarray  (N, T, F)
    occ_nt  : np.ndarray  (N, T)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    N, T, F  = X_np.shape
    X_t      = torch.tensor(X_np, dtype=torch.float32)
    s_t      = (torch.tensor(X_static_np, dtype=torch.float32)
                if X_static_np is not None else None)

    def _predict(x_tensor, s_tensor):
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                xb = x_tensor[i:i+batch_size].to(device)
                sb = s_tensor[i:i+batch_size].to(device) if s_tensor is not None else None
                preds.append(model(xb, sb).cpu().numpy().flatten())
        return np.concatenate(preds)

    base_preds = _predict(X_t, s_t)
    occ_nt     = np.zeros((N, T), dtype=np.float32)

    for t in range(T):
        X_occ        = X_np.copy()
        X_occ[:, t, :] = 0.0
        occ_preds    = _predict(torch.tensor(X_occ, dtype=torch.float32), s_t)
        occ_nt[:, t] = base_preds - occ_preds

    occ_ntf = np.repeat(occ_nt[:, :, np.newaxis], F, axis=2)
    return occ_ntf, occ_nt


def load_truth_ntf(sim_dir, tag, feature_names, n_patients, n_visits, test_indices):
    """Load and align simulation truth arrays with the Python feature ordering.

    Reads truth_icd_flat and truth_atc_flat CSVs, reshapes to [N, T, F],
    subsets to test_indices, and aligns columns to match feature_names ordering.

    Parameters
    ----------
    sim_dir       : path to simulation output directory
    tag           : scenario tag string (e.g. 'N3000_V20_trig0.15_esc0.10_bg0.010')
    feature_names : list of F feature names used in Python (tokens + Static_Bin/Cont)
    n_patients    : total number of patients (train + test)
    n_visits      : number of visits
    test_indices  : array of 0-based patient indices in the test set
                    (these are Python indices into the full patient array)

    Returns
    -------
    truth_ntf : np.ndarray [N_test, T, F]  truth weights aligned to feature_names
                Features not in truth (Static_Bin, Static_Cont) get weight 0.
    """
    icd_path = os.path.join(sim_dir, f"truth_icd_flat_{tag}.csv")
    atc_path = os.path.join(sim_dir, f"truth_atc_flat_{tag}.csv")

    icd_df = pd.read_csv(icd_path)
    atc_df = pd.read_csv(atc_path)

    # Drop index columns — keep only feature columns
    icd_cols = [c for c in icd_df.columns if c.startswith('ICD_')]
    atc_cols = [c for c in atc_df.columns if c.startswith('ATC_')]

    # Reshape ICD: [N*T, N_ICD] -> [N, T, N_ICD]
    icd_mat = icd_df[icd_cols].values.reshape(n_patients, n_visits, len(icd_cols))
    atc_mat = atc_df[atc_cols].values.reshape(n_patients, n_visits, len(atc_cols))

    # Build full truth array aligned to feature_names
    F           = len(feature_names)
    truth_full  = np.zeros((n_patients, n_visits, F), dtype=np.float32)

    for fi, fname in enumerate(feature_names):
        if fname in icd_cols:
            col_idx = icd_cols.index(fname)
            truth_full[:, :, fi] = icd_mat[:, :, col_idx]
        elif fname in atc_cols:
            col_idx = atc_cols.index(fname)
            truth_full[:, :, fi] = atc_mat[:, :, col_idx]
        # Static_Bin, Static_Cont, and any unknown features get 0

    # Subset to test patients
    truth_ntf = truth_full[test_indices]   # [N_test, T, F]
    return truth_ntf


def get_truth_ntf(sc, scenario_store, sim_dir, scenarios):
    """Return truth_ntf for scenario sc, loading and caching on first call."""
    s = scenario_store[sc]
    if 'truth_ntf' not in s:
        tag          = dict(scenarios)[sc]
        n_pat        = int(tag.split('_')[0][1:])
        test_indices = (s['pid_test'] - 1).astype(int)
        s['truth_ntf'] = load_truth_ntf(
            sim_dir=sim_dir, tag=tag, feature_names=s['feature_names'],
            n_patients=n_pat, n_visits=s['n_visits'], test_indices=test_indices,
        )
        print(f"  [truth] {sc}: {int((s['truth_ntf']>0).sum())} non-zero cells")
    return s['truth_ntf']


import matplotlib.colors as _mcolors
TRUTH_CMAP = _mcolors.LinearSegmentedColormap.from_list(
    'truth', ['#ffffff', '#f5a623', '#c0392b']
)


# ==============================================================================
# 7.  ADDITIONAL HELPERS
# ==============================================================================

def compute_auprc(model, X, y, flat: bool = False) -> float:
    """Compute Area Under the Precision-Recall Curve.

    Parameters
    ----------
    model : sklearn estimator (flat=True) or PyTorch nn.Module (flat=False)
    X     : feature matrix — shape (N, F) for flat, (N, T, F) for RNN
    y     : binary labels (N,)
    flat  : True → use model.predict_proba (sklearn); False → forward pass

    Returns
    -------
    float  AUPRC score
    """
    if flat:
        return average_precision_score(y, model.predict_proba(X)[:, 1])
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    return average_precision_score(y, p)


def make_predict_fn(model, T: int, F: int, device=None,
                    X_static_np=None):
    """Return a flat-input predict function bound to *model*.

    The returned callable accepts a 2-D array of shape (N, T*F), reshapes it
    to (N, T, F), runs a forward pass, and returns a 1-D probability array.
    This is the format expected by shap.KernelExplainer.

    Parameters
    ----------
    X_static_np : np.ndarray (N_total, n_static) or None
        When provided, static is fixed at its mean value across the dataset.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device).eval()

    if X_static_np is not None:
        s_mean = torch.tensor(
            X_static_np.mean(axis=0, keepdims=True).astype(np.float32),
            device=device,
        )
    else:
        s_mean = None

    def _predict(X_flat_in: np.ndarray) -> np.ndarray:
        X3d = torch.tensor(
            X_flat_in.reshape(-1, T, F).astype(np.float32), device=device)
        s = s_mean.expand(X3d.shape[0], -1) if s_mean is not None else None
        with torch.no_grad():
            return model(X3d, s).cpu().numpy().flatten()

    return _predict


def make_timeshap_predict_fn(model, X_static_np, device=None):
    """Return a TimeSHAP-compatible predict function bound to *model*.

    The returned callable accepts a 3-D array of shape (N, T, F_seq), runs a
    forward pass with static fixed at its test-set mean, and returns an array
    of shape (N, 1).  This is the format expected by timeshap.explainer
    functions (local_pruning, local_event, local_feat, local_cell_level).

    Parameters
    ----------
    model        : trained PyTorch nn.Module with forward(x, x_static) signature
    X_static_np  : np.ndarray (N, n_static) — static features; mean is used
    device       : torch.device (auto-detected if None)
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device).eval()

    s_mean = torch.tensor(
        X_static_np.mean(axis=0, keepdims=True).astype(np.float32),
        device=device,
    )

    def _predict(X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            score = model(X_t, s_mean.expand(X_t.shape[0], -1))
        return score.detach().cpu().numpy().reshape(-1, 1)

    return _predict


class ModelWrapper:
    """Thin wrapper that exposes a ``predict(X)`` method required by WindowSHAP.

    WindowSHAP expects a model object with a ``predict`` method that accepts a
    3-D numpy array (N, T, F) and returns an array of shape (N, 1).

    Parameters
    ----------
    predict_fn : callable (N, T, F) -> (N, 1)
        Any predict function, e.g. one returned by make_windowshap_predict_fn.
    """

    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_fn(X)


def make_windowshap_predict_fn(model, X_static_np, device=None):
    """Return a WindowSHAP-compatible predict function bound to *model*.

    The returned callable accepts a 3-D array of shape (N, T, F_seq), runs a
    forward pass with static fixed at its test-set mean, and returns an array
    of shape (N, 1).  Wrap the result in ModelWrapper to get an object with
    the ``.predict()`` interface required by WindowSHAP.

    Parameters
    ----------
    model        : trained PyTorch nn.Module with forward(x, x_static) signature
    X_static_np  : np.ndarray (N, n_static) — static features; mean is used
    device       : torch.device (auto-detected if None)

    Example
    -------
    >>> fn       = make_windowshap_predict_fn(model_gru, X_static_test, device)
    >>> model_ws = ModelWrapper(fn)
    >>> ws.StationaryWindowSHAP(model=model_ws, ...)
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device).eval()

    s_mean = torch.tensor(
        X_static_np.mean(axis=0, keepdims=True).astype(np.float32),
        device=device,
    )

    def _predict(X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(X_t, s_mean.expand(X_t.shape[0], -1))
        return out.cpu().numpy().reshape(-1, 1)

    return _predict


def rename_features(series: pd.Series, feature_names: list) -> pd.Series:
    """Map TimeSHAP's generic 'Feature N' labels to actual feature names.

    Parameters
    ----------
    series        : pd.Series of feature label strings (e.g. 'Feature 3')
    feature_names : ordered list of true feature names (index-aligned)
    """
    return series.apply(
        lambda x: feature_names[int(x.replace('Feature ', ''))]
        if isinstance(x, str) and x.startswith('Feature ') else x
    )


def add_explanation_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'Explanation' column equal to 'Shapley Value'.

    Some official TimeSHAP plot functions require an 'Explanation' column
    rather than 'Shapley Value'.
    """
    df = df.copy()
    df['Explanation'] = df['Shapley Value']
    return df


def prepare_event_data(event_data: pd.DataFrame,
                        pruning_idx: int,
                        T: int) -> pd.DataFrame:
    """Filter TimeSHAP event data to the valid window.

    Visits before *pruning_idx* are collapsed into a single
    'Pruned Events' row whose Shapley Value is the sum of all pruned visits.
    Phantom visits (index < -T) are dropped entirely.

    Parameters
    ----------
    event_data  : raw TimeSHAP event-level DataFrame
    pruning_idx : first visit kept as an individual event
    T           : total number of visits in the sequence

    Returns
    -------
    pd.DataFrame with pruned prefix collapsed
    """
    def _classify(x: str) -> str:
        if not x.startswith('Event '):
            return 'other'
        t = int(x.replace('Event ', ''))
        if t < -T:
            return 'phantom'
        elif t < pruning_idx:
            return 'pruned'
        return 'valid'

    ev = event_data.copy()
    ev['_class'] = ev['Feature'].apply(_classify)
    valid = ev[ev['_class'] == 'valid'].copy()
    pruned_sum = ev[ev['_class'] == 'pruned']['Shapley Value'].sum()

    prev_row = pd.DataFrame({
        'Random seed':   [ev['Random seed'].iloc[0]],
        'NSamples':      [ev['NSamples'].iloc[0]],
        'Feature':       ['Pruned Events'],
        'Shapley Value': [pruned_sum],
        '_class':        ['other'],
    })
    result = pd.concat([valid, prev_row], ignore_index=True)
    return result.drop(columns=['_class'])


def _patched_solve(self, fraction_evaluated: float, dim: int):
    """Monkeypatch for shap.explainers._kernel.Kernel.solve.

    Forces plain OLS (l1_reg=0) instead of LassoLarsIC, which fails when
    nsamples < n_features — a common situation when T*F is large.

    Usage
    -----
    import shap.explainers._kernel as _shap_kernel
    _original = _shap_kernel.Kernel.solve
    _shap_kernel.Kernel.solve = _patched_solve
    try:
        ...   # call shap
    finally:
        _shap_kernel.Kernel.solve = _original
    """
    import shap.explainers._kernel as _shap_kernel
    _original_solve = _shap_kernel.Kernel.solve.__wrapped__ \
        if hasattr(_shap_kernel.Kernel.solve, '__wrapped__') \
        else _shap_kernel.Kernel.solve
    saved = self.l1_reg
    self.l1_reg = 0
    result = _original_solve(self, fraction_evaluated, dim)
    self.l1_reg = saved
    return result


def plot_case_with_truth(case_title: str,
                          scenario_store: dict,
                          scenarios_ordered: list,
                          SIM_DIR: str,
                          SCENARIOS: list,
                          OUTPUT_DIR: str,
                          TOP_K: int = 15) -> None:
    """Plot EG / IG / Ground-Truth / Attention for one case type across scenarios.

    Produces a 4-row × 2-column figure per scenario (AttnGRU left,
    AttnLSTM right):
      Row 0 : EG heatmap
      Row 1 : IG heatmap
      Row 2 : Ground-truth heatmap  (orange colormap, values in {0, 0.5, 1})
      Row 3 : Attention weights bar chart

    All rows share the same top-K features (ranked by combined XAI signal),
    so y-axis labels are aligned across rows.

    Parameters
    ----------
    case_title        : display name of the case study (e.g. 'Pure Trigger')
    scenario_store    : dict populated by the main scenario loop
    scenarios_ordered : list of scenario names in display order
    SIM_DIR           : directory containing simulation CSV outputs
    SCENARIOS         : list of (scenario_name, tag) tuples
    OUTPUT_DIR        : root directory where per-scenario sub-folders live
    TOP_K             : number of features to display (default 15)
    """
    for sc in scenarios_ordered:
        s = scenario_store[sc]
        fnames = s['feature_names']
        n_vis = s['n_visits']
        p_idx = s['case_studies'].get(case_title)
        if p_idx is None:
            print(f"  [{sc}] '{case_title}' not available — skipping")
            continue

        pid = int(s['pid_test'][p_idx])
        X_tens = s['X_test_tensor']
        truth_ntf = get_truth_ntf(sc, scenario_store, SIM_DIR, SCENARIOS)

        # ── shared top-K features ─────────────────────────────────────────
        top_idx = np.argsort(
            np.abs(s['eg_attn_gru_ntf' ][p_idx]).sum(axis=0) +
            np.abs(s['eg_attn_lstm_ntf'][p_idx]).sum(axis=0) +
            np.abs(s['ig_attn_gru_ntf' ][p_idx]).sum(axis=0) +
            np.abs(s['ig_attn_lstm_ntf'][p_idx]).sum(axis=0)
        )[::-1][:TOP_K]
        top_fnames = [fnames[j] for j in top_idx]
        visits = [f'V{t+1}' for t in range(n_vis)]

        # ── ground-truth slice [TOP_K, T] ─────────────────────────────────
        truth_pat = truth_ntf[p_idx][:, top_idx].T

        # ── figure ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(22, 18))
        fig.suptitle(f'{sc}  |  {case_title}  |  PID {pid}',
                     fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(4, 2, figure=fig,
                               height_ratios=[3, 3, 2, 1.2],
                               hspace=0.50, wspace=0.35)

        for col, (mname, eg_key, ig_key, attn_key) in enumerate([
            ('AttnGRU',  'eg_attn_gru_ntf',  'ig_attn_gru_ntf',  'model_attn_gru'),
            ('AttnLSTM', 'eg_attn_lstm_ntf', 'ig_attn_lstm_ntf', 'model_attn_lstm'),
        ]):
            # Row 0: EG
            ax_eg = fig.add_subplot(gs[0, col])
            attr_eg = s[eg_key][p_idx]
            vmax_eg = np.abs(attr_eg[:, top_idx]).max() + 1e-8
            sns.heatmap(attr_eg[:, top_idx].T, ax=ax_eg,
                        cmap='RdBu_r', center=0,
                        vmin=-vmax_eg, vmax=vmax_eg,
                        xticklabels=visits, yticklabels=top_fnames,
                        cbar_kws={'label': 'EG attr.', 'shrink': 0.8})
            ax_eg.set_title(f'EG | {mname}', fontsize=10, fontweight='bold')
            ax_eg.set_xlabel('')
            ax_eg.tick_params(axis='x', labelsize=7, rotation=90)

            # Row 1: IG
            ax_ig = fig.add_subplot(gs[1, col])
            attr_ig = s[ig_key][p_idx]
            vmax_ig = np.abs(attr_ig[:, top_idx]).max() + 1e-8
            sns.heatmap(attr_ig[:, top_idx].T, ax=ax_ig,
                        cmap='RdBu_r', center=0,
                        vmin=-vmax_ig, vmax=vmax_ig,
                        xticklabels=visits, yticklabels=top_fnames,
                        cbar_kws={'label': 'IG attr.', 'shrink': 0.8})
            ax_ig.set_title(f'IG | {mname}', fontsize=10, fontweight='bold')
            ax_ig.set_xlabel('')
            ax_ig.tick_params(axis='x', labelsize=7, rotation=90)

            # Row 2: Ground-truth
            ax_gt = fig.add_subplot(gs[2, col])
            sns.heatmap(truth_pat, ax=ax_gt,
                        cmap=TRUTH_CMAP, vmin=0, vmax=1,
                        xticklabels=visits, yticklabels=top_fnames,
                        cbar_kws={'label': 'Truth weight', 'shrink': 0.8},
                        linewidths=0.3, linecolor='#e0e0e0')
            ax_gt.set_title(f'Ground Truth | {mname}',
                            fontsize=10, fontweight='bold', color='#c0392b')
            ax_gt.set_xlabel('')
            ax_gt.tick_params(axis='x', labelsize=7, rotation=90)

            # Row 3: Attention
            ax_attn = fig.add_subplot(gs[3, col])
            model = s[attn_key]
            w = model.get_attention_weights(X_tens[p_idx:p_idx+1])[0]
            ax_attn.bar(range(n_vis), w, color='steelblue', alpha=0.85)
            ax_attn.set_xticks(range(n_vis))
            ax_attn.set_xticklabels(visits, fontsize=7, rotation=90)
            ax_attn.set_ylabel('Attn weight', fontsize=8)
            ax_attn.set_title(f'Attention | {mname}',
                              fontsize=10, fontweight='bold')
            ax_attn.grid(True, alpha=0.3, axis='y')
            ax_attn.set_xlim(-0.5, n_vis - 0.5)

        safe_case = (case_title.replace(' ', '_')
                               .replace(':', '')
                               .replace('+', '_'))
        out_path = os.path.join(OUTPUT_DIR, sc,
                                f'attn_vs_eg_ig_gt_{safe_case}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'  Saved: {out_path}')




# ==============================================================================
# 8.  CONDITIONAL SHAP — pure Python implementation
#     Aas et al. (2021) observational/conditional approach
#     No shapr/R dependency
# ==============================================================================

def compute_conditional_shap(
    model,
    X_seq_train: np.ndarray,
    X_seq_test: np.ndarray,
    case_indices: list,
    T: int,
    n_tokens: int,
    device=None,
    X_static_train: np.ndarray = None,
    X_static_test: np.ndarray = None,
    approach: str = "conditional_gaussian",
    background_size: int = 200,
    n_combinations: int = 2000,
    n_MC_samples: int = 50,
    seed: int = 42,
    gt_feature_indices: list = None,
) -> tuple:
    """Conditional SHAP via pure Python — Aas et al. (2021) Section 3.1.

    Players are every (visit t, GT feature f) cell:
      T=20 visits x 5 GT features = 100 players.

    The conditional distribution p(x_S_bar | x_S) is estimated as a
    multivariate Gaussian parameterised by the sample mean and covariance
    of the background set in player space.

    WLS solver uses the Lundberg & Lee efficiency-constrained formulation:
    drop the last player and recover it exactly from
      sum(phi) = v(full) - v(empty)
    guaranteeing efficiency by construction.

    Parameters
    ----------
    model               : trained PyTorch nn.Module
    X_seq_train         : (N_train, T, F_seq)
    X_seq_test          : (N_test,  T, F_seq)
    case_indices        : list of int into X_seq_test
    T                   : number of visits
    n_tokens            : F_seq — total number of sequence features
    device              : torch.device (auto-detected if None)
    X_static_train      : (N_train, n_static) or None
    X_static_test       : (N_test,  n_static) or None
    approach            : always 'conditional_gaussian' in this implementation
    background_size     : number of training samples for covariance estimation
    n_combinations      : number of coalitions to sample (>= 2000 for 100 players)
    n_MC_samples        : MC draws per coalition for E[f(x)|x_S]
    seed                : RNG seed
    gt_feature_indices  : feature indices to treat as Shapley players.
                          Must be provided explicitly. Get correct indices with:
                          sorted([feature_names.index(f) for f in
                            ['ICD_1','ICD_2','ICD_3','ATC_5','ICD_10']])

    Returns
    -------
    cshap_ntf     : np.ndarray  (N_cases, T, F_seq)
        Conditional Shapley values. Non-zero only at gt_feature_indices columns.
    cshap_sd_ntf  : np.ndarray  (N_cases, T, F_seq)
        Zeros — SD not estimated in this implementation.
    phi0_per_case : np.ndarray  (N_cases,)
        v(empty set) per case — the baseline prediction used by the solver.
        Required for correct efficiency verification in the notebook.
    phi_full_per_case : np.ndarray  (N_cases,)
        v(full set) per case — f(x*) in player space.
    """
    import math
    from scipy.linalg import solve as sp_solve, cholesky, solve_triangular

    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1.  Device + static mean
    # ------------------------------------------------------------------
    if device is None:
        device = next(model.parameters()).device
    model.to(device).eval()

    s_mean = None
    if X_static_test is not None:
        s_mean = torch.tensor(
            X_static_test.mean(axis=0, keepdims=True).astype(np.float32),
            device=device,
        )

    # ------------------------------------------------------------------
    # 2.  GT features — players are (visit, feature) cells
    #     player p = t * n_gt + g
    #       => visit t  = p // n_gt
    #       => gt feat g = p % n_gt
    #       => actual feature index = gt_feature_indices[g]
    # ------------------------------------------------------------------
    if gt_feature_indices is None:
        raise ValueError(
            "gt_feature_indices must be provided explicitly.\n"
            "Get correct indices with:\n"
            "  sorted([feature_names.index(f) for f in "
            "['ICD_1','ICD_2','ICD_3','ATC_5','ICD_10']])"
        )
    gt_feature_indices = sorted(gt_feature_indices)
    n_gt      = len(gt_feature_indices)
    n_players = T * n_gt                          # e.g. 20 * 5 = 100

    # ------------------------------------------------------------------
    # 3.  Background sample
    # ------------------------------------------------------------------
    rng    = np.random.default_rng(seed)
    bg_idx = rng.choice(
        len(X_seq_train),
        size=min(background_size, len(X_seq_train)),
        replace=False,
    )

    # Background mean over full (T, F) — used for non-GT features
    bg_mean_tf = X_seq_train[bg_idx].mean(axis=0).astype(np.float32)  # (T, F)

    # ------------------------------------------------------------------
    # 4.  Flatten sequences to player vectors
    #     (N, T, F) -> (N, T*n_gt) extracting only GT feature cells
    # ------------------------------------------------------------------
    def _seq_to_players(X_nTF):
        N   = X_nTF.shape[0]
        out = np.zeros((N, n_players), dtype=np.float32)
        for t in range(T):
            for g, f in enumerate(gt_feature_indices):
                out[:, t * n_gt + g] = X_nTF[:, t, f]
        return out

    X_bg_flat    = _seq_to_players(X_seq_train[bg_idx])     # (bg, n_players)
    X_cases_flat = _seq_to_players(X_seq_test[case_indices]) # (N_cases, n_players)

    # ------------------------------------------------------------------
    # 5.  Gaussian parameters estimated from background
    # ------------------------------------------------------------------
    mu  = X_bg_flat.mean(axis=0).astype(np.float64)          # (n_players,)
    cov = np.cov(X_bg_flat.T).astype(np.float64)             # (n_players, n_players)
    cov += np.eye(n_players) * 1e-4                          # ridge regularisation

    # Precompute Cholesky of full covariance for stable marginal sampling
    try:
        L_full = cholesky(cov, lower=True)
    except np.linalg.LinAlgError:
        cov   += np.eye(n_players) * 1e-3
        L_full = cholesky(cov, lower=True)

    # ------------------------------------------------------------------
    # 6.  Predict function in player space
    #     Input:  (N, n_players) — binary value per (visit, GT feature)
    #     Output: (N,) predicted probabilities
    #
    #     Non-GT features held at background mean on every call.
    #     GT feature values placed at exact (visit, feature) positions —
    #     no division or clipping, model sees same values it was trained on.
    # ------------------------------------------------------------------
    def _predict_from_players(X_play):
        N      = X_play.shape[0]
        X_full = np.tile(bg_mean_tf, (N, 1, 1))              # (N, T, F)
        for t in range(T):
            for g, f in enumerate(gt_feature_indices):
                X_full[:, t, f] = X_play[:, t * n_gt + g]
        X_3d = torch.tensor(X_full, dtype=torch.float32, device=device)
        s    = s_mean.expand(N, -1) if s_mean is not None else None
        with torch.no_grad():
            return model(X_3d, s).cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # 7.  Conditional expectation v(S) = E[f(x) | x_S = x*_S]
    #
    #     Gaussian conditional (Aas et al. 2021, equations 10 & 11):
    #       mu_cond  = mu_Sbar + Cov(Sbar,S) Cov(S,S)^{-1} (x*_S - mu_S)
    #       cov_cond = Cov(Sbar,Sbar) - Cov(Sbar,S) Cov(S,S)^{-1} Cov(S,Sbar)
    #
    #     Cholesky-based solve used throughout for numerical stability.
    # ------------------------------------------------------------------
    def _conditional_mean(x_star, coalition_mask):
        S     = np.where(coalition_mask)[0]
        S_bar = np.where(~coalition_mask)[0]

        # Edge case: all features present — return f(x*) directly
        if len(S_bar) == 0:
            return float(_predict_from_players(x_star[np.newaxis, :])[0])

        # Edge case: no features present — sample from marginal N(mu, cov)
        if len(S) == 0:
            z       = np.random.randn(n_MC_samples, n_players)
            samples = (mu + z @ L_full.T).astype(np.float32)
            return float(_predict_from_players(samples).mean())

        # Partition covariance matrix
        mu_s     = mu[S]
        mu_sbar  = mu[S_bar]
        cov_ss   = cov[np.ix_(S,     S)]
        cov_sbs  = cov[np.ix_(S_bar, S)]
        cov_sbsb = cov[np.ix_(S_bar, S_bar)]

        # Cholesky solve for conditional parameters
        cov_ss_reg = cov_ss + np.eye(len(S)) * 1e-6
        diff       = x_star[S].astype(np.float64) - mu_s

        try:
            L_ss   = cholesky(cov_ss_reg, lower=True)
            alpha  = solve_triangular(
                L_ss.T,
                solve_triangular(L_ss, diff, lower=True),
                lower=False,
            )
            B      = solve_triangular(
                L_ss.T,
                solve_triangular(L_ss, cov_sbs.T, lower=True),
                lower=False,
            )
        except np.linalg.LinAlgError:
            cov_ss_inv = np.linalg.pinv(cov_ss_reg)
            alpha      = cov_ss_inv @ diff
            B          = cov_ss_inv @ cov_sbs.T

        # Conditional mean and covariance (Aas et al. Eq. 10-11)
        mu_cond   = mu_sbar + cov_sbs @ alpha
        cov_cond  = cov_sbsb - cov_sbs @ B
        cov_cond += np.eye(len(S_bar)) * 1e-6

        # Sample via Cholesky reparameterisation: x = mu_cond + L_cond @ z
        try:
            L_cond = cholesky(cov_cond, lower=True)
        except np.linalg.LinAlgError:
            cov_cond += np.eye(len(S_bar)) * 1e-3
            L_cond    = cholesky(cov_cond, lower=True)

        z            = np.random.randn(n_MC_samples, len(S_bar))
        samples_sbar = mu_cond + z @ L_cond.T        # (n_MC_samples, |S_bar|)

        X_mc           = np.tile(x_star.astype(np.float64), (n_MC_samples, 1))
        X_mc[:, S_bar] = samples_sbar
        return float(_predict_from_players(X_mc.astype(np.float32)).mean())

    # ------------------------------------------------------------------
    # 8.  Shapley kernel weights in log space
    #     Avoids C(100,50) overflow for n_players=100
    # ------------------------------------------------------------------
    def _log_kernel_weight(M, s):
        """log w(M,s) = log(M-1) - logC(M,s) - log(s) - log(M-s)"""
        return (math.log(M - 1)
                - (math.lgamma(M + 1) - math.lgamma(s + 1)
                   - math.lgamma(M - s + 1))
                - math.log(s)
                - math.log(M - s))

    def _kernel_weight(M, s):
        if s == 0 or s == M:
            return 1e8
        return math.exp(_log_kernel_weight(M, s))

    # ------------------------------------------------------------------
    # 9.  Sample coalitions proportional to kernel weights
    # ------------------------------------------------------------------
    def _sample_coalitions(M, n_samples, rng_seed):
        rng_c       = np.random.default_rng(rng_seed)
        sizes       = np.arange(1, M)
        log_weights = np.array(
            [_log_kernel_weight(M, s) for s in sizes], dtype=np.float64)
        log_weights -= log_weights.max()              # prevent underflow
        weights      = np.exp(log_weights)
        weights     /= weights.sum()

        coalitions = [np.zeros(M, dtype=bool),        # empty — always first
                      np.ones(M,  dtype=bool)]         # full  — always last
        for _ in range(n_samples - 2):
            size = int(rng_c.choice(sizes, p=weights))
            cols = rng_c.choice(M, size=size, replace=False)
            mask = np.zeros(M, dtype=bool)
            mask[cols] = True
            coalitions.append(mask)
        return coalitions

    # ------------------------------------------------------------------
    # 10.  Build coalition matrix Z and kernel weight vector
    # ------------------------------------------------------------------
    coalitions = _sample_coalitions(n_players, n_combinations, seed)
    n_coal     = len(coalitions)

    sizes_dist = [int(c.sum()) for c in coalitions]
    print(f"  Coalitions: n={n_coal}, size-1={sizes_dist.count(1)}, "
          f"size-(M-1)={sizes_dist.count(n_players-1)}, "
          f"mean={np.mean(sizes_dist):.1f}", flush=True)

    Z      = np.array([c.astype(np.float64) for c in coalitions])
    W_diag = np.array(
        [_kernel_weight(n_players, int(c.sum())) for c in coalitions],
        dtype=np.float64)

    # ------------------------------------------------------------------
    # 11.  Main loop — explain one patient at a time
    # ------------------------------------------------------------------
    sv_all            = np.zeros((len(case_indices), n_players), dtype=np.float32)
    phi0_per_case     = np.zeros(len(case_indices), dtype=np.float32)
    phi_full_per_case = np.zeros(len(case_indices), dtype=np.float32)

    for ci, x_star in enumerate(X_cases_flat):

        # Compute v(S) for every coalition
        vS = np.zeros(n_coal, dtype=np.float64)
        for j, mask in enumerate(coalitions):
            vS[j] = _conditional_mean(x_star, mask)

        phi0     = vS[0]     # v(empty) = E[f(x)]
        phi_full = vS[-1]    # v(full)  = f(x*) in player space
        target   = phi_full - phi0

        # Store for notebook verification
        phi0_per_case[ci]     = phi0
        phi_full_per_case[ci] = phi_full

        y = vS - phi0

        # Interior coalitions only (exclude endpoints from WLS)
        interior = [i for i, c in enumerate(coalitions)
                    if 0 < int(c.sum()) < n_players]
        Z_int    = Z[interior]
        y_int    = y[interior]
        W_int    = W_diag[interior]

        # --------------------------------------------------------------
        # Efficiency-constrained WLS — Lundberg & Lee (2017)
        #
        # Enforce sum(phi) = target exactly by substitution:
        #   phi[M-1] = target - sum(phi[0:M-1])
        #
        # Substitute into WLS objective:
        #   y_adj = y_int - z_last * target
        #   Z_adj = Z_red - z_last[:, None]
        # Then solve: min ||W^{1/2}(y_adj - Z_adj phi_red)||^2
        # Recover: phi_last = target - sum(phi_red)
        #
        # This makes efficiency exact by construction — not approximate.
        # --------------------------------------------------------------
        z_last = Z_int[:, -1]
        Z_red  = Z_int[:, :-1]
        y_adj  = y_int - z_last * target
        Z_adj  = Z_red - z_last[:, np.newaxis]

        W_mat  = np.diag(W_int)
        ZtWZ   = Z_adj.T @ W_mat @ Z_adj
        ZtWZ  += np.eye(n_players - 1) * 1e-8
        ZtWy   = Z_adj.T @ W_mat @ y_adj

        try:
            phi_red = sp_solve(ZtWZ, ZtWy, assume_a='sym')
        except Exception:
            phi_red = np.linalg.lstsq(ZtWZ, ZtWy, rcond=None)[0]

        # Recover last phi exactly from efficiency constraint
        phi_last = target - phi_red.sum()
        phi      = np.append(phi_red, phi_last).astype(np.float32)

        # Log efficiency — should be machine precision (~1e-8)
        efficiency_error = abs(float(phi.sum()) - target)
        print(f"  ci={ci}: target={target:.4f}  "
              f"sum(phi)={phi.sum():.4f}  "
              f"error={efficiency_error:.2e}", flush=True)

        sv_all[ci] = phi

    # ------------------------------------------------------------------
    # 12.  Expand (N_cases, n_players) -> (N_cases, T, F_seq)
    #      Place each player's value at its exact (visit, feature) cell.
    #      All non-GT feature columns remain zero.
    # ------------------------------------------------------------------
    cshap_ntf    = np.zeros((len(case_indices), T, n_tokens), dtype=np.float32)
    cshap_sd_ntf = np.zeros((len(case_indices), T, n_tokens), dtype=np.float32)

    for p in range(n_players):
        t = p // n_gt
        g = p % n_gt
        f = gt_feature_indices[g]
        cshap_ntf[:, t, f] = sv_all[:, p]

    return cshap_ntf, cshap_sd_ntf, phi0_per_case, phi_full_per_case


