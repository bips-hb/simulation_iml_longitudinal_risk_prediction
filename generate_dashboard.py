"""
generate_dashboard.py
=====================
Run this as a notebook cell or standalone script to produce an interactive
HTML dashboard for the EHR XAI benchmark results.

Usage in notebook:
    exec(open('generate_dashboard.py').read())
    # or paste the generate_dashboard() call directly

The dashboard is a single self-contained HTML file — open it in any browser,
no server required. Images are embedded as base64 so the file is portable.

Set OUTPUT_DIR and SCENARIO_NAME before running, or pass them as arguments.
"""

import os, base64, json
from pathlib import Path


def generate_dashboard(
    output_dir: str,
    scenario_name: str = "baseline",
    dashboard_path: str = None,
    embed_images: bool = True,
):
    """
    Generate an interactive HTML dashboard for one scenario's XAI results.

    Parameters
    ----------
    output_dir     : directory where all .png files live
    scenario_name  : label shown in the dashboard header
    dashboard_path : where to save the HTML (default: output_dir/dashboard.html)
    embed_images   : if True, embed images as base64 (portable single file)
                     if False, use relative paths (faster, but keep pngs alongside)
    """
    if dashboard_path is None:
        dashboard_path = os.path.join(output_dir, "dashboard.html")

    # ── catalogue available plots ──────────────────────────────────────────
    CASES = [
        ("Pure Trigger",          "Pure_Trigger"),
        ("Pure Chronic",          "Pure_Chronic"),
        ("Pure Late Escalation",  "Pure_Late_Escalation"),
        ("Trigger + Chronic",     "Mixed_Trigger___Chronic"),
        ("Trigger + Escalation",  "Mixed_Trigger___Escalation"),
        ("Chronic + Escalation",  "Mixed_Chronic___Escalation"),
        ("All Three",             "Mixed_All_Three"),
        ("False Positive",        "False_Positive"),
    ]

    GLOBAL_PLOTS = [
        ("Training Curves",           "training_curves.png"),
        ("Performance Summary",       "auc_summary_table.png"),
        ("ROC / PR Curves",           "roc_pr_curves.png"),
        ("IG Global Bars",            "ig_global_bars.png"),
        ("EG Global Bars",            "eg_global_bars.png"),
        ("Attention Weights (global)","attention_weights.png"),
    ]

    CASE_PLOT_TYPES = [
        ("All Methods",              "all_methods_{}.png"),
        ("KernelSHAP",               "kernelshap_all_models_{}.png"),
        ("TimeSHAP",                 "timeshap_all_models_{}.png"),
        ("TimeSHAP Pruning",         "timeshap_pruning_{}.png"),
        ("KS + TS",                  "ks_ts_{}.png"),
        ("KS + TS + GT",             "ks_ts_gt_{}.png"),
        ("KS + WS + TS",             "ks_ws_ts_{}.png"),
        ("KS + WS + TS + GT",        "ks_ws_ts_gt_{}.png"),
        ("EG Cases",                 "eg_case_{}.png"),
        ("IG Cases",                 "ig_case_{}.png"),
        ("EG + IG",                  "eg_ig_{}.png"),
        ("EG + IG + GT",             "eg_ig_gt_{}.png"),
        ("XAI + Attention",          "attn_xai_{}.png"),
        ("XAI + Attention (scenario)", "attn_xai_{}_SCENARIO.png"),
        ("Attn-weighted EG",         "attn_weighted_eg_{}.png"),
        ("Attn Disagreement",        "attn_disagreement_{}.png"),
        ("Visit-level Comparison",   "attn_vs_xai_visit_{}.png"),
    ]

    def img_src(fname):
        # Handle SCENARIO placeholder — try scenario-suffixed filename first
        sc_name = os.path.basename(output_dir.rstrip('/'))
        resolved = fname.replace('_SCENARIO', f'_{sc_name}')
        fpath = os.path.join(output_dir, resolved)
        if not os.path.exists(fpath):
            # fallback: try plain filename without scenario suffix
            fpath = os.path.join(output_dir, fname.replace('_SCENARIO', ''))
        if not os.path.exists(fpath):
            return None
        if embed_images:
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{b64}"
        return os.path.relpath(fpath, output_dir)

    def _cf_html_for_case(sc_dir, case_label):
        """Build combined CF HTML for all models, filtered to case_label rows."""
        import csv as _csv
        MODEL_KEYS_CF   = ['gru', 'lstm', 'attn_gru', 'attn_lstm', 'cnn']
        MODEL_LABELS_CF = ['GRU', 'LSTM', 'AttnGRU', 'AttnLSTM', 'CNN']
        parts = []
        for mkey, mlabel in zip(MODEL_KEYS_CF, MODEL_LABELS_CF):
            fpath = os.path.join(sc_dir, f'cf_edits_{mkey}.csv')
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, newline='', encoding='utf-8') as f:
                    rows = list(_csv.reader(f))
                if not rows:
                    continue
                header = rows[0]
                case_rows = [r for r in rows[1:]
                             if any(case_label.lower() in c.lower() for c in r)]
                if not case_rows:
                    continue
                th   = ''.join(f'<th>{h}</th>' for h in header)
                body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                               for r in case_rows)
                tbl  = (f'<table class="cf-table"><thead><tr>{th}</tr></thead>'
                        f'<tbody>{body}</tbody></table>')
                parts.append(f'<h3>{mlabel}</h3>{tbl}')
            except Exception:
                continue
        if not parts:
            return None
        return ('<div style="display:flex;flex-direction:column;gap:24px;">'
                + ''.join(parts) + '</div>')

    # ── build data structure for JS ────────────────────────────────────────
    # global_tabs: list of {label, src}
    global_tabs = []
    for label, fname in GLOBAL_PLOTS:
        src = img_src(fname)
        if src:
            global_tabs.append({"label": label, "src": src})

    # case_data: list of {label, slug, plots: [{label, src|html, type}]}
    case_data = []
    for case_label, case_slug in CASES:
        plots = []
        for plot_label, pattern in CASE_PLOT_TYPES:
            fname = pattern.format(case_slug)
            src = img_src(fname)
            if src:
                plots.append({"label": plot_label, "src": src, "type": "img"})
        cf_html = _cf_html_for_case(output_dir, case_label)
        if cf_html:
            plots.append({"label": "Counterfactuals", "html": cf_html, "type": "table"})
        # Only include case if it has at least one image plot (not just CF)
        has_img = any(p.get("type") == "img" for p in plots)
        if plots and has_img:
            case_data.append({
                "label": case_label,
                "slug":  case_slug,
                "plots": plots,
            })

    global_json = json.dumps(global_tabs)
    cases_json  = json.dumps(case_data)

    # ── HTML ───────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAI Dashboard — {scenario_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {{
    --bg:       #0f1117;
    --surface:  #181c27;
    --border:   #2a2f3d;
    --accent:   #4f8ef7;
    --accent2:  #e94560;
    --text:     #d4d8e8;
    --muted:    #6b7280;
    --radius:   8px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  /* ── Header ── */
  header {{
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface);
    position: sticky; top: 0; z-index: 100;
  }}
  header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  header span {{
    font-size: 12px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Layout ── */
  .layout {{
    display: flex;
    flex: 1;
    overflow: hidden;
    height: calc(100vh - 57px);
  }}

  /* ── Sidebar ── */
  aside {{
    width: 220px;
    min-width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 16px 0;
  }}
  .sidebar-section {{
    padding: 6px 16px 4px;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
  }}
  .sidebar-item {{
    padding: 7px 16px;
    cursor: pointer;
    font-size: 13px;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .sidebar-item:hover {{
    background: rgba(79,142,247,0.08);
    color: #fff;
  }}
  .sidebar-item.active {{
    border-left-color: var(--accent);
    background: rgba(79,142,247,0.12);
    color: var(--accent);
    font-weight: 600;
  }}

  /* ── Main content ── */
  main {{
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
  }}

  .section-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Plot type tabs ── */
  .tab-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 18px;
  }}
  .tab-btn {{
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-size: 12px;
    font-family: 'IBM Plex Sans', sans-serif;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .tab-btn:hover {{
    border-color: var(--accent);
    color: var(--text);
  }}
  .tab-btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }}

  /* ── Image viewer ── */
  .img-container {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    text-align: center;
  }}
  .img-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    cursor: zoom-in;
    transition: transform 0.2s;
  }}
  .img-label {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Empty state ── */
  .empty {{
    color: var(--muted);
    font-size: 13px;
    padding: 48px;
    text-align: center;
    font-style: italic;
  }}

  /* ── Lightbox ── */
  #lightbox {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
  }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{
    max-width: 96vw;
    max-height: 96vh;
    border-radius: 4px;
    box-shadow: 0 8px 48px rgba(0,0,0,0.8);
  }}
  #lightbox-close {{
    position: fixed;
    top: 16px; right: 20px;
    color: #fff;
    font-size: 28px;
    cursor: pointer;
    line-height: 1;
    opacity: 0.7;
  }}
  #lightbox-close:hover {{ opacity: 1; }}


  /* ── CF table ── */
  .cf-container {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px; overflow-x: auto;
  }}
  .cf-container h3 {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;
  }}
  .cf-table {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  .cf-table th {{
    background: #1e2333; color: var(--muted); font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .cf-table td {{
    padding: 6px 10px; border-bottom: 1px solid #1e2333;
    vertical-align: top; line-height: 1.4;
  }}
  .cf-table tr:hover td {{ background: rgba(79,142,247,0.05); }}
  .cf-table td:first-child {{ font-weight: 600; color: var(--accent); white-space: nowrap; }}
  /* scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <h1>EHR XAI Dashboard</h1>
  <span>// {scenario_name}</span>
</header>

<div class="layout">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>

<div id="lightbox">
  <span id="lightbox-close">&#x2715;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const GLOBAL = {global_json};
const CASES  = {cases_json};

let currentSection = '__global__';
let currentPlotIdx = 0;

// ── Sidebar ───────────────────────────────────────────────────────────────
function buildSidebar() {{
  const aside = document.getElementById('sidebar');
  aside.innerHTML = '';

  // Global section
  const gLabel = document.createElement('div');
  gLabel.className = 'sidebar-section';
  gLabel.textContent = 'Global';
  aside.appendChild(gLabel);

  const gItem = document.createElement('div');
  gItem.className = 'sidebar-item' + (currentSection === '__global__' ? ' active' : '');
  gItem.textContent = 'Overview';
  gItem.onclick = () => {{ currentSection = '__global__'; currentPlotIdx = 0; render(); }};
  aside.appendChild(gItem);

  // Case sections
  const cLabel = document.createElement('div');
  cLabel.className = 'sidebar-section';
  cLabel.textContent = 'Case Studies';
  aside.appendChild(cLabel);

  CASES.forEach((c, i) => {{
    const item = document.createElement('div');
    item.className = 'sidebar-item' + (currentSection === c.slug ? ' active' : '');
    item.textContent = c.label;
    item.title = c.label;
    item.onclick = () => {{ currentSection = c.slug; currentPlotIdx = 0; render(); }};
    aside.appendChild(item);
  }});
}}

// ── Main ──────────────────────────────────────────────────────────────────
function render() {{
  buildSidebar();
  const main = document.getElementById('main');

  if (currentSection === '__global__') {{
    if (GLOBAL.length === 0) {{
      main.innerHTML = '<div class="empty">No global plots found.</div>';
      return;
    }}
    const title = document.createElement('div');
    title.className = 'section-title';
    title.textContent = 'Global Results';
    main.innerHTML = '';
    main.appendChild(title);

    // Tab bar
    const tabBar = document.createElement('div');
    tabBar.className = 'tab-bar';
    GLOBAL.forEach((p, i) => {{
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (i === currentPlotIdx ? ' active' : '');
      btn.textContent = p.label;
      btn.onclick = () => {{ currentPlotIdx = i; render(); }};
      tabBar.appendChild(btn);
    }});
    main.appendChild(tabBar);

    // Image
    const p = GLOBAL[currentPlotIdx];
    const cont = document.createElement('div');
    cont.className = 'img-container';
    const img = document.createElement('img');
    img.src = p.src;
    img.alt = p.label;
    img.onclick = () => openLightbox(p.src);
    cont.appendChild(img);
    const lbl = document.createElement('div');
    lbl.className = 'img-label';
    lbl.textContent = p.label;
    cont.appendChild(lbl);
    main.appendChild(cont);

  }} else {{
    const caseObj = CASES.find(c => c.slug === currentSection);
    if (!caseObj) return;

    const title = document.createElement('div');
    title.className = 'section-title';
    title.textContent = caseObj.label;
    main.innerHTML = '';
    main.appendChild(title);

    if (caseObj.plots.length === 0) {{
      main.innerHTML += '<div class="empty">No plots found for this case.</div>';
      return;
    }}

    // Tab bar
    const tabBar = document.createElement('div');
    tabBar.className = 'tab-bar';
    caseObj.plots.forEach((p, i) => {{
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (i === currentPlotIdx ? ' active' : '');
      btn.textContent = p.label;
      btn.onclick = () => {{ currentPlotIdx = i; render(); }};
      tabBar.appendChild(btn);
    }});
    main.appendChild(tabBar);

    // Image or table
    const p = caseObj.plots[currentPlotIdx];
    if (p.type === 'table') {{
      const cont = document.createElement('div');
      cont.className = 'cf-container';
      cont.innerHTML = p.html;
      main.appendChild(cont);
    }} else {{
      const cont = document.createElement('div');
      cont.className = 'img-container';
      const img = document.createElement('img');
      img.src = p.src;
      img.alt = p.label;
      img.onclick = () => openLightbox(p.src);
      cont.appendChild(img);
      const lbl = document.createElement('div');
      lbl.className = 'img-label';
      lbl.textContent = p.label + ' — click to enlarge';
      cont.appendChild(lbl);
      main.appendChild(cont);
    }}
  }}
}}

// ── Lightbox ──────────────────────────────────────────────────────────────
function openLightbox(src) {{
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}}
document.getElementById('lightbox').onclick = () =>
  document.getElementById('lightbox').classList.remove('open');
document.getElementById('lightbox-close').onclick = (e) => {{
  e.stopPropagation();
  document.getElementById('lightbox').classList.remove('open');
}};

// Keyboard navigation
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
  if (e.key === 'ArrowRight') {{
    const plots = currentSection === '__global__' ? GLOBAL
      : (CASES.find(c => c.slug === currentSection)?.plots || []);
    if (currentPlotIdx < plots.length - 1) {{ currentPlotIdx++; render(); }}
  }}
  if (e.key === 'ArrowLeft') {{
    if (currentPlotIdx > 0) {{ currentPlotIdx--; render(); }}
  }}
}});

render();
</script>
</body>
</html>"""

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(dashboard_path) / 1e6
    print(f"Dashboard saved → {dashboard_path}  ({size_mb:.1f} MB)")
    return dashboard_path


# ── Usage ─────────────────────────────────────────────────────────────────
# Call this from your notebook:
#
#   from generate_dashboard import generate_dashboard
#   generate_dashboard(out, scenario_name=SCENARIO_NAME)
#
# Or for eICU:
#   generate_dashboard(OUTPUT_DIR, scenario_name=f'eICU {SUFFIX}')


def generate_multi_dashboard(
    scenarios: list,
    dashboard_path: str,
    embed_images: bool = True,
):
    """
    Generate a combined interactive HTML dashboard for multiple scenarios
    (e.g. eICU L1 + L2, or all simulation scenarios).

    Parameters
    ----------
    scenarios      : list of (label, output_dir) tuples
                     e.g. [('eICU L1', '/path/to/L1/'), ('eICU L2', '/path/to/L2/')]
    dashboard_path : where to save the combined HTML file
    embed_images   : embed images as base64 (True = portable, False = faster build)
    """
    import os, base64, json

    GLOBAL_PLOTS = [
        ("Training Curves",            "training_curves.png"),
        ("Performance Summary",        "auc_summary_table.png"),
        ("ROC / PR Curves",            "roc_pr_curves.png"),
        ("IG Global Bars",             "ig_global_bars.png"),
        ("EG Global Bars",             "eg_global_bars.png"),
        ("Attention Weights (global)", "attention_weights.png"),
    ]

    CASES = [
        ("High Confidence",  "High_Confidence"),   
        ("False Positive",             "False_Positive"),
        ("False Negative",             "False_Negative"),
        ("Borderline",                 "Borderline"),
        ("Pure Trigger",               "Pure_Trigger"),                 
        ("Pure Chronic",               "Pure_Chronic"),
        ("Pure Late Escalation",       "Pure_Late_Escalation"),
        ("Trigger + Chronic",          "Mixed_Trigger___Chronic"),
        ("Trigger + Escalation",       "Mixed_Trigger___Escalation"),
        ("Chronic + Escalation",       "Mixed_Chronic___Escalation"),
        ("All Three",                  "Mixed_All_Three"),
    ]

    CASE_PLOT_TYPES = [
        ("All Methods",              "all_methods_{}.png"),
        ("KernelSHAP",               "kernelshap_all_models_{}.png"),
        ("TimeSHAP",                 "timeshap_all_models_{}.png"),
        ("TimeSHAP Pruning",         "timeshap_pruning_{}.png"),
        ("KS + TS",                  "ks_ts_{}.png"),
        ("KS + TS + GT",             "ks_ts_gt_{}.png"),
        ("KS + WS + TS",             "ks_ws_ts_{}.png"),
        ("KS + WS + TS + GT",        "ks_ws_ts_gt_{}.png"),
        ("EG Cases",                 "eg_case_{}.png"),
        ("IG Cases",                 "ig_case_{}.png"),
        ("EG + IG",                  "eg_ig_{}.png"),
        ("EG + IG + GT",             "eg_ig_gt_{}.png"),
        ("XAI + Attention",          "attn_xai_{}.png"),
        ("XAI + Attention (scenario)", "attn_xai_{}_SCENARIO.png"),
        ("Attn-weighted EG",         "attn_weighted_eg_{}.png"),
        ("Attn Disagreement",        "attn_disagreement_{}.png"),
        ("Visit-level Comparison",   "attn_vs_xai_visit_{}.png"),
    ]

    def img_src(fpath):
        if not os.path.exists(fpath):
            return None
        if embed_images:
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{b64}"
        return os.path.relpath(fpath, os.path.dirname(dashboard_path))

    def img_src_sc(sc_dir, fname):
        """Try scenario-suffixed filename first, then plain filename."""
        sc_name = os.path.basename(sc_dir.rstrip('/'))
        resolved = fname.replace('_SCENARIO', f'_{sc_name}')
        fpath = os.path.join(sc_dir, resolved)
        if not os.path.exists(fpath):
            fpath = os.path.join(sc_dir, fname.replace('_SCENARIO', ''))
        return img_src(fpath)

    def _cf_html_for_case(sc_dir, case_label):
        import csv as _csv
        MODEL_KEYS_CF   = ['gru', 'lstm', 'attn_gru', 'attn_lstm', 'cnn']
        MODEL_LABELS_CF = ['GRU', 'LSTM', 'AttnGRU', 'AttnLSTM', 'CNN']
        parts = []
        for mkey, mlabel in zip(MODEL_KEYS_CF, MODEL_LABELS_CF):
            fpath = os.path.join(sc_dir, f'cf_edits_{mkey}.csv')
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, newline='', encoding='utf-8') as f:
                    rows = list(_csv.reader(f))
                if not rows:
                    continue
                header = rows[0]
                case_rows = [r for r in rows[1:]
                             if any(case_label.lower() in c.lower() for c in r)]
                if not case_rows:
                    continue
                th   = ''.join(f'<th>{h}</th>' for h in header)
                body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                               for r in case_rows)
                tbl  = (f'<table class="cf-table"><thead><tr>{th}</tr></thead>'
                        f'<tbody>{body}</tbody></table>')
                parts.append(f'<h3>{mlabel}</h3>{tbl}')
            except Exception:
                continue
        if not parts:
            return None
        return ('<div style="display:flex;flex-direction:column;gap:24px;">'
                + ''.join(parts) + '</div>')

    # Build data for all scenarios
    all_scenarios = []
    for sc_label, sc_dir in scenarios:
        global_tabs = []
        for label, fname in GLOBAL_PLOTS:
            src = img_src(os.path.join(sc_dir, fname))
            if src:
                global_tabs.append({"label": label, "src": src})

        case_data = []
        for case_label, case_slug in CASES:
            plots = []
            for plot_label, pattern in CASE_PLOT_TYPES:
                src = img_src_sc(sc_dir, pattern.format(case_slug))
                if src:
                    plots.append({"label": plot_label, "src": src, "type": "img"})
            cf_html = _cf_html_for_case(sc_dir, case_label)
            if cf_html:
                plots.append({"label": "Counterfactuals", "html": cf_html, "type": "table"})
            has_img = any(p.get("type") == "img" for p in plots)
            if plots and has_img:
                case_data.append({"label": case_label, "slug": case_slug, "plots": plots})

        all_scenarios.append({
            "label":   sc_label,
            "global":  global_tabs,
            "cases":   case_data,
        })

    data_json = json.dumps(all_scenarios)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EHR XAI Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  :root {{
    --bg: #0f1117; --surface: #181c27; --border: #2a2f3d;
    --accent: #4f8ef7; --accent2: #e94560; --text: #d4d8e8; --muted: #6b7280;
    --radius: 8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'IBM Plex Sans', sans-serif; background: var(--bg);
          color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }}

  /* ── Scenario tabs ── */
  .scenario-bar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 0 0 0 24px; display: flex; align-items: stretch;
    position: sticky; top: 0; z-index: 200; min-height: 49px;
    overflow: hidden;
  }}
  .scenario-bar .logo {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
    color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0 16px 0 0; margin-right: 12px; flex-shrink: 0;
    border-right: 1px solid var(--border);
    display: flex; align-items: center;
  }}
  #sc-tabs {{
    display: flex; align-items: stretch; flex-direction: row;
    overflow-x: auto; overflow-y: hidden; flex: 1;
    scrollbar-width: none;
  }}
  #sc-tabs::-webkit-scrollbar {{ display: none; }}
  .sc-tab {{
    padding: 0 18px; cursor: pointer; font-size: 13px; font-weight: 400;
    color: var(--muted); border-bottom: 3px solid transparent;
    transition: all 0.15s; white-space: nowrap; flex-shrink: 0;
    display: flex; align-items: center;
  }}
  .sc-tab:hover {{ color: var(--text); }}
  .sc-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }}

  /* ── Layout ── */
  .layout {{ display: flex; flex: 1; overflow: hidden; height: calc(100vh - 49px); }}

  /* ── Sidebar ── */
  aside {{
    width: 220px; min-width: 220px; background: var(--surface);
    border-right: 1px solid var(--border); overflow-y: auto; padding: 16px 0;
  }}
  .sidebar-section {{
    padding: 6px 16px 4px; font-size: 10px; font-family: 'IBM Plex Mono', monospace;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 8px;
  }}
  .sidebar-item {{
    padding: 7px 16px; cursor: pointer; font-size: 13px;
    border-left: 3px solid transparent; transition: all 0.15s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .sidebar-item:hover {{ background: rgba(79,142,247,0.08); color: #fff; }}
  .sidebar-item.active {{
    border-left-color: var(--accent); background: rgba(79,142,247,0.12);
    color: var(--accent); font-weight: 600;
  }}

  /* ── Main ── */
  main {{ flex: 1; overflow-y: auto; padding: 24px 28px; }}
  .section-title {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--muted); margin-bottom: 14px;
    padding-bottom: 6px; border-bottom: 1px solid var(--border);
  }}

  /* ── Plot tabs ── */
  .tab-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 18px; }}
  .tab-btn {{
    padding: 5px 12px; border-radius: 20px; border: 1px solid var(--border);
    background: transparent; color: var(--muted); font-size: 12px;
    font-family: 'IBM Plex Sans', sans-serif; cursor: pointer; transition: all 0.15s;
  }}
  .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .tab-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }}

  /* ── Image ── */
  .img-container {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px; text-align: center;
  }}
  .img-container img {{ max-width: 100%; height: auto; border-radius: 4px; cursor: zoom-in; }}
  .img-label {{ font-size: 11px; color: var(--muted); margin-top: 8px; font-family: 'IBM Plex Mono', monospace; }}
  .empty {{ color: var(--muted); font-size: 13px; padding: 48px; text-align: center; font-style: italic; }}

  /* ── Lightbox ── */
  #lightbox {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92);
    z-index: 1000; align-items: center; justify-content: center; cursor: zoom-out;
  }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 96vw; max-height: 96vh; border-radius: 4px; box-shadow: 0 8px 48px rgba(0,0,0,0.8); }}
  #lb-close {{ position: fixed; top: 16px; right: 20px; color: #fff; font-size: 28px; cursor: pointer; opacity: 0.7; }}
  #lb-close:hover {{ opacity: 1; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<div class="scenario-bar">
  <div class="logo">EHR XAI</div>
  <div id="sc-tabs"></div>
</div>

<div class="layout">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>

<div id="lightbox">
  <span id="lb-close">&#x2715;</span>
  <img id="lb-img" src="" alt="">
</div>

<script>
const SCENARIOS = {data_json};

let currentSc      = 0;
let currentSection = '__global__';
let currentPlot    = 0;

function buildScenarioTabs() {{
  const bar = document.getElementById('sc-tabs');
  bar.innerHTML = '';
  SCENARIOS.forEach((sc, i) => {{
    const tab = document.createElement('div');
    tab.className = 'sc-tab' + (i === currentSc ? ' active' : '');
    tab.textContent = sc.label;
    tab.onclick = () => {{ currentSc = i; currentSection = '__global__'; currentPlot = 0; render(); }};
    bar.appendChild(tab);
  }});
}}

function buildSidebar() {{
  const sc    = SCENARIOS[currentSc];
  const aside = document.getElementById('sidebar');
  aside.innerHTML = '';

  const gl = document.createElement('div');
  gl.className = 'sidebar-section'; gl.textContent = 'Global';
  aside.appendChild(gl);
  const gi = document.createElement('div');
  gi.className = 'sidebar-item' + (currentSection === '__global__' ? ' active' : '');
  gi.textContent = 'Overview';
  gi.onclick = () => {{ currentSection = '__global__'; currentPlot = 0; render(); }};
  aside.appendChild(gi);

  const cl = document.createElement('div');
  cl.className = 'sidebar-section'; cl.textContent = 'Case Studies';
  aside.appendChild(cl);
  sc.cases.forEach(c => {{
    const item = document.createElement('div');
    item.className = 'sidebar-item' + (currentSection === c.slug ? ' active' : '');
    item.textContent = c.label; item.title = c.label;
    item.onclick = () => {{ currentSection = c.slug; currentPlot = 0; render(); }};
    aside.appendChild(item);
  }});
}}

function renderMain() {{
  const sc   = SCENARIOS[currentSc];
  const main = document.getElementById('main');
  main.innerHTML = '';

  let plots, titleText;
  if (currentSection === '__global__') {{
    plots = sc.global; titleText = sc.label + ' — Global';
  }} else {{
    const c = sc.cases.find(x => x.slug === currentSection);
    if (!c) return;
    plots = c.plots; titleText = sc.label + ' — ' + c.label;
  }}

  const title = document.createElement('div');
  title.className = 'section-title'; title.textContent = titleText;
  main.appendChild(title);

  if (!plots || plots.length === 0) {{
    main.innerHTML += '<div class="empty">No plots found.</div>'; return;
  }}

  const tabBar = document.createElement('div');
  tabBar.className = 'tab-bar';
  plots.forEach((p, i) => {{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === currentPlot ? ' active' : '');
    btn.textContent = p.label;
    btn.onclick = () => {{ currentPlot = i; renderMain(); }};
    tabBar.appendChild(btn);
  }});
  main.appendChild(tabBar);

  const p = plots[currentPlot];
  if (p.type === 'table') {{
    const cont = document.createElement('div');
    cont.className = 'cf-container';
    cont.innerHTML = p.html;
    main.appendChild(cont);
  }} else {{
    const cont = document.createElement('div');
    cont.className = 'img-container';
    const img = document.createElement('img');
    img.src = p.src; img.alt = p.label;
    img.onclick = () => {{ document.getElementById('lb-img').src = p.src; document.getElementById('lightbox').classList.add('open'); }};
    cont.appendChild(img);
    const lbl = document.createElement('div');
    lbl.className = 'img-label'; lbl.textContent = p.label + ' — click to enlarge';
    cont.appendChild(lbl);
    main.appendChild(cont);
  }}
}}

function render() {{ buildScenarioTabs(); buildSidebar(); renderMain(); }}

document.getElementById('lightbox').onclick    = () => document.getElementById('lightbox').classList.remove('open');
document.getElementById('lb-close').onclick    = e  => {{ e.stopPropagation(); document.getElementById('lightbox').classList.remove('open'); }};
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
  const sc = SCENARIOS[currentSc];
  const plots = currentSection === '__global__' ? sc.global
    : (sc.cases.find(x => x.slug === currentSection)?.plots || []);
  if (e.key === 'ArrowRight' && currentPlot < plots.length - 1) {{ currentPlot++; renderMain(); }}
  if (e.key === 'ArrowLeft'  && currentPlot > 0)                {{ currentPlot--; renderMain(); }}
}});

render();
</script>
</body>
</html>"""

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(dashboard_path) / 1e6
    print(f"Multi-scenario dashboard saved → {dashboard_path}  ({size_mb:.1f} MB)")
    return dashboard_path