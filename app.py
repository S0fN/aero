"""
Aerospace TPMS Lattice ML Surrogate — Streamlit Interface
=========================================================
Interactive forward-prediction and inverse-design explorer for
additive-manufactured lattice structures, powered by a trained
GBR surrogate (Su et al. 2025 pipeline).

Deploy: Hugging Face Spaces (Docker / Streamlit SDK)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import joblib, json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerospace Lattice ML Surrogate",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# White-theme CSS (no emojis, scientific style)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #ffffff;
    color: #1a1a1a;
}
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }

/* ── Header ── */
.app-header {
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #0a0a0a;
    margin: 0;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #555;
    margin-top: 0.15rem;
    font-style: italic;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 4px;
    margin-bottom: 0.6rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-left: 3px solid #1a1a1a;
    border-radius: 4px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
}
.metric-card .mc-label {
    font-size: 0.7rem;
    color: #777;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .mc-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0a0a0a;
    line-height: 1.2;
}
.metric-card .mc-unit {
    font-size: 0.72rem;
    color: #555;
}

/* ── Tabs ── */
[data-testid="stTab"] { font-size: 0.85rem; font-weight: 500; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #e0e0e0;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555;
    margin-top: 1.2rem;
}

/* ── Sliders & widgets ── */
.stSlider label { font-size: 0.82rem !important; }
.stSelectbox label { font-size: 0.82rem !important; }

/* ── Tables ── */
.stDataFrame { font-size: 0.82rem; }

/* ── Alert boxes ── */
.info-box {
    background: #f0f4ff;
    border-left: 3px solid #3b5bdb;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: #1a1a1a;
    margin-bottom: 0.8rem;
}
.warn-box {
    background: #fffbec;
    border-left: 3px solid #e67700;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: #1a1a1a;
}

/* ── Buttons ── */
.stButton > button {
    background: #1a1a1a;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    font-size: 0.83rem;
    font-weight: 600;
    padding: 0.45rem 1.1rem;
    letter-spacing: 0.3px;
}
.stButton > button:hover { background: #333; }

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #e8e8e8; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match notebook's FEATURE_COLS)
# ─────────────────────────────────────────────────────────────────────────────
MATERIALS = {
    "Ti6Al4V":  {"E_s": 114.0, "sigma_s": 880.0,  "rho_s": 4430.0},
    "AlSi10Mg": {"E_s":  70.0, "sigma_s": 325.0,  "rho_s": 2670.0},
    "316L_SS":  {"E_s": 193.0, "sigma_s": 480.0,  "rho_s": 7900.0},
    "PLA":      {"E_s":   3.5, "sigma_s":  65.0,  "rho_s": 1240.0},
    "TPU":      {"E_s":  0.04, "sigma_s":   8.0,  "rho_s": 1200.0},
    "GPR":      {"E_s": 1.916, "sigma_s": 64.35,  "rho_s": 1180.0},
    "HEC":      {"E_s": 1.001, "sigma_s": 30.75,  "rho_s": 1150.0},
    "HTB":      {"E_s": 1.291, "sigma_s": 36.02,  "rho_s": 1200.0},
    "DUR":      {"E_s": 1.260, "sigma_s": 32.00,  "rho_s": 1100.0},
    "BMC":      {"E_s": 1.600, "sigma_s": 45.00,  "rho_s": 1170.0},
}
PROCESSES = {
    "LPBF": {"min_wall": 0.20, "surface_penalty": 0.05, "support_risk": 0.25},
    "EBM":  {"min_wall": 0.40, "surface_penalty": 0.03, "support_risk": 0.15},
    "FDM":  {"min_wall": 0.40, "surface_penalty": 0.10, "support_risk": 0.35},
    "SLA":  {"min_wall": 0.10, "surface_penalty": 0.02, "support_risk": 0.10},
    "Material_Jetting": {"min_wall": 0.15, "surface_penalty": 0.03, "support_risk": 0.12},
}
TPMS_PARAMS = {
    "gyroid":    {"C1": 0.300, "n1": 2.10, "C2": 0.300, "n2": 1.50, "ea_factor": 1.05},
    "diamond":   {"C1": 0.350, "n1": 1.90, "C2": 0.350, "n2": 1.40, "ea_factor": 1.10},
    "primitive": {"C1": 0.200, "n1": 2.30, "C2": 0.250, "n2": 1.60, "ea_factor": 0.90},
    "iwp":       {"C1": 0.280, "n1": 2.00, "C2": 0.300, "n2": 1.50, "ea_factor": 1.00},
}
TARGET_COLS = ["E_eff_GPa", "sigma_y_MPa", "EA_vol_MJm3"]
TARGET_LABELS = {
    "E_eff_GPa":   ("Effective Stiffness",   "E*",  "GPa"),
    "sigma_y_MPa": ("Yield Strength",         "σ_y", "MPa"),
    "EA_vol_MJm3": ("Vol. Energy Absorption", "EA",  "MJ/m³"),
}

# Full feature columns (matching Section 3 of notebook)
# These must match exactly what the scaler was fitted on
FEATURE_COLS_CORE = [
    "relative_density", "cell_size_mm", "wall_thickness_mm",
    "source_Synthetic", "source_FEA", "source_Experimental",
]
TPMS_DUMMIES  = [f"tpms_family_{t}" for t in TPMS_PARAMS]
MAT_DUMMIES   = [f"material_{m}"    for m in MATERIALS]
PROC_DUMMIES  = [f"process_{p}"     for p in PROCESSES]

# Matplotlib style
PLT_STYLE = {
    "axes.facecolor":   "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.color":       "#ebebeb",
    "grid.linewidth":   0.6,
    "text.color":       "#1a1a1a",
    "axes.labelcolor":  "#1a1a1a",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
}
TPMS_COLORS = {
    "gyroid":    "#1a1a1a",
    "diamond":   "#555555",
    "primitive": "#aaaaaa",
    "iwp":       "#888888",
}
TPMS_LS = {
    "gyroid": "-", "diamond": "--", "primitive": "-.", "iwp": ":"
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading surrogate model…")
def load_surrogate():
    model_dir = Path("ml_surrogate")
    model  = joblib.load(model_dir / "gbr_surrogate.pkl")
    scaler = joblib.load(model_dir / "feature_scaler.pkl")
    meta   = json.loads((model_dir / "surrogate_meta.json").read_text())
    return model, scaler, meta

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    p = Path("unified_training_set.csv")
    if p.exists():
        return pd.read_csv(p)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(tpms, material, process, rho, cs, source="FEA", meta=None):
    """Build a feature vector matching FEATURE_COLS from the notebook."""
    wall = cs * 0.17 * (rho / 0.30) ** 0.5

    # Determine actual feature columns from metadata if available
    if meta and "feature_cols" in meta:
        feat_cols = meta["feature_cols"]
    else:
        feat_cols = FEATURE_COLS_CORE + TPMS_DUMMIES + MAT_DUMMIES + PROC_DUMMIES

    row = {c: 0.0 for c in feat_cols}

    # Numerics
    row["relative_density"]  = rho
    row["cell_size_mm"]      = cs
    row["wall_thickness_mm"] = wall

    # Source flags
    src_key = f"source_{source}"
    if src_key in row:
        row[src_key] = 1.0
    elif "source_Synthetic" in row:
        row["source_Synthetic"] = 1.0

    # One-hots
    tk = f"tpms_family_{tpms}"
    mk = f"material_{material}"
    pk = f"process_{process}"
    for k in [tk, mk, pk]:
        if k in row:
            row[k] = 1.0

    x = np.array([row[c] for c in feat_cols], dtype=np.float32).reshape(1, -1)
    return x


def predict_properties(model, scaler, tpms, material, process, rho, cs, meta=None):
    x = build_feature_vector(tpms, material, process, rho, cs, meta=meta)
    x_sc = scaler.transform(x)
    pred = model.predict(x_sc)[0]
    return {k: max(0.0, float(v)) for k, v in zip(TARGET_COLS, pred)}


def compute_manufacturability(tpms, material, process, rho, cs):
    mp = MATERIALS.get(material, {"rho_s": 4430.0})
    pp = PROCESSES.get(process, {"min_wall": 0.3, "surface_penalty": 0.05, "support_risk": 0.2})
    wall = cs * 0.17 * (rho / 0.30) ** 0.5
    wall_ok = min(1.0, wall / pp["min_wall"])
    score = wall_ok * (1.0 - pp["surface_penalty"]) * (1.0 - 0.5 * pp["support_risk"] * rho)
    return round(np.clip(score, 0.0, 1.0), 4)


def compute_sea(props, material, rho):
    rho_eff = rho * MATERIALS.get(material, {"rho_s": 4430.0})["rho_s"]
    sea = (props["EA_vol_MJm3"] * 1e6) / rho_eff / 1000.0 if rho_eff > 0 else 0.0
    return sea

# ─────────────────────────────────────────────────────────────────────────────
# Inverse-design selector (heuristic, matching notebook logic)
# ─────────────────────────────────────────────────────────────────────────────
def inverse_design(stiffness_priority, ea_priority, rho_max, material, process):
    """
    Heuristic mapping from performance targets to TPMS family + geometry.
    Mirrors the inverse-design block described in the Word guide (§2.2).
    """
    scores = {}
    for tpms, tp in TPMS_PARAMS.items():
        # Stiffness score: higher C1, lower n1 → better at moderate density
        s_stiff = tp["C1"] * (0.3 ** tp["n1"])
        # EA score: ea_factor × strength proxy
        s_ea = tp["ea_factor"] * tp["C2"] * (0.3 ** tp["n2"])
        combined = stiffness_priority * s_stiff + ea_priority * s_ea
        scores[tpms] = combined
    best_tpms = max(scores, key=scores.get)

    # Geometry: balance density vs printability
    pp = PROCESSES.get(process, {"min_wall": 0.3})
    rho_sel = min(rho_max, 0.35)  # conservative pick inside feasible window
    cs_sel  = 4.0                 # standard cell size for first release
    wall    = cs_sel * 0.17 * (rho_sel / 0.30) ** 0.5
    feasible = wall >= pp["min_wall"]
    return best_tpms, rho_sel, cs_sel, scores, feasible

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers (white theme, scientific style)
# ─────────────────────────────────────────────────────────────────────────────
def apply_style(ax, title="", xlabel="", ylabel="", legend=True):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(fontsize=7.5, framealpha=0.0, edgecolor="none")


def fig_density_sweep(model, scaler, material, process, meta):
    """Stiffness and EA vs relative density for all TPMS families."""
    rho_arr = np.linspace(0.05, 0.60, 50)
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

        for tpms in TPMS_PARAMS:
            E_vals, EA_vals, SEA_vals = [], [], []
            for rho in rho_arr:
                p = predict_properties(model, scaler, tpms, material, process, rho, 4.0, meta)
                sea = compute_sea(p, material, rho)
                E_vals.append(p["E_eff_GPa"])
                EA_vals.append(p["EA_vol_MJm3"])
                SEA_vals.append(sea)
            kw = dict(color=TPMS_COLORS[tpms], ls=TPMS_LS[tpms], lw=1.6, label=tpms)
            axes[0].plot(rho_arr, E_vals,  **kw)
            axes[1].plot(rho_arr, EA_vals, **kw)
            axes[2].plot(rho_arr, SEA_vals, **kw)

        apply_style(axes[0], f"Effective Stiffness vs Relative Density\n{material} · {process}",
                    "Relative density ρ*", "E* [GPa]")
        apply_style(axes[1], "Volumetric Energy Absorption vs Relative Density",
                    "Relative density ρ*", "EA [MJ/m³]")
        apply_style(axes[2], "Specific Energy Absorption vs Relative Density",
                    "Relative density ρ*", "SEA [kJ/kg]")
        fig.tight_layout(pad=1.5)
    return fig


def fig_radar(props_by_tpms):
    """Normalised radar chart comparing all TPMS families."""
    labels = ["E* (GPa)", "σ_y (MPa)", "EA (MJ/m³)"]
    keys   = TARGET_COLS
    N = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    all_vals = np.array([[props_by_tpms[t][k] for k in keys] for t in TPMS_PARAMS])
    maxv = all_vals.max(axis=0)
    maxv[maxv == 0] = 1.0

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        ax.set_facecolor("#ffffff")
        ax.spines["polar"].set_color("#cccccc")
        ax.grid(color="#e0e0e0", linewidth=0.6)

        for tpms in TPMS_PARAMS:
            vals = [props_by_tpms[tpms][k] / maxv[i] for i, k in enumerate(keys)]
            vals += vals[:1]
            ax.plot(angles, vals, ls=TPMS_LS[tpms], lw=2.0,
                    color=TPMS_COLORS[tpms], label=tpms)
            ax.fill(angles, vals, alpha=0.07, color=TPMS_COLORS[tpms])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.5, color="#888")
        ax.set_title("TPMS Performance Trade-off\n(normalised, ML predicted)", fontsize=9,
                     fontweight="bold", pad=14)
        ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05),
                  fontsize=7.5, framealpha=0, edgecolor="none")
        fig.tight_layout()
    return fig


def fig_property_bars(props_by_tpms):
    """Grouped bar chart of all 3 targets across TPMS families."""
    tpms_list = list(TPMS_PARAMS)
    x = np.arange(len(tpms_list))
    width = 0.27

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        grays = ["#1a1a1a", "#666666", "#aaaaaa", "#cccccc"]

        for ax, (key, (name, sym, unit)) in zip(axes, TARGET_LABELS.items()):
            vals = [props_by_tpms[t][key] for t in tpms_list]
            bars = ax.bar(x, vals, width=0.55, color=grays, edgecolor="#ffffff",
                          linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([t.capitalize() for t in tpms_list], fontsize=7.5)
            apply_style(ax, f"{name}\n[{unit}]", "", f"{sym} [{unit}]", legend=False)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7, color="#333")

        fig.tight_layout(pad=1.2)
    return fig


def fig_parity(model, scaler, meta):
    """Surrogate parity plot using Gibson-Ashby analytical values as ground truth."""
    rng = np.random.RandomState(42)
    rhos = rng.uniform(0.05, 0.60, 200)
    css  = rng.choice([2, 4, 6, 8, 10], 200).astype(float)
    tpms_list = list(TPMS_PARAMS)
    mats = list(MATERIALS)[:5]  # original 5

    actual_E, pred_E, actual_S, pred_S = [], [], [], []
    for rho, cs in zip(rhos, css):
        tpms = rng.choice(tpms_list)
        mat  = rng.choice(mats)
        tp = TPMS_PARAMS[tpms]
        mp = MATERIALS[mat]
        E_true   = tp["C1"] * mp["E_s"]    * rho ** tp["n1"]
        sig_true = tp["C2"] * mp["sigma_s"] * rho ** tp["n2"]
        p = predict_properties(model, scaler, tpms, mat, "LPBF", rho, cs, meta)
        actual_E.append(E_true)
        pred_E.append(p["E_eff_GPa"])
        actual_S.append(sig_true)
        pred_S.append(p["sigma_y_MPa"])

    actual_E = np.array(actual_E); pred_E = np.array(pred_E)
    actual_S = np.array(actual_S); pred_S = np.array(pred_S)

    from sklearn.metrics import r2_score
    r2_E = r2_score(actual_E, pred_E)
    r2_S = r2_score(actual_S, pred_S)

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

        for ax, act, prd, lbl, r2 in [
            (axes[0], actual_E, pred_E, "E* [GPa]", r2_E),
            (axes[1], actual_S, pred_S, "σ_y [MPa]", r2_S),
        ]:
            lo, hi = min(act.min(), prd.min()), max(act.max(), prd.max())
            ax.scatter(act, prd, s=7, alpha=0.5, color="#444", linewidths=0)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Ideal (1:1)")
            apply_style(ax, f"Parity Plot — {lbl}\nR² = {r2:.4f}",
                        f"Analytical {lbl}", f"ML Predicted {lbl}")
            ax.text(0.04, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top")

        fig.tight_layout(pad=1.5)
    return fig


def fig_cell_size_sensitivity(model, scaler, tpms, material, process, rho, meta):
    """Sensitivity of all targets to unit cell size."""
    cs_arr = np.linspace(1.5, 12.0, 40)
    E_vals, sig_vals, ea_vals = [], [], []
    for cs in cs_arr:
        p = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
        E_vals.append(p["E_eff_GPa"])
        sig_vals.append(p["sigma_y_MPa"])
        ea_vals.append(p["EA_vol_MJm3"])

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, vals, lbl in [
            (axes[0], E_vals,   "E* [GPa]"),
            (axes[1], sig_vals, "σ_y [MPa]"),
            (axes[2], ea_vals,  "EA [MJ/m³]"),
        ]:
            ax.plot(cs_arr, vals, lw=2, color="#1a1a1a")
            ax.axvline(4.0, color="#aaa", lw=0.8, ls="--", label="4 mm ref.")
            apply_style(ax, f"Cell Size Sensitivity — {lbl}\n{tpms} · {material} · ρ*={rho:.2f}",
                        "Cell size [mm]", lbl)
        fig.tight_layout(pad=1.2)
    return fig


def fig_material_comparison(model, scaler, tpms, process, rho, cs, meta):
    """Bar chart comparing properties across materials for a fixed design."""
    mats = ["Ti6Al4V", "AlSi10Mg", "316L_SS", "PLA", "TPU"]
    results = {}
    for mat in mats:
        if process not in ["FDM", "SLA"] or mat in ["PLA", "TPU", "GPR"]:
            try:
                results[mat] = predict_properties(model, scaler, tpms, mat, process, rho, cs, meta)
            except Exception:
                pass

    if not results:
        return None

    mats_avail = list(results)
    E_vals  = [results[m]["E_eff_GPa"]   for m in mats_avail]
    sig_vals = [results[m]["sigma_y_MPa"] for m in mats_avail]
    ea_vals  = [results[m]["EA_vol_MJm3"] for m in mats_avail]

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        grays = ["#1a1a1a", "#3d3d3d", "#666666", "#999999", "#cccccc"]

        for ax, vals, (name, sym, unit) in zip(
            axes,
            [E_vals, sig_vals, ea_vals],
            TARGET_LABELS.values()
        ):
            x = np.arange(len(mats_avail))
            bars = ax.bar(x, vals, color=grays[:len(mats_avail)],
                          edgecolor="#fff", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(mats_avail, rotation=20, ha="right", fontsize=7.5)
            apply_style(ax, f"{name} by Material\n{tpms} · ρ*={rho:.2f} · {process}",
                        "", f"{sym} [{unit}]", legend=False)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7, color="#333")
        fig.tight_layout(pad=1.2)
    return fig


def fig_pareto(model, scaler, material, process, meta):
    """Pareto front: stiffness vs energy absorption, coloured by density."""
    rho_arr = np.linspace(0.05, 0.60, 20)
    cs_arr  = [2.0, 4.0, 6.0]
    records = []
    for tpms in TPMS_PARAMS:
        for rho in rho_arr:
            for cs in cs_arr:
                p = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
                records.append({"tpms": tpms, "rho": rho, "cs": cs,
                                 "E": p["E_eff_GPa"], "EA": p["EA_vol_MJm3"]})
    df_p = pd.DataFrame(records)

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for tpms in TPMS_PARAMS:
            sub = df_p[df_p["tpms"] == tpms]
            sc = ax.scatter(sub["E"], sub["EA"], c=sub["rho"],
                            cmap="Greys", vmin=0.0, vmax=0.65,
                            s=20, alpha=0.75,
                            marker=["o","s","^","D"][list(TPMS_PARAMS).index(tpms)],
                            label=tpms, edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Relative density ρ*", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        apply_style(ax,
                    f"Design Space — Stiffness vs Energy Absorption\n{material} · {process}",
                    "Effective Stiffness E* [GPa]",
                    "Volumetric EA [MJ/m³]")
        fig.tight_layout()
    return fig


def fig_eda_from_csv(df):
    """EDA panel from unified_training_set.csv."""
    num_cols = [c for c in ["relative_density", "cell_size_mm", "E_eff_MPa",
                             "sigma_y_MPa", "EA_vol_MJm3"] if c in df.columns]
    if len(num_cols) < 2:
        return None

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 3.4))
        for ax, col in zip(axes, num_cols):
            vals = df[col].dropna()
            ax.hist(vals, bins=30, color="#1a1a1a", alpha=0.75, edgecolor="#fff", linewidth=0.3)
            ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.5)
        fig.suptitle("High-Fidelity Dataset — Variable Distributions", fontsize=10,
                     fontweight="bold", y=1.02)
        fig.tight_layout()
    return fig


def fig_corr_from_csv(df):
    """Correlation matrix from dataset."""
    num_cols = [c for c in ["relative_density", "cell_size_mm",
                             "E_eff_MPa", "sigma_y_MPa", "EA_vol_MJm3"] if c in df.columns]
    if len(num_cols) < 3:
        return None
    corr = df[num_cols].corr()

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(corr.values, cmap="RdGy", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels([c.replace("_", "\n") for c in num_cols], fontsize=7.5)
        ax.set_yticklabels([c.replace("_", "\n") for c in num_cols], fontsize=7.5)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(corr.values[i, j]) > 0.6 else "#333")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("Feature Correlation Matrix (High-Fidelity Dataset)",
                     fontsize=9, fontweight="bold", pad=8)
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, meta = load_surrogate()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

df_hf = load_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <p class="app-title">Aerospace TPMS Lattice — ML Surrogate Interface</p>
  <p class="app-subtitle">
    GBR forward surrogate for additive-manufactured lattice structures &nbsp;|&nbsp;
    Gibson-Ashby-calibrated synthetic dataset &plus; high-fidelity experimental data &nbsp;|&nbsp;
    Su et al. (2025) pipeline
  </p>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"Surrogate model failed to load: `{LOAD_ERROR}`. "
             "Ensure `ml_surrogate/gbr_surrogate.pkl`, `feature_scaler.pkl`, "
             "and `surrogate_meta.json` are present.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Session state — persist results so Generate button controls recompute
# ─────────────────────────────────────────────────────────────────────────────
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = {}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — design request
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Design Parameters")

    material = st.selectbox("Material", list(MATERIALS.keys()), index=0)
    process  = st.selectbox("AM Process", list(PROCESSES.keys()), index=0)
    tpms     = st.selectbox("TPMS Family", list(TPMS_PARAMS.keys()), index=0)
    rho      = st.slider("Relative Density ρ*", 0.05, 0.60, 0.30, 0.01,
                         format="%.2f")
    cs       = st.slider("Cell Size [mm]", 1.5, 12.0, 4.0, 0.5, format="%.1f")

    st.markdown("### Inverse Design Targets")
    stiffness_priority = st.slider("Stiffness Priority", 0.0, 1.0, 0.6, 0.05)
    ea_priority        = st.slider("Energy Absorption Priority", 0.0, 1.0, 0.4, 0.05)
    rho_max            = st.slider("Max Allowable Density", 0.10, 0.60, 0.40, 0.01)

    st.markdown("---")
    run_clicked = st.button("Generate Analysis", use_container_width=True)

    st.markdown("### Model Info")
    if "model_name" in meta:
        st.caption(f"Model: {meta.get('model_name', 'GBR')}")
    if "n_estimators" in meta:
        st.caption(f"Estimators: {meta.get('n_estimators', 300)}")
    if "r2_scores" in meta:
        for tgt, r2v in meta["r2_scores"].items():
            st.caption(f"R\u00b2 ({tgt}): {r2v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Run surrogate — on first load OR when Generate is clicked
# ─────────────────────────────────────────────────────────────────────────────
current_inputs = dict(
    material=material, process=process, tpms=tpms,
    rho=rho, cs=cs,
    stiffness_priority=stiffness_priority,
    ea_priority=ea_priority,
    rho_max=rho_max,
)

if run_clicked or (not st.session_state.results_ready):
    with st.spinner("Running surrogate model…"):
        _props    = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
        _manuf    = compute_manufacturability(tpms, material, process, rho, cs)
        _sea      = compute_sea(_props, material, rho)
        _rho_eff  = rho * MATERIALS[material]["rho_s"]
        _wall_mm  = cs * 0.17 * (rho / 0.30) ** 0.5
        _pp_min   = PROCESSES[process]["min_wall"]
        _printable = _wall_mm >= _pp_min
        _all_tpms_props = {
            t: predict_properties(model, scaler, t, material, process, rho, cs, meta)
            for t in TPMS_PARAMS
        }
        # Inverse design
        _best_tpms, _rho_sel, _cs_sel, _scores, _feasible = inverse_design(
            stiffness_priority, ea_priority, rho_max, material, process
        )
        _inv_props = predict_properties(model, scaler, _best_tpms, material, process,
                                        _rho_sel, _cs_sel, meta)
        _inv_manuf = compute_manufacturability(_best_tpms, material, process, _rho_sel, _cs_sel)
        _inv_all   = {t: predict_properties(model, scaler, t, material, process,
                                             _rho_sel, _cs_sel, meta) for t in TPMS_PARAMS}

        st.session_state.update(dict(
            props=_props, manuf=_manuf, sea=_sea,
            rho_eff=_rho_eff, wall_mm=_wall_mm, pp_min=_pp_min, printable=_printable,
            all_tpms_props=_all_tpms_props,
            best_tpms=_best_tpms, rho_sel=_rho_sel, cs_sel=_cs_sel,
            scores=_scores, inv_feasible=_feasible,
            inv_props=_inv_props, inv_manuf=_inv_manuf, inv_all=_inv_all,
            inv_stiffness_priority=stiffness_priority, inv_ea_priority=ea_priority,
            results_ready=True,
        ))
        st.session_state.last_inputs = current_inputs

# Bind display variables from session state
props           = st.session_state.props
manuf           = st.session_state.manuf
sea             = st.session_state.sea
rho_eff         = st.session_state.rho_eff
wall_mm         = st.session_state.wall_mm
pp_min          = st.session_state.pp_min
printable       = st.session_state.printable
all_tpms_props  = st.session_state.all_tpms_props
best_tpms       = st.session_state.best_tpms
rho_sel         = st.session_state.rho_sel
cs_sel          = st.session_state.cs_sel
scores          = st.session_state.scores
inv_feasible    = st.session_state.inv_feasible
inv_props       = st.session_state.inv_props
inv_manuf       = st.session_state.inv_manuf
inv_all         = st.session_state.inv_all
inv_sp          = st.session_state.inv_stiffness_priority
inv_ea          = st.session_state.inv_ea_priority

# ─────────────────────────────────────────────────────────────────────────────
# Metric strip
# ─────────────────────────────────────────────────────────────────────────────
cols = st.columns(6)
metric_data = [
    ("Effective Stiffness", f"{props['E_eff_GPa']:.4f}", "GPa"),
    ("Yield Strength",      f"{props['sigma_y_MPa']:.3f}", "MPa"),
    ("Vol. Energy Abs.",    f"{props['EA_vol_MJm3']:.4f}", "MJ/m\u00b3"),
    ("Spec. Energy Abs.",   f"{sea:.3f}", "kJ/kg"),
    ("Manufacturability",   f"{manuf:.3f}", "/ 1.00"),
    ("Effective Density",   f"{rho_eff:.1f}", "kg/m\u00b3"),
]
for col, (lbl, val, unit) in zip(cols, metric_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="mc-label">{lbl}</div>
          <div class="mc-value">{val}</div>
          <div class="mc-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

# Printability warning
if not printable:
    st.markdown(f"""
    <div class="warn-box">
    Wall thickness {wall_mm:.3f} mm is below the minimum printable wall for {process}
    ({pp_min:.2f} mm). Consider reducing cell size or increasing relative density.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Forward Prediction",
    "Inverse Design",
    "Density Sweep",
    "Material Comparison",
    "Design Space",
    "Dataset EDA",
    "Model Diagnostics",
])

# ── Tab 0: Forward Prediction ─────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-label">TPMS Family Comparison — Current Process / Material / Density / Cell Size</p>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.pyplot(fig_property_bars(all_tpms_props), use_container_width=True)
    with col_r:
        st.pyplot(fig_radar(all_tpms_props), use_container_width=True)

    st.markdown('<p class="section-label">Detailed Results — Selected Design</p>',
                unsafe_allow_html=True)

    design_row = {
        "TPMS Family":          tpms,
        "Material":             material,
        "AM Process":           process,
        "Relative Density":     f"{rho:.3f}",
        "Cell Size [mm]":       f"{cs:.1f}",
        "Wall Thickness [mm]":  f"{wall_mm:.4f}",
        "E* [GPa]":             f"{props['E_eff_GPa']:.5f}",
        "\u03c3_y [MPa]":       f"{props['sigma_y_MPa']:.4f}",
        "EA [MJ/m\u00b3]":      f"{props['EA_vol_MJm3']:.5f}",
        "SEA [kJ/kg]":          f"{sea:.3f}",
        "Manufacturability":    f"{manuf:.4f}",
        "Printable":            "Yes" if printable else "No — below min wall",
    }
    st.dataframe(pd.DataFrame([design_row]).T.rename(columns={0: "Value"}),
                 use_container_width=True)

    st.markdown('<p class="section-label">Cell Size Sensitivity</p>', unsafe_allow_html=True)
    st.pyplot(fig_cell_size_sensitivity(model, scaler, tpms, material, process, rho, meta),
              use_container_width=True)

# ── Tab 1: Inverse Design ─────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-label">Heuristic Inverse Design — Target to TPMS Selector</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="info-box">
        <b>Recommended Design</b><br>
        TPMS Family: <b>{best_tpms.upper()}</b><br>
        Relative Density: <b>{rho_sel:.2f}</b><br>
        Cell Size: <b>{cs_sel:.1f} mm</b><br>
        Feasible: <b>{"Yes" if inv_feasible else "No"}</b>
        </div>
        """, unsafe_allow_html=True)

        rec_table = {
            "E* [GPa]":          f"{inv_props['E_eff_GPa']:.4f}",
            "\u03c3_y [MPa]":    f"{inv_props['sigma_y_MPa']:.3f}",
            "EA [MJ/m\u00b3]":   f"{inv_props['EA_vol_MJm3']:.4f}",
            "Manufacturability": f"{inv_manuf:.4f}",
        }
        st.dataframe(pd.DataFrame([rec_table]).T.rename(columns={0: "Predicted"}),
                     use_container_width=True)

    with col_b:
        score_vals = [scores[t] for t in TPMS_PARAMS]
        with plt.rc_context(PLT_STYLE):
            fig_inv, ax_inv = plt.subplots(figsize=(5, 2.8))
            x = np.arange(len(TPMS_PARAMS))
            ax_inv.bar(x, score_vals,
                       color=["#1a1a1a" if t == best_tpms else "#cccccc" for t in TPMS_PARAMS],
                       edgecolor="#fff", linewidth=0.5)
            ax_inv.set_xticks(x)
            ax_inv.set_xticklabels([t.capitalize() for t in TPMS_PARAMS], fontsize=8)
            apply_style(ax_inv,
                        f"Inverse Design Score\n(stiffness w={inv_sp:.2f}, EA w={inv_ea:.2f})",
                        "", "Combined Score", legend=False)
            ax_inv.text(x[list(TPMS_PARAMS).index(best_tpms)],
                        score_vals[list(TPMS_PARAMS).index(best_tpms)] * 1.03,
                        "best", ha="center", fontsize=7.5, color="#1a1a1a")
            fig_inv.tight_layout()
        st.pyplot(fig_inv, use_container_width=True)

    st.markdown('<p class="section-label">Predicted Trade-off — Inverse Selected Design</p>',
                unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([1, 1.8])
    with col_r1:
        st.pyplot(fig_radar(inv_all), use_container_width=True)
    with col_r2:
        st.pyplot(fig_property_bars(inv_all), use_container_width=True)

# ── Tab 2: Density Sweep ─────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-label">Surrogate Sweep: Properties vs Relative Density</p>',
                unsafe_allow_html=True)
    st.markdown(f"Fixed: **{material}** \u00b7 **{process}** \u00b7 cell size = **{cs:.1f} mm**")
    st.pyplot(fig_density_sweep(model, scaler, material, process, meta),
              use_container_width=True)

    with st.expander("View sweep data table"):
        rho_arr = np.round(np.linspace(0.05, 0.60, 23), 3)
        sweep_rows = []
        for rho_i in rho_arr:
            p = predict_properties(model, scaler, tpms, material, process, rho_i, cs, meta)
            m = compute_manufacturability(tpms, material, process, rho_i, cs)
            s = compute_sea(p, material, rho_i)
            sweep_rows.append({
                "\u03c1*":       rho_i,
                "E* [GPa]":      round(p["E_eff_GPa"], 5),
                "\u03c3_y [MPa]": round(p["sigma_y_MPa"], 4),
                "EA [MJ/m\u00b3]": round(p["EA_vol_MJm3"], 5),
                "SEA [kJ/kg]":   round(s, 3),
                "Manuf.":        round(m, 4),
            })
        st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True)

# ── Tab 3: Material Comparison ───────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-label">Material Comparison — Fixed Geometry</p>',
                unsafe_allow_html=True)
    fig_mat = fig_material_comparison(model, scaler, tpms, process, rho, cs, meta)
    if fig_mat:
        st.pyplot(fig_mat, use_container_width=True)
    else:
        st.info("No compatible materials found for the selected process.")

# ── Tab 4: Design Space ───────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-label">Pareto Landscape — Stiffness vs Energy Absorption</p>',
                unsafe_allow_html=True)
    st.markdown(f"Fixed: **{material}** \u00b7 **{process}** \u00b7 varying \u03c1* and cell size across all TPMS families")
    with st.spinner("Computing design space\u2026"):
        st.pyplot(fig_pareto(model, scaler, material, process, meta),
                  use_container_width=True)
    st.markdown("""
    <div class="info-box">
    Each point represents one design configuration (TPMS family \u00d7 relative density \u00d7 cell size).
    Colour encodes relative density. Use this to identify Pareto-optimal regions before
    committing to fabrication.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 5: Dataset EDA ────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-label">High-Fidelity Dataset — Exploratory Analysis</p>',
                unsafe_allow_html=True)
    if df_hf is not None:
        st.markdown(f"Dataset: **{len(df_hf):,} rows** \u00b7 **{len(df_hf.columns)} columns**")
        col_e1, col_e2 = st.columns([1.6, 1])
        with col_e1:
            fig_eda = fig_eda_from_csv(df_hf)
            if fig_eda:
                st.pyplot(fig_eda, use_container_width=True)
        with col_e2:
            fig_corr = fig_corr_from_csv(df_hf)
            if fig_corr:
                st.pyplot(fig_corr, use_container_width=True)
        with st.expander("Preview dataset"):
            st.dataframe(df_hf.head(50), use_container_width=True)
        if "source" in df_hf.columns:
            src_counts = df_hf["source"].value_counts()
            with plt.rc_context(PLT_STYLE):
                fig_src, ax_src = plt.subplots(figsize=(5, 2.8))
                ax_src.bar(src_counts.index, src_counts.values, color="#1a1a1a",
                           edgecolor="#fff", linewidth=0.5)
                apply_style(ax_src, "Dataset Composition by Source", "Source", "Count", legend=False)
                fig_src.tight_layout()
            st.pyplot(fig_src, use_container_width=True)
    else:
        st.info("`unified_training_set.csv` not found in the Space root. "
                "Upload it alongside the model files to enable EDA.")

# ── Tab 6: Model Diagnostics ──────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-label">Surrogate Parity — ML vs Gibson-Ashby Analytical</p>',
                unsafe_allow_html=True)
    with st.spinner("Computing parity\u2026"):
        st.pyplot(fig_parity(model, scaler, meta), use_container_width=True)
    st.markdown('<p class="section-label">Model Metadata</p>', unsafe_allow_html=True)
    st.json(meta)
    if "lucie_validation" in meta:
        st.markdown('<p class="section-label">LUCIE Literature Validation</p>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(meta["lucie_validation"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small style='color:#aaa'>GBR surrogate trained on 10,000 Gibson-Ashby synthetic samples "
    "+ high-fidelity experimental data. Su et al. (2025) \u2014 "
    "<em>Generative AI in Lattice Structure Design for Additive Manufacturing</em>.</small>",
    unsafe_allow_html=True
)