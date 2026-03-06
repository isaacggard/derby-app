"""
MLB Home Run Derby Prediction App
==================================
A professional Streamlit dashboard showcasing XGBoost Poisson + Bootstrap
Monte Carlo predictions for the 2025 MLB Home Run Derby.

Author: Isaac Gard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from PIL import Image
from pathlib import Path
import re

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="2025 HR Derby Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
HEADSHOTS_DIR = ASSETS_DIR / "headshots"

# ─────────────────────────────────────────────
# Team mapping
# ─────────────────────────────────────────────
TEAM_MAP = {
    "Byron Buxton": "MIN",
    "Junior Caminero": "TB",
    "Jazz Chisholm Jr.": "NYY",
    "Oneil Cruz": "PIT",
    "Matt Olson": "ATL",
    "Cal Raleigh": "SEA",
    "Brent Rooker": "OAK",
    "James Wood": "WSH",
}

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
ACCENT_BLUE = "#4fc3f7"
ACCENT_GOLD = "#ffd54f"
ACCENT_GREEN = "#69f0ae"
ACCENT_RED = "#ff5252"
TEXT_SECONDARY = "#9e9e9e"

ROUND_COLORS = {
    1: "#4fc3f7",  # blue
    2: "#ffd54f",  # gold
    3: "#69f0ae",  # green
}
ROUND_LABELS = {1: "Round 1", 2: "Round 2", 3: "Championship"}

# ─────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        min-width: 290px;
        max-width: 330px;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(145deg, #1a1f2e 0%, #222842 100%);
        border: 1px solid rgba(79,195,247,0.12);
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        margin-bottom: 10px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(79,195,247,0.35);
        box-shadow: 0 4px 20px rgba(79,195,247,0.08);
    }
    .metric-card .metric-label {
        color: #8a919e;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .metric-card .metric-sub {
        color: #8a919e;
        font-size: 0.78rem;
        line-height: 1.4;
    }

    /* ---- Accent colored metric values ---- */
    .metric-card .metric-value.gold { color: #ffd54f; }
    .metric-card .metric-value.blue { color: #4fc3f7; }
    .metric-card .metric-value.green { color: #69f0ae; }
    .metric-card .metric-value.red { color: #ff5252; }

    /* ---- Insight card ---- */
    .insight-card {
        background: linear-gradient(145deg, #151a26 0%, #1a2236 100%);
        border-left: 3px solid #4fc3f7;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 14px 0;
        color: #c0cad8;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .insight-card strong {
        color: #4fc3f7;
    }

    /* ---- Section headers ---- */
    .section-header {
        color: #e8eaed;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(79,195,247,0.15);
        padding-bottom: 8px;
        margin: 26px 0 14px 0;
        letter-spacing: 0.3px;
    }

    /* ---- Tables ---- */
    .stDataFrame, .stTable {
        color: #e0e0e0 !important;
    }
    thead tr th {
        background-color: #1a1f2e !important;
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    tbody tr td {
        color: #e0e0e0 !important;
    }
    tbody tr:nth-child(even) {
        background-color: rgba(26,31,46,0.5) !important;
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 0.78rem;
        padding: 28px 0 10px 0;
        border-top: 1px solid rgba(79,195,247,0.08);
        margin-top: 40px;
    }

    /* ---- Reduce default padding ---- */
    .block-container {
        padding-top: 2rem;
    }

    /* ---- Expander ---- */
    .stExpander {
        border: 1px solid rgba(79,195,247,0.1) !important;
        border-radius: 10px !important;
    }

    /* ---- Sidebar buttons: white text on dark background ---- */
    section[data-testid="stSidebar"] button {
        background-color: #1a1f2e !important;
        border: 1px solid rgba(79,195,247,0.2) !important;
    }
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span {
        color: #ffffff !important;
    }

    /* ---- Uniform blue hover outline for sidebar buttons ---- */
    section[data-testid="stSidebar"] button:hover {
        border-color: #4fc3f7 !important;
        box-shadow: 0 0 0 1px #4fc3f7 !important;
        background-color: #222842 !important;
    }

    /* ---- Probability pills: blue hover outline ---- */
    .prob-pill {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .prob-pill:hover {
        border-color: #4fc3f7 !important;
        box-shadow: 0 0 0 1px #4fc3f7;
    }

    /* ---- Force dark mode on Streamlit toolbar/header ---- */
    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    .stDeployButton, [data-testid="stStatusWidget"] {
        color: #e0e0e0 !important;
    }
</style>
"""


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert player name to filesystem-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def crop_to_square(img: Image.Image) -> Image.Image:
    """Center-crop an image to a square aspect ratio."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_image_safe(path: Path, size: tuple | None = None) -> Image.Image | None:
    """Load a PNG image, crop to square, and optionally resize.
    Loads at native resolution for maximum quality."""
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
        img = crop_to_square(img)
        if size:
            img = img.resize(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def get_headshot(player: str, size: tuple = (60, 60)) -> Image.Image | None:
    """Get a player headshot image."""
    slug = slugify(player)
    path = HEADSHOTS_DIR / f"{slug}.png"
    img = load_image_safe(path, size)
    if img is None:
        placeholder = ASSETS_DIR / "placeholder.png"
        img = load_image_safe(placeholder, size)
    return img


def get_logo(size: tuple = (220, 220)) -> Image.Image | None:
    """Load the HR Derby logo."""
    return load_image_safe(ASSETS_DIR / "hr_derby_logo.png", size)


def metric_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    """Generate HTML for a styled metric card."""
    color_cls = f" {color}" if color else ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_cls}">{value}</div>
        {sub_html}
    </div>
    """


def insight_card(text: str) -> str:
    """Generate HTML for an insight card."""
    return f'<div class="insight-card">{text}</div>'


def section_header(text: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def dark_layout(fig, title="", height=450):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=15, family="Inter"), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.7)",
        font=dict(color="#c0c8d4", size=12, family="Inter"),
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="rgba(79,195,247,0.15)",
            borderwidth=1,
            font=dict(color="#c0c8d4"),
        ),
        xaxis=dict(gridcolor="rgba(79,195,247,0.06)", zerolinecolor="rgba(79,195,247,0.1)"),
        yaxis=dict(gridcolor="rgba(79,195,247,0.06)", zerolinecolor="rgba(79,195,247,0.1)"),
    )
    return fig


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load and validate all CSV data files."""
    errors = []
    data = {}

    # ---- predicted_lambdas ----
    try:
        df = pd.read_csv(DATA_DIR / "predicted_lambdas.csv")
        required = {"Player", "Round", "RoundLength", "lambda_HRs"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"predicted_lambdas.csv missing columns: {missing}")
        else:
            data["lambdas"] = df
    except FileNotFoundError:
        errors.append("predicted_lambdas.csv not found in data/")

    # ---- final_predictions ----
    try:
        df = pd.read_csv(DATA_DIR / "final_predictions.csv")
        required = {"Player", "Top4Prob", "FinalProb", "WinProb"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"final_predictions.csv missing columns: {missing}")
        else:
            data["preds"] = df
    except FileNotFoundError:
        errors.append("final_predictions.csv not found in data/")

    # ---- MC_stats ----
    mc_path = DATA_DIR / "MC_stats.csv"
    if not mc_path.exists():
        mc_path = DATA_DIR / "mc_stats.csv"
    try:
        df = pd.read_csv(mc_path)
        required = {"Player", "Round", "MC_mean", "MC_std", "MC_p5", "MC_p25",
                     "MC_p75", "MC_p95", "lambda", "n_samples"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"MC_stats.csv missing columns: {missing}")
        else:
            data["mc"] = df
    except FileNotFoundError:
        errors.append("MC_stats.csv not found in data/")

    # ---- actual_results ----
    try:
        df = pd.read_csv(DATA_DIR / "actual_results.csv")
        rename_map = {}
        if "Name" in df.columns and "Player" not in df.columns:
            rename_map["Name"] = "Player"
        if "Score" in df.columns and "ActualHR" not in df.columns:
            rename_map["Score"] = "ActualHR"
        if rename_map:
            df = df.rename(columns=rename_map)
        required = {"Player", "Round", "ActualHR"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"actual_results.csv missing columns: {missing}")
        else:
            data["actual"] = df
    except FileNotFoundError:
        errors.append("actual_results.csv not found in data/")

    return data, errors


# ─────────────────────────────────────────────
# Page: Overall Summary
# ─────────────────────────────────────────────

def render_overall(data: dict):
    """Render the Overall Summary page."""

    preds = data["preds"]
    lambdas = data["lambdas"]
    mc = data["mc"]
    actual = data.get("actual")

    # ---- Title ----
    st.markdown(
        "<h1 style='color:#ffffff; margin-bottom:2px; font-weight:700;'>"
        "2025 Home Run Derby Forecast</h1>"
        "<p style='color:#8a919e; font-size:0.95rem; margin-top:0;'>"
        "XGBoost Poisson × Bootstrap Monte Carlo Simulation</p>",
        unsafe_allow_html=True,
    )

    # ---- KPI cards ----
    preds_sorted = preds.sort_values("WinProb", ascending=False).reset_index(drop=True)
    favorite = preds_sorted.iloc[0]

    # Most volatile
    avg_std = mc.groupby("Player")["MC_std"].mean()
    most_volatile = avg_std.idxmax()
    vol_val = avg_std.max()

    # Highest avg predicted score
    avg_lambda = lambdas.groupby("Player")["lambda_HRs"].mean()
    top_avg_player = avg_lambda.idxmax()
    top_avg_val = avg_lambda.max()

    # Actual winner
    actual_winner_str = "TBD"
    if actual is not None and 3 in actual["Round"].values:
        r3 = actual[actual["Round"] == 3].sort_values("ActualHR", ascending=False)
        if len(r3) > 0:
            actual_winner_str = r3.iloc[0]["Player"]

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card(
            "Predicted Favorite",
            favorite["Player"],
            f'{favorite["WinProb"]:.1%} win prob',
            "gold",
        ), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card(
            "Most Volatile",
            most_volatile,
            f"Avg σ = {vol_val:.2f} HRs",
            "red",
        ), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card(
            "Highest Avg Expected HRs",
            f"{top_avg_val:.1f}",
            top_avg_player,
            "blue",
        ), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card(
            "Actual Winner",
            actual_winner_str,
            "From actual results",
            "green",
        ), unsafe_allow_html=True)

    st.markdown("")

    # ---- Ranking table ----
    section_header("Advancement Probability by Player")

    rank_df = preds_sorted.copy()
    rank_df.index = range(1, len(rank_df) + 1)
    rank_df.index.name = "Rank"
    rank_df["Team"] = rank_df["Player"].map(TEAM_MAP).fillna("")

    display_df = rank_df[["Player", "Team"]].copy()
    display_df["Top 4 %"] = rank_df["Top4Prob"].apply(lambda x: f"{x:.1%}")
    display_df["Final %"] = rank_df["FinalProb"].apply(lambda x: f"{x:.1%}")
    display_df["Win %"] = rank_df["WinProb"].apply(lambda x: f"{x:.1%}")

    # Highlight Cal Raleigh (actual winner) rows
    def highlight_winner(row):
        if row["Player"] == "Cal Raleigh":
            return ["background-color: rgba(105,240,174,0.15); color: #e0e0e0"] * len(row)
        return [""] * len(row)

    styled_df = display_df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=330)

    # ---- Win probability bar chart ----
    section_header("Win Probability by Player")

    bar_df = preds_sorted.sort_values("WinProb", ascending=True)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=bar_df["Player"],
        x=bar_df["WinProb"] * 100,
        orientation="h",
        marker=dict(
            color=bar_df["WinProb"],
            colorscale=[[0, "#1a2744"], [0.4, "#2a6cb5"], [0.7, "#4fc3f7"], [1, "#ffd54f"]],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=bar_df["WinProb"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
        textfont=dict(color="#c0c8d4", size=12),
    ))
    dark_layout(fig, title="", height=370)
    fig.update_layout(
        xaxis_title="Win Probability (%)",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Predicted vs Actual comparison ----
    if actual is not None and len(actual) > 0:
        section_header("Predicted vs. Actual Results")

        comp = preds_sorted[["Player", "WinProb"]].copy()
        comp["Predicted Rank"] = range(1, len(comp) + 1)

        for rnd in [1, 2, 3]:
            rnd_lam = lambdas[lambdas["Round"] == rnd][["Player", "lambda_HRs"]].copy()
            rnd_lam = rnd_lam.rename(columns={"lambda_HRs": f"R{rnd} λ"})
            comp = comp.merge(rnd_lam, on="Player", how="left")

            rnd_act = actual[actual["Round"] == rnd][["Player", "ActualHR"]].copy()
            rnd_act = rnd_act.rename(columns={"ActualHR": f"R{rnd} Actual"})
            comp = comp.merge(rnd_act, on="Player", how="left")

        comp_display = comp.copy()
        comp_display["WinProb"] = comp_display["WinProb"].apply(lambda x: f"{x:.1%}")
        for rnd in [1, 2, 3]:
            lam_col = f"R{rnd} λ"
            act_col = f"R{rnd} Actual"
            if lam_col in comp_display.columns:
                comp_display[lam_col] = comp_display[lam_col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "–"
                )
            if act_col in comp_display.columns:
                comp_display[act_col] = comp_display[act_col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "–"
                )

        # Mark Caminero R2 with asterisk (round ended early)
        if "R2 Actual" in comp_display.columns:
            caminero_mask = comp_display["Player"] == "Junior Caminero"
            comp_display.loc[caminero_mask, "R2 Actual"] = comp_display.loc[caminero_mask, "R2 Actual"].apply(
                lambda x: f"{x}*" if x != "–" else x
            )

        # Rename lambda columns to Expected HRs for display
        rename_cols = {}
        for rnd in [1, 2, 3]:
            if f"R{rnd} λ" in comp_display.columns:
                rename_cols[f"R{rnd} λ"] = f"R{rnd} Expected HRs"
        comp_display = comp_display.rename(columns=rename_cols)

        comp_display = comp_display.rename(columns={"WinProb": "Win %"})

        display_cols = ["Predicted Rank", "Player", "Win %",
                        "R1 Expected HRs", "R1 Actual", "R2 Expected HRs", "R2 Actual", "R3 Expected HRs", "R3 Actual"]
        display_cols = [c for c in display_cols if c in comp_display.columns]

        # Highlight Cal Raleigh (actual winner) rows
        def highlight_winner_comp(row):
            if row["Player"] == "Cal Raleigh":
                return ["background-color: rgba(105,240,174,0.15); color: #e0e0e0"] * len(row)
            return [""] * len(row)

        styled_comp = comp_display[display_cols].style.apply(highlight_winner_comp, axis=1)
        st.dataframe(styled_comp, use_container_width=True, height=330, hide_index=True)

        # Footnote for asterisk
        st.markdown(
            "<p style='color:#6c7280;font-size:0.78rem;margin-top:4px;'>"
            "* Round ended early — Caminero passed Buxton's 7 HRs to clinch advancement.</p>",
            unsafe_allow_html=True,
        )

        # Winner callout
        pred_fav = favorite["Player"]
        if actual_winner_str not in ("TBD", "N/A"):
            if pred_fav == actual_winner_str:
                st.success(f"✅  The model correctly predicted **{pred_fav}** as the winner!")
            else:
                st.info(
                    f"The model predicted **{pred_fav}** as the favorite, "
                    f"but **{actual_winner_str}** won the derby."
                )

    # ---- Interpretation ----
    section_header("Interpretation")
    st.markdown(
        insight_card("""
    <strong>Model Insights</strong>
    <ol style="margin-top:10px; padding-left:20px;">
        <li style="margin-bottom:14px;">
            <strong>Cruz vs Raleigh: Favorite vs Champion</strong><br>
            Oneil Cruz entered as the model’s statistical favorite (27.4% win probability), while Cal Raleigh ranked second (18.8%) and ultimately won the derby. Raleigh projected slightly higher in expected HR totals per round, but Cruz advanced from Round 1 more frequently in simulations. Because the derby is a three-round elimination tournament, that higher advancement rate translated into the most simulated wins overall.
        </li>
        <li style="margin-bottom:14px;">
            <strong>High Tournament Volatility</strong><br>
            The derby is highly volatile. Even the top projected player won only ~27% of the 10,000 simulations, meaning the outcome of any single tournament is heavily influenced by round-to-round variance.
        </li>
        <li style="margin-bottom:14px;">
            <strong>Core Hitting Skill Drives Performance</strong><br>
            Underlying hitting quality drove most of the model’s predictions. Metrics like Zone%, wOBA, max exit velocity, and ISO ranked among the most important features, suggesting the model leaned heavily on a player’s ability to consistently generate hard contact.
        </li>
        <li>
            <strong>Derby Format Influences Outcomes</strong><br>
            Variables tied to the derby rules — including round length and the bonus-time HR distance threshold — were among the model’s more important predictors. The bonus-time threshold is the distance a home run must travel to earn additional time in a round. These features capture how format changes across seasons affect the number of pitches hitters see and the overall scoring environment.
    </ol>
    """),
    unsafe_allow_html=True
)

    # ---- Methodology ----
    with st.expander("📐 How It Works"):
        st.markdown("""
        **Custom Historical Dataset**

        The model is trained on a custom dataset of MLB Home Run Derby performances dating back to 2015. The dataset was manually assembled and curated from multiple sources, combining derby results, player statistics, and event-specific features into a structured modeling dataset.

        **XGBoost Poisson Regression**

        Each player's expected home runs per round (λ) are estimated using an XGBoost model with a Poisson objective. The model learns the relationship between player characteristics and derby performance to predict scoring outcomes for each round of the competition.

        **Bootstrap + Monte Carlo Simulation**

        To capture both model uncertainty and the randomness of the derby format:

        1. Bootstrap the training data
            * 100 resampled versions of the historical dataset are generated. An XGBoost Poisson model is trained on each sample, producing 100 slightly different fitted models.

        2. Generate round-level expectations
            * Each model predicts the expected number of home runs (λ) for every player and round in the 2025 derby field.

        3. Simulate the tournament bracket
            * For each fitted model, 100 tournaments are simulated by drawing home run totals from a Poisson distribution and advancing players through the derby bracket.

        This produces 10,000 simulated tournaments in total. Advancement and win probabilities are calculated from the aggregate results across all simulations.
        """)


# ─────────────────────────────────────────────
# Page: Player Detail
# ─────────────────────────────────────────────

def render_player(player: str, data: dict):
    """Render a player detail page."""

    preds = data["preds"]
    lambdas = data["lambdas"]
    mc = data["mc"]
    actual = data.get("actual")

    player_lambdas = lambdas[lambdas["Player"] == player].sort_values("Round")
    player_mc = mc[mc["Player"] == player].sort_values("Round")
    player_preds = preds[preds["Player"] == player]

    player_actual = None
    if actual is not None:
        player_actual = actual[actual["Player"] == player].sort_values("Round")

    # ---- Header ----
    hcol1, hcol2 = st.columns([1, 3], gap="small")
    with hcol1:
        img = get_headshot(player, size=(180, 180))
        if img:
            st.image(img, width=180, output_format="PNG")
    with hcol2:
        team = TEAM_MAP.get(player, "")
        team_str = f" <span style='color:#6c7280;font-size:1rem;font-weight:400;'>  •  {team}</span>" if team else ""
        st.markdown(
            f"<h1 style='color:#ffffff;margin-bottom:2px;font-weight:700;'>{player}{team_str}</h1>",
            unsafe_allow_html=True,
        )
        if len(player_preds) > 0:
            pp = player_preds.iloc[0]
            # Build probability pills
            pills = (
                f"<span class='prob-pill' style='display:inline-block;background:rgba(255,213,79,0.12);border:1px solid rgba(255,213,79,0.3);border-radius:20px;padding:4px 14px;margin-right:8px;cursor:default;'>"
                f"Win <b style=\"color:#ffd54f\">{pp['WinProb']:.1%}</b></span>"
                f"<span class='prob-pill' style='display:inline-block;background:rgba(79,195,247,0.12);border:1px solid rgba(79,195,247,0.3);border-radius:20px;padding:4px 14px;margin-right:8px;cursor:default;'>"
                f"Final <b style=\"color:#4fc3f7\">{pp['FinalProb']:.1%}</b></span>"
                f"<span class='prob-pill' style='display:inline-block;background:rgba(105,240,174,0.12);border:1px solid rgba(105,240,174,0.3);border-radius:20px;padding:4px 14px;cursor:default;'>"
                f"Top 4 <b style=\"color:#69f0ae\">{pp['Top4Prob']:.1%}</b></span>"
            )
            st.markdown(pills, unsafe_allow_html=True)

    st.markdown("")

    # ---- Expected HR cards with bootstrap CI ----
    section_header("Expected HRs by Round")

    # Row 1: Predicted lambda with bootstrap CI
    cols = st.columns(3)
    for i, rnd in enumerate([1, 2, 3]):
        row = player_lambdas[player_lambdas["Round"] == rnd]
        if len(row) > 0:
            r = row.iloc[0]
            lam_val = f"{r['lambda_HRs']:.1f}"
            # Construct subtitle with bootstrap CI if available
            sub_parts = []
            if "lambda_boot_p5" in r.index and "lambda_boot_p95" in r.index:
                p5 = r.get("lambda_boot_p5", np.nan)
                p95 = r.get("lambda_boot_p95", np.nan)
                if pd.notna(p5) and pd.notna(p95):
                    sub_parts.append(f"90% CI: [{p5:.1f}, {p95:.1f}]")
        else:
            lam_val = "–"
            sub_parts = []

        with cols[i]:
            st.markdown(metric_card(
                f"Round {rnd} Expected HRs",
                lam_val,
                " · ".join(sub_parts) if sub_parts else "",
            ), unsafe_allow_html=True)

    # Row 2: Actual HRs
    if player_actual is not None and len(player_actual) > 0:
        cols2 = st.columns(3)
        for i, rnd in enumerate([1, 2, 3]):
            act_row = player_actual[player_actual["Round"] == rnd]
            if len(act_row) > 0:
                act_val = str(int(act_row.iloc[0]["ActualHR"]))
            else:
                act_val = "–"
            with cols2[i]:
                st.markdown(metric_card(
                    f"Round {rnd} Actual",
                    act_val,
                ), unsafe_allow_html=True)

    # ---- Poisson distribution chart ----
    section_header("Predicted HR Distributions by Round")

    fig = go.Figure()
    actual_annotations = []

    for rnd in [1, 2, 3]:
        row = player_lambdas[player_lambdas["Round"] == rnd]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lam = r["lambda_HRs"]

        # Use bootstrap mean lambda if available for the theoretical curve
        lam_curve = lam
        if "lambda_boot_mean" in r.index and pd.notna(r.get("lambda_boot_mean")):
            lam_curve = r["lambda_boot_mean"]

        mc_row = player_mc[player_mc["Round"] == rnd]
        if len(mc_row) > 0:
            x_max = int(mc_row.iloc[0]["MC_p95"]) + 8
        else:
            x_max = int(lam + 5 * np.sqrt(lam)) + 2
        x_max = max(x_max, 10)

        x = np.arange(0, x_max + 1)
        pmf = poisson.pmf(x, lam_curve)
        pmf_pct = pmf * 100  # Convert to percentage

        color = ROUND_COLORS[rnd]
        r_hex, g_hex, b_hex = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        fig.add_trace(go.Scatter(
            x=x, y=pmf_pct,
            mode="lines+markers",
            name=f"{ROUND_LABELS[rnd]} (λ={lam_curve:.1f})",
            line=dict(color=color, width=2.5),
            marker=dict(size=4, color=color),
            fill="tozeroy",
            fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.07)",
            hovertemplate="%{x} HRs: %{y:.1f}%<extra>" + ROUND_LABELS[rnd] + "</extra>",
        ))

        # Collect actual HR annotation
        if player_actual is not None:
            act_row = player_actual[player_actual["Round"] == rnd]
            if len(act_row) > 0:
                act_hr = int(act_row.iloc[0]["ActualHR"])
                actual_annotations.append((rnd, act_hr, color))

    # Add actual HR vertical lines with staggered y-positions
    y_offsets = [0.96, 0.86, 0.76]
    for idx, (rnd, act_hr, color) in enumerate(actual_annotations):
        y_pos = y_offsets[idx] if idx < len(y_offsets) else 0.96 - idx * 0.10
        fig.add_vline(
            x=act_hr, line_dash="dash", line_color=color, line_width=2, opacity=0.7,
        )
        fig.add_annotation(
            x=act_hr, y=y_pos, yref="paper",
            text=f"R{rnd} Actual: {act_hr}",
            showarrow=True, arrowhead=0, arrowcolor=color,
            ax=35, ay=0,
            font=dict(color=color, size=10),
            bgcolor="rgba(14,17,23,0.8)", borderpad=3,
        )

    dark_layout(fig, title="Poisson PMF by Round", height=430)
    fig.update_layout(
        xaxis_title="Home Runs",
        yaxis_title="Probability (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Player comparison ----
    section_header("Compare with Another Player")
    all_players = sorted(preds["Player"].unique())
    other_players = [p for p in all_players if p != player]
    compare_to = st.selectbox("Select a player to compare:", other_players, key=f"compare_{player}")

    if compare_to:
        comp_preds = preds[preds["Player"] == compare_to]
        if len(comp_preds) > 0 and len(player_preds) > 0:
            pp = player_preds.iloc[0]
            cp = comp_preds.iloc[0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card(
                    player,
                    f"{pp['WinProb']:.1%}",
                    "Win Probability",
                    "gold",
                ), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card(
                    compare_to,
                    f"{cp['WinProb']:.1%}",
                    "Win Probability",
                    "blue",
                ), unsafe_allow_html=True)

            # Round 1 Poisson comparison
            p1_lam_row = player_lambdas[player_lambdas["Round"] == 1]
            c_lambdas = lambdas[lambdas["Player"] == compare_to]
            p2_lam_row = c_lambdas[c_lambdas["Round"] == 1]

            if len(p1_lam_row) > 0 and len(p2_lam_row) > 0:
                lam1 = p1_lam_row.iloc[0]["lambda_HRs"]
                lam2 = p2_lam_row.iloc[0]["lambda_HRs"]

                x_max = int(max(lam1, lam2) + 5 * np.sqrt(max(lam1, lam2))) + 2
                p_win = 0.0
                for k1 in range(0, x_max + 1):
                    for k2 in range(0, x_max + 1):
                        if k1 > k2:
                            p_win += poisson.pmf(k1, lam1) * poisson.pmf(k2, lam2)

                st.markdown(insight_card(
                    f"<strong>Round 1 Head-to-Head:</strong> "
                    f"{player} has a <b>{p_win:.1%}</b> chance of out-homering "
                    f"{compare_to} in Round 1 "
                    f"(λ = {lam1:.1f} vs λ = {lam2:.1f})."
                ), unsafe_allow_html=True)

    # ---- Interpretation card ----
    section_header("Interpretation")

    # Player-specific interpretations comparing predictions to actual results
    interpretations = {
        "Oneil Cruz": (
            "The model pegged Cruz as the overall favorite with a 27.4% win probability and the highest "
            "Round 1 prediction (19.0). He delivered in R1 with a derby-best 21 HRs but "
            "fell to 13 in R2, below his expected 15. He was eliminated in the semifinal by the eventual champion Cal Raleigh. "
            "The model was right about his power ceiling but the R2 drop-off led to his dismissal."
        ),
        "Cal Raleigh": (
            "Raleigh was the model's #2 pick with an 18.8% win probability, and he went on to win the "
            "entire derby. He hit 17 in R1, 19 in R2, and 18 in the Championship. His actual scores closely tracked the model's "
            "expectations across all three rounds, making him one of the best-predicted players."
        ),
        "Junior Caminero": (
            "Caminero was ranked #3 with a 15.2% win probability. He outperformed in R1 with 21 HRs "
            "(predicted 17.4), his R2 ended early at 8 after passing Buxton's 7. He reached the "
            "Championship and hit 15 (predicted 16.6), losing to Raleigh. The model correctly identified "
            "him as a top-4 contender but underestimated his R1 performance."
        ),
        "James Wood": (
            "Wood was ranked #4 with a 15.0% win probability and a strong R1 prediction (λ = 19.1). "
            "He hit 16 in R1, falling short of expectations, and was eliminated after the first round. "
            "The model overestimated his performance, his actual R1 output landed below the model's lower "
            "confidence interval."
        ),
        "Brent Rooker": (
            "Rooker was ranked #5 with a 9.3% win probability and an R1 prediction of 15.4. He hit 17 "
            "in R1, slightly beating his prediction, but it wasn't enough to advance past the first round. "
            "The model accurately captured Rooker as a mid-tier contender."
        ),
        "Matt Olson": (
            "Olson was ranked #6 with an 8.0% win probability. He hit 15 in R1 against a prediction "
            "of 16.1, close to target but not enough to advance. The model correctly placed as a mid-tier contender."
        ),
        "Byron Buxton": (
            "Buxton was ranked #7 with just a 4.2% win probability, but he far exceeded expectations in R1 with 20 HRs "
            "(predicted 14.8). He advanced to R2 but scored only 7, falling below his predicted 13.7. The model significantly " 
            "underestimated his R1 power but was right that he was unlikely to win it all."
        ),
        "Jazz Chisholm Jr.": (
            "Chisholm was the model's last-place pick with just a 2.2% win probability and an R1 prediction "
            "of 13.0. He hit only 3 home runs in R1, the worst performance of the derby by a wide margin. "
            "The model was directionally correct that he was the weakest contender, though even the model "
            "didn't predict such a low score."
        ),
    }

    interp_text = interpretations.get(
        player,
        f"No detailed interpretation available for {player}."
    )
    st.markdown(insight_card(
        f"<strong>{player} Analysis:</strong> {interp_text}"
    ), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────

def render_sidebar(players: list) -> str:
    """Build the sidebar navigation, returns the selected page."""

    with st.sidebar:
        # Logo
        logo = get_logo(size=(260, 260))
        if logo:
            st.image(logo, use_container_width=True, output_format="PNG")

        st.markdown(
            "<h3 style='text-align:center;color:#ffd54f;margin:0 0 4px 0;font-weight:700;'>"
            "2025 HR Derby Prediction Dashboard</h3>",
            unsafe_allow_html=True,
        )

        if st.button("Overall Summary", key="nav_overall", use_container_width=True):
            st.session_state["selected"] = "Overall"

        st.markdown("---")
        st.markdown(
            "<p style='color:#6c7280;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:1.8px;margin-bottom:6px;font-weight:600;'>Players</p>",
            unsafe_allow_html=True,
        )

        for player in players:
            col_img, col_name = st.columns([1, 4])
            with col_img:
                img = get_headshot(player, size=(34, 34))
                if img:
                    st.image(img, width=34, output_format="PNG")
            with col_name:
                team = TEAM_MAP.get(player, "")
                label = f"{player}" + (f"  ({team})" if team else "")
                if st.button(label, key=f"nav_{player}", use_container_width=True):
                    st.session_state["selected"] = player

        st.markdown("---")

        st.markdown(
            "<div style='text-align:center;color:#444;font-size:0.7rem;padding:8px 0;'>"
            "Built by <b>Isaac Gard</b><br>Last updated Mar 2026</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("selected", "Overall")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    data, errors = load_data()

    if errors:
        for err in errors:
            st.error(f"⚠️ Data error: {err}")
        st.stop()

    players = data["preds"].sort_values("WinProb", ascending=False)["Player"].tolist()

    if "selected" not in st.session_state:
        st.session_state["selected"] = "Overall"

    selected = render_sidebar(players)

    if selected == "Overall":
        render_overall(data)
    elif selected in players:
        render_player(selected, data)
    else:
        st.warning(f"Unknown page: {selected}")

    st.markdown(
        '<div class="footer-text">Built by <b>Isaac Gard</b> &nbsp;•&nbsp; '
        '2025 MLB Home Run Derby Forecast &nbsp;•&nbsp; Last updated Mar 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
