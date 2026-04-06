import io
import os
import tempfile
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mplsoccer import Pitch

from data_utils import bool01
from ui_theme import THEMES


CHART_REQUIREMENTS: Dict[str, List[str]] = {
    "Delivery Heatmap": ["set_piece_type", "x2", "y2"],
    "Delivery End Scatter": ["set_piece_type", "x2", "y2"],
    "Outcome Distribution": ["set_piece_type"],
    "Target Zone Breakdown": ["x2", "y2"],
    "First Contact Win By Zone": ["x2", "y2"],
    "Routine Breakdown": ["set_piece_type"],
    "Shot Map": ["x", "y"],
    "Second Ball Map": ["x", "y"],
    "Defensive Vulnerability Map": ["x", "y"],
    "Taker Profile": ["set_piece_type"],
    "Structure Zone Averages": [],
}


def make_pitch(theme_name="The Athletic Dark"):
    theme = THEMES[theme_name]
    stripe = True if theme.get("pitch_stripe") else False
    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=64,
        line_zorder=2,
        pitch_color=theme["pitch"],
        line_color=theme["pitch_lines"],
        stripe=stripe,
        stripe_color=theme.get("pitch_stripe"),
    )


def apply_flip_y(df: pd.DataFrame, flip_y: bool = False) -> pd.DataFrame:
    out = df.copy()
    if not flip_y:
        return out

    for c in ["y", "y2", "y3"]:
        if c in out.columns:
            out[c] = 64 - pd.to_numeric(out[c], errors="coerce")

    return out


def themed_bar(ax, theme):
    ax.set_facecolor(theme["panel"])
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])
    ax.tick_params(colors=theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    ax.xaxis.label.set_color(theme["muted"])
    ax.title.set_color(theme["text"])


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=260, bbox_inches="tight", pad_inches=0.25)
    buf.seek(0)
    return buf.getvalue()


def save_report_pdf(figures: List, filename="set_piece_report.pdf"):
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, filename)
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
    with open(pdf_path, "rb") as f:
        return f.read()


def infer_zone_from_xy(x, y):
    try:
        x = float(x)
        y = float(y)
    except Exception:
        return "unknown"

    if x >= 88 and y <= 18:
        return "near_post"
    if x >= 88 and y >= 46:
        return "far_post"
    if x >= 84 and 18 < y < 46:
        return "central"
    return "edge"


def get_set_piece_series(df: pd.DataFrame) -> pd.Series:
    if "set_piece_type" in df.columns:
        return df["set_piece_type"].fillna("unknown").astype(str)
    if "event" in df.columns:
        return df["event"].fillna("unknown").astype(str)
    if "outcome" in df.columns:
        return df["outcome"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)


def get_target_zone_series(df: pd.DataFrame) -> pd.Series:
    if "target_zone" in df.columns:
        return df["target_zone"].fillna("unknown").astype(str)

    if "x2" in df.columns and "y2" in df.columns:
        return df.apply(lambda r: infer_zone_from_xy(r.get("x2"), r.get("y2")), axis=1)

    return pd.Series(["unknown"] * len(df), index=df.index)


def get_first_contact_win_series(df: pd.DataFrame) -> pd.Series:
    if "first_contact_win" in df.columns:
        return bool01(df["first_contact_win"])

    spt = get_set_piece_series(df).astype(str).str.lower()
    return spt.isin(["successful", "win", "won", "first_contact"]).astype(int)


def get_second_ball_win_series(df: pd.DataFrame) -> pd.Series:
    if "second_ball_win" in df.columns:
        return bool01(df["second_ball_win"])

    spt = get_set_piece_series(df).astype(str).str.lower()
    return spt.isin(["second_ball_win", "second ball win", "won_second_ball"]).astype(int)


def chart_delivery_heatmap(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    dff = apply_flip_y(df, flip_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 66)

    dd = dff.dropna(subset=["x2", "y2"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(dd["x2"], dd["y2"], ax=ax, fill=True, levels=50, alpha=0.75)
        except Exception:
            pitch.scatter(dd["x2"], dd["y2"], ax=ax, s=55, color="#38bdf8", alpha=0.75)

    ax.set_title("Delivery End Location Heatmap", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_delivery_end_scatter(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    dff = apply_flip_y(df, flip_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 66)

    dd = dff.dropna(subset=["x2", "y2"]).copy()

    if "delivery_type" in dd.columns:
        color_map = {
            "inswing": "#00C2FF",
            "outswing": "#FFD400",
            "straight": "#FF8A00",
        }
        for dtype, grp in dd.groupby("delivery_type"):
            pitch.scatter(
                grp["x2"],
                grp["y2"],
                ax=ax,
                s=85,
                color=color_map.get(str(dtype).lower(), "#E6E6E6"),
                edgecolors="white",
                linewidth=1.2,
                label=str(dtype).title(),
                alpha=0.9,
            )
        leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
        if leg:
            for t in leg.get_texts():
                t.set_color(theme["text"])
    else:
        pitch.scatter(
            dd["x2"], dd["y2"],
            ax=ax, s=85, color="#38bdf8", edgecolors="white", linewidth=1.2, alpha=0.9
        )

    ax.set_title("Delivery End Scatter", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_outcome_distribution(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(theme["bg"])

    counts = get_set_piece_series(df).astype(str).str.lower().value_counts()
    colors = []
    for x in counts.index:
        if x in ["successful", "corner", "free_kick"]:
            colors.append("#00FF6A")
        elif x in ["unsuccessful", "failed", "loss"]:
            colors.append("#FF4D4D")
        else:
            colors.append("#00C2FF")

    ax.bar(counts.index, counts.values, color=colors)
    themed_bar(ax, theme)
    ax.set_title("Set Piece Type / Outcome Distribution", fontsize=16, weight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_target_zone_breakdown(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(theme["bg"])

    counts = get_target_zone_series(df).value_counts()
    ax.bar(counts.index, counts.values, color="#38bdf8")
    themed_bar(ax, theme)
    ax.set_title("Target Zone Breakdown", fontsize=16, weight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_first_contact_win_by_zone(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])

    dd = df.copy()
    dd["zone_calc"] = get_target_zone_series(dd)
    dd["fc_win_calc"] = get_first_contact_win_series(dd)

    summary = dd.groupby("zone_calc", dropna=False)["fc_win_calc"].mean().sort_values(ascending=False) * 100
    ax.bar(summary.index.astype(str), summary.values, color="#00C2FF")
    themed_bar(ax, theme)
    ax.set_title("First Contact Win % By Zone", fontsize=16, weight="bold")
    ax.set_ylabel("Win %")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_routine_breakdown(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])

    if "routine_type" in df.columns and df["routine_type"].notna().any():
        counts = df["routine_type"].fillna("unclassified").value_counts().head(10)
    else:
        counts = get_target_zone_series(df).value_counts().head(10)

    ax.barh(counts.index[::-1], counts.values[::-1], color="#FFD400")
    themed_bar(ax, theme)
    ax.set_title("Routine Breakdown", fontsize=16, weight="bold")
    ax.set_xlabel("Count")
    return fig


def chart_shot_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    dff = apply_flip_y(df, flip_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 66)

    dd = dff.dropna(subset=["x", "y"]).copy()

    size = 160
    if "xg" in dd.columns:
        xg = pd.to_numeric(dd["xg"], errors="coerce").fillna(0)
        size = 80 + xg * 450

    pitch.scatter(
        dd["x"], dd["y"],
        ax=ax,
        s=size,
        color="#00FF6A",
        edgecolors="white",
        linewidth=1.4,
        alpha=0.92,
    )
    ax.set_title("Set Piece Shot / Event Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_second_ball_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    dff = apply_flip_y(df, flip_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 66)

    dd = dff.dropna(subset=["x", "y"]).copy()
    dd["second_ball_calc"] = get_second_ball_win_series(dd)

    win = dd[dd["second_ball_calc"] == 1]
    lose = dd[dd["second_ball_calc"] == 0]

    if len(win):
        pitch.scatter(
            win["x"], win["y"],
            ax=ax, s=120, color="#00FF6A", edgecolors="white", linewidth=1.3, label="Won"
        )
    if len(lose):
        pitch.scatter(
            lose["x"], lose["y"],
            ax=ax, s=120, facecolors="none", edgecolors="#FF4D4D", linewidth=2.0, label="Lost"
        )

    if len(win) or len(lose):
        leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2)
        if leg:
            for t in leg.get_texts():
                t.set_color(theme["text"])

    ax.set_title("Second Ball Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_defensive_vulnerability_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    dff = apply_flip_y(df, flip_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 66)

    dd = dff.dropna(subset=["x", "y"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(dd["x"], dd["y"], ax=ax, fill=True, levels=40, alpha=0.72)
        except Exception:
            pitch.scatter(dd["x"], dd["y"], ax=ax, s=75, color="#FF4D4D", edgecolors="white", linewidth=1.1)

    ax.set_title("Defensive Vulnerability Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_taker_profile(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(theme["bg"])

    if "taker" in df.columns and "sequence_id" in df.columns:
        seq_counts = df.groupby("taker")["sequence_id"].nunique().sort_values(ascending=False).head(10)
    elif "taker" in df.columns:
        seq_counts = df["taker"].value_counts().head(10)
    else:
        seq_counts = get_set_piece_series(df).value_counts().head(10)

    ax.barh(seq_counts.index[::-1], seq_counts.values[::-1], color="#38bdf8")
    themed_bar(ax, theme)
    ax.set_title("Taker / Event Profile", fontsize=16, weight="bold")
    ax.set_xlabel("Count")
    return fig


def chart_structure_zone_averages(df: pd.DataFrame, theme_name: str, flip_y: bool = False):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])

    cols = ["players_near_post", "players_far_post", "players_6yard", "players_penalty"]
    existing = [c for c in cols if c in df.columns]

    if existing:
        means = df[existing].apply(pd.to_numeric, errors="coerce").mean().fillna(0)
        labels_map = {
            "players_near_post": "Near Post",
            "players_far_post": "Far Post",
            "players_6yard": "6 Yard",
            "players_penalty": "Penalty",
        }
        labels = [labels_map[c] for c in existing]
        values = means.values
    else:
        zone_counts = get_target_zone_series(df).value_counts()
        labels = zone_counts.index.tolist()
        values = zone_counts.values

    colors = ["#00C2FF", "#FFD400", "#00FF6A", "#A78BFA"][:len(values)]
    ax.bar(labels, values, color=colors)
    themed_bar(ax, theme)
    ax.set_title("Structure / Zone Summary", fontsize=16, weight="bold")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=15)
    return fig


CHART_BUILDERS = {
    "Delivery Heatmap": chart_delivery_heatmap,
    "Delivery End Scatter": chart_delivery_end_scatter,
    "Outcome Distribution": chart_outcome_distribution,
    "Target Zone Breakdown": chart_target_zone_breakdown,
    "First Contact Win By Zone": chart_first_contact_win_by_zone,
    "Routine Breakdown": chart_routine_breakdown,
    "Shot Map": chart_shot_map,
    "Second Ball Map": chart_second_ball_map,
    "Defensive Vulnerability Map": chart_defensive_vulnerability_map,
    "Taker Profile": chart_taker_profile,
    "Structure Zone Averages": chart_structure_zone_averages,
}
