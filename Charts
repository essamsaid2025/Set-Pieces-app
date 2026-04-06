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
    "Delivery End Scatter": ["set_piece_type", "x2", "y2", "delivery_type", "side"],
    "Outcome Distribution": ["outcome"],
    "Target Zone Breakdown": ["target_zone"],
    "First Contact Win By Zone": ["target_zone", "first_contact_win"],
    "Routine Breakdown": ["routine_type"],
    "Shot Map": ["phase", "x", "y"],
    "xG By Routine": ["routine_type", "xg"],
    "Second Ball Map": ["phase", "x", "y"],
    "Defensive Vulnerability Map": ["set_piece_type", "phase", "x", "y", "team"],
    "Taker Profile": ["taker", "sequence_id"],
    "Structure Zone Averages": ["players_near_post", "players_far_post", "players_6yard", "players_penalty"],
}


def make_pitch(theme_name="The Athletic Dark"):
    theme = THEMES[theme_name]
    stripe = True if theme.get("pitch_stripe") else False
    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=100,
        line_zorder=2,
        pitch_color=theme["pitch"],
        line_color=theme["pitch_lines"],
        stripe=stripe,
        stripe_color=theme.get("pitch_stripe"),
    )


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


def chart_delivery_heatmap(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    dd = df.dropna(subset=["x2", "y2"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(dd["x2"], dd["y2"], ax=ax, fill=True, levels=50, alpha=0.75)
        except Exception:
            pitch.scatter(dd["x2"], dd["y2"], ax=ax, s=55, color="#38bdf8", alpha=0.75)
    ax.set_title("Delivery End Location Heatmap", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_delivery_end_scatter(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    dd = df.dropna(subset=["x2", "y2"]).copy()
    color_map = {"inswing": "#00C2FF", "outswing": "#FFD400", "straight": "#FF8A00"}
    for dtype, grp in dd.groupby("delivery_type"):
        pitch.scatter(
            grp["x2"], grp["y2"], ax=ax, s=85,
            color=color_map.get(dtype, "#E6E6E6"), edgecolors="white", linewidth=1.2,
            label=str(dtype).title(), alpha=0.9
        )
    leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    if leg:
        for t in leg.get_texts():
            t.set_color(theme["text"])
    ax.set_title("Delivery End Scatter", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_outcome_distribution(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(theme["bg"])
    counts = df["outcome"].fillna("unknown").value_counts()
    colors = ["#00FF6A" if x == "successful" else "#FF4D4D" if x == "unsuccessful" else "#00C2FF" for x in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    themed_bar(ax, theme)
    ax.set_title("Outcome Distribution", fontsize=16, weight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_target_zone_breakdown(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(theme["bg"])
    counts = df["target_zone"].fillna("unknown").value_counts()
    ax.bar(counts.index, counts.values, color="#38bdf8")
    themed_bar(ax, theme)
    ax.set_title("Target Zone Breakdown", fontsize=16, weight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_first_contact_win_by_zone(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    dd = df.dropna(subset=["target_zone", "first_contact_win"]).copy()
    dd["first_contact_win"] = bool01(dd["first_contact_win"])
    summary = dd.groupby("target_zone", dropna=False)["first_contact_win"].mean().sort_values(ascending=False) * 100
    ax.bar(summary.index.astype(str), summary.values, color="#00C2FF")
    themed_bar(ax, theme)
    ax.set_title("First Contact Win % By Zone", fontsize=16, weight="bold")
    ax.set_ylabel("Win %")
    ax.tick_params(axis="x", rotation=25)
    return fig


def chart_routine_breakdown(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    counts = df["routine_type"].fillna("unclassified").value_counts().head(10)
    ax.barh(counts.index[::-1], counts.values[::-1], color="#FFD400")
    themed_bar(ax, theme)
    ax.set_title("Routine Breakdown", fontsize=16, weight="bold")
    ax.set_xlabel("Count")
    return fig


def chart_shot_map(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    dd = df[df["phase"] == "shot"].dropna(subset=["x", "y"]).copy()
    size = 160
    if "xg" in dd.columns:
        xg = pd.to_numeric(dd["xg"], errors="coerce").fillna(0)
        size = 80 + xg * 450
    pitch.scatter(dd["x"], dd["y"], ax=ax, s=size, color="#00FF6A", edgecolors="white", linewidth=1.4, alpha=0.92)
    ax.set_title("Set Piece Shot Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_xg_by_routine(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    dd = df.dropna(subset=["routine_type"]).copy()
    dd["xg"] = pd.to_numeric(dd["xg"], errors="coerce")
    summary = dd.groupby("routine_type", dropna=False)["xg"].sum().sort_values(ascending=False).head(10)
    ax.barh(summary.index[::-1], summary.values[::-1], color="#00FF6A")
    themed_bar(ax, theme)
    ax.set_title("xG By Routine", fontsize=16, weight="bold")
    ax.set_xlabel("Total xG")
    return fig


def chart_second_ball_map(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    dd = df[df["phase"] == "second_ball"].dropna(subset=["x", "y"]).copy()
    if "second_ball_win" in dd.columns:
        dd["second_ball_win"] = bool01(dd["second_ball_win"])
        win = dd[dd["second_ball_win"] == 1]
        lose = dd[dd["second_ball_win"] == 0]
        if len(win):
            pitch.scatter(win["x"], win["y"], ax=ax, s=120, color="#00FF6A", edgecolors="white", linewidth=1.3, label="Won")
        if len(lose):
            pitch.scatter(lose["x"], lose["y"], ax=ax, s=120, facecolors="none", edgecolors="#FF4D4D", linewidth=2.0, label="Lost")
        leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2)
        if leg:
            for t in leg.get_texts():
                t.set_color(theme["text"])
    else:
        pitch.scatter(dd["x"], dd["y"], ax=ax, s=120, color="#00C2FF", edgecolors="white", linewidth=1.3)
    ax.set_title("Second Ball Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_defensive_vulnerability_map(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    pitch = make_pitch(theme_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    dd = df[df["phase"].isin(["shot", "first_contact"])].dropna(subset=["x", "y"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(dd["x"], dd["y"], ax=ax, fill=True, levels=40, alpha=0.72)
        except Exception:
            pitch.scatter(dd["x"], dd["y"], ax=ax, s=75, color="#FF4D4D", edgecolors="white", linewidth=1.1)
    ax.set_title("Defensive Vulnerability Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def chart_taker_profile(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    seq_counts = df.groupby("taker")["sequence_id"].nunique().sort_values(ascending=False).head(10)
    ax.barh(seq_counts.index[::-1], seq_counts.values[::-1], color="#38bdf8")
    themed_bar(ax, theme)
    ax.set_title("Taker Profile — Set Pieces Taken", fontsize=16, weight="bold")
    ax.set_xlabel("Unique Set Pieces")
    return fig


def chart_structure_zone_averages(df: pd.DataFrame, theme_name: str):
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    cols = ["players_near_post", "players_far_post", "players_6yard", "players_penalty"]
    means = df[cols].apply(pd.to_numeric, errors="coerce").mean().fillna(0)
    labels = ["Near Post", "Far Post", "6 Yard", "Penalty"]
    ax.bar(labels, means.values, color=["#00C2FF", "#FFD400", "#00FF6A", "#A78BFA"])
    themed_bar(ax, theme)
    ax.set_title("Attacking Structure Zone Averages", fontsize=16, weight="bold")
    ax.set_ylabel("Average Players")
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
    "xG By Routine": chart_xg_by_routine,
    "Second Ball Map": chart_second_ball_map,
    "Defensive Vulnerability Map": chart_defensive_vulnerability_map,
    "Taker Profile": chart_taker_profile,
    "Structure Zone Averages": chart_structure_zone_averages,
}
