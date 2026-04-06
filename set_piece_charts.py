import io
import os
import tempfile
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mplsoccer import Pitch

from data_utils import bool01
from ui_theme import THEMES, build_chart_style


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


def resolve_style(theme_name: str, style_overrides: Optional[dict] = None) -> dict:
    return build_chart_style(theme_name, style_overrides or {})


def apply_global_rcparams(style: dict):
    mpl.rcParams["font.family"] = style["font_family"]
    mpl.rcParams["axes.titlesize"] = style["title_size"]
    mpl.rcParams["axes.labelsize"] = style["label_size"]
    mpl.rcParams["xtick.labelsize"] = style["tick_size"]
    mpl.rcParams["ytick.labelsize"] = style["tick_size"]
    mpl.rcParams["legend.fontsize"] = style["legend_size"]


def make_pitch(style: dict):
    stripe = True if style.get("pitch_stripe") else False
    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=64,
        line_zorder=2,
        linewidth=style["line_width"],
        pitch_color=style["pitch"],
        line_color=style["pitch_lines"],
        stripe=stripe,
        stripe_color=style.get("pitch_stripe"),
    )


def apply_flip_y(df: pd.DataFrame, flip_y: bool = False) -> pd.DataFrame:
    out = df.copy()
    if not flip_y:
        return out

    for c in ["y", "y2", "y3"]:
        if c in out.columns:
            out[c] = 64 - pd.to_numeric(out[c], errors="coerce")

    return out


def themed_bar(ax, style: dict):
    ax.set_facecolor(style["panel"])
    for spine in ax.spines.values():
        spine.set_color(style["lines"])
        spine.set_linewidth(1.0)

    ax.tick_params(
        colors=style["muted"],
        labelsize=style["tick_size"],
    )

    ax.yaxis.label.set_color(style["muted"])
    ax.xaxis.label.set_color(style["muted"])
    ax.title.set_color(style["text"])

    if style.get("show_grid", True):
        ax.grid(axis="y", alpha=style["grid_alpha"], color=style["lines"], linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)


def style_pitch_axes(ax, style: dict):
    ax.set_facecolor(style["pitch"])
    ax.set_xlim(-style["pitch_pad_x"], 100 + style["pitch_pad_x"])
    ax.set_ylim(-style["pitch_pad_y"], 64 + style["pitch_pad_y"])

    if not style.get("show_ticks", True):
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def set_chart_title(ax, title: str, style: dict):
    if style.get("show_title", True):
        ax.set_title(
            title,
            color=style["text"],
            fontsize=style["title_size"],
            fontweight=style["title_weight"],
            pad=12,
        )


def style_legend(leg, style: dict):
    if leg is None:
        return
    frame = leg.get_frame()
    if frame is not None:
        frame.set_facecolor(style["panel"])
        frame.set_edgecolor(style["lines"])
        frame.set_alpha(0.95)

    for txt in leg.get_texts():
        txt.set_color(style["text"])


def fig_to_png_bytes(fig, dpi: int = 260):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.25)
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


def _base_figure(style: dict, figsize=(8, 6)):
    apply_global_rcparams(style)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(style["bg"])
    ax.set_facecolor(style["panel"])
    return fig, ax


def chart_delivery_heatmap(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = apply_flip_y(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x2", "y2"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(
                dd["x2"], dd["y2"],
                ax=ax,
                fill=True,
                levels=50,
                alpha=style["kde_alpha"],
                cmap="Blues",
            )
        except Exception:
            pitch.scatter(
                dd["x2"], dd["y2"],
                ax=ax,
                s=style["marker_size"] * 0.65,
                color=style["accent"],
                alpha=style["alpha"],
            )

    set_chart_title(ax, "Delivery End Location Heatmap", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_delivery_end_scatter(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = apply_flip_y(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x2", "y2"]).copy()

    if "delivery_type" in dd.columns:
        color_map = {
            "inswing": style["accent"],
            "outswing": style["warning"],
            "straight": style["accent_2"],
            "driven": style["success"],
            "short": style["danger"],
        }
        for dtype, grp in dd.groupby("delivery_type"):
            pitch.scatter(
                grp["x2"],
                grp["y2"],
                ax=ax,
                s=style["marker_size"],
                color=color_map.get(str(dtype).lower(), style["text"]),
                edgecolors=style["pitch_lines"],
                linewidth=style["marker_edge_width"],
                label=str(dtype).title(),
                alpha=style["alpha"],
            )

        if style.get("show_legend", True):
            leg = ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
            style_legend(leg, style)
    else:
        pitch.scatter(
            dd["x2"], dd["y2"],
            ax=ax,
            s=style["marker_size"],
            color=style["accent"],
            edgecolors=style["pitch_lines"],
            linewidth=style["marker_edge_width"],
            alpha=style["alpha"],
        )

    set_chart_title(ax, "Delivery End Scatter", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_outcome_distribution(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.4, 4.6))

    counts = get_set_piece_series(df).astype(str).str.lower().value_counts()
    colors = []
    for x in counts.index:
        if x in ["successful", "corner", "free_kick"]:
            colors.append(style["success"])
        elif x in ["unsuccessful", "failed", "loss"]:
            colors.append(style["danger"])
        else:
            colors.append(style["accent"])

    ax.bar(counts.index, counts.values, color=colors, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Set Piece Type / Outcome Distribution", style)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_target_zone_breakdown(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.4, 4.6))

    counts = get_target_zone_series(df).value_counts()
    ax.bar(counts.index, counts.values, color=style["accent"], edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Target Zone Breakdown", style)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_first_contact_win_by_zone(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6, 4.8))

    dd = df.copy()
    dd["zone_calc"] = get_target_zone_series(dd)
    dd["fc_win_calc"] = get_first_contact_win_series(dd)

    summary = dd.groupby("zone_calc", dropna=False)["fc_win_calc"].mean().sort_values(ascending=False) * 100
    ax.bar(summary.index.astype(str), summary.values, color=style["accent"], edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "First Contact Win % By Zone", style)
    ax.set_ylabel("Win %")
    ax.tick_params(axis="x", rotation=25)

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_routine_breakdown(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6, 4.8))

    if "routine_type" in df.columns and df["routine_type"].notna().any():
        counts = df["routine_type"].fillna("unclassified").value_counts().head(10)
    else:
        counts = get_target_zone_series(df).value_counts().head(10)

    ax.barh(
        counts.index[::-1],
        counts.values[::-1],
        color=style["warning"],
        edgecolor=style["lines"],
        linewidth=0.8,
    )
    themed_bar(ax, style)
    set_chart_title(ax, "Routine Breakdown", style)
    ax.set_xlabel("Count")

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_shot_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = apply_flip_y(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x", "y"]).copy()

    size = style["marker_size"] * 1.6
    if "xg" in dd.columns:
        xg = pd.to_numeric(dd["xg"], errors="coerce").fillna(0)
        size = 35 + xg * 500

    pitch.scatter(
        dd["x"], dd["y"],
        ax=ax,
        s=size,
        color=style["success"],
        edgecolors=style["pitch_lines"],
        linewidth=style["marker_edge_width"],
        alpha=style["alpha"],
    )
    set_chart_title(ax, "Set Piece Shot / Event Map", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_second_ball_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = apply_flip_y(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x", "y"]).copy()
    dd["second_ball_calc"] = get_second_ball_win_series(dd)

    win = dd[dd["second_ball_calc"] == 1]
    lose = dd[dd["second_ball_calc"] == 0]

    if len(win):
        pitch.scatter(
            win["x"], win["y"],
            ax=ax,
            s=style["marker_size"] * 1.25,
            color=style["success"],
            edgecolors=style["pitch_lines"],
            linewidth=style["marker_edge_width"],
            label="Won",
            alpha=style["alpha"],
        )
    if len(lose):
        pitch.scatter(
            lose["x"], lose["y"],
            ax=ax,
            s=style["marker_size"] * 1.25,
            facecolors="none",
            edgecolors=style["danger"],
            linewidth=style["line_width"] + 0.6,
            label="Lost",
            alpha=style["alpha"],
        )

    if (len(win) or len(lose)) and style.get("show_legend", True):
        leg = ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2)
        style_legend(leg, style)

    set_chart_title(ax, "Second Ball Map", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_defensive_vulnerability_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = apply_flip_y(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x", "y"]).copy()
    if len(dd):
        try:
            pitch.kdeplot(
                dd["x"], dd["y"],
                ax=ax,
                fill=True,
                levels=40,
                alpha=style["kde_alpha"],
                cmap="Reds",
            )
        except Exception:
            pitch.scatter(
                dd["x"], dd["y"],
                ax=ax,
                s=style["marker_size"] * 0.8,
                color=style["danger"],
                edgecolors=style["pitch_lines"],
                linewidth=style["marker_edge_width"],
                alpha=style["alpha"],
            )

    set_chart_title(ax, "Defensive Vulnerability Map", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_taker_profile(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.8, 4.8))

    if "taker" in df.columns and "sequence_id" in df.columns:
        seq_counts = df.groupby("taker")["sequence_id"].nunique().sort_values(ascending=False).head(10)
    elif "taker" in df.columns:
        seq_counts = df["taker"].value_counts().head(10)
    else:
        seq_counts = get_set_piece_series(df).value_counts().head(10)

    ax.barh(
        seq_counts.index[::-1],
        seq_counts.values[::-1],
        color=style["accent"],
        edgecolor=style["lines"],
        linewidth=0.8,
    )
    themed_bar(ax, style)
    set_chart_title(ax, "Taker / Event Profile", style)
    ax.set_xlabel("Count")

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_structure_zone_averages(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6, 4.8))

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

    color_cycle = [style["accent"], style["warning"], style["success"], style["accent_2"]]
    colors = color_cycle[:len(values)]

    ax.bar(labels, values, color=colors, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Structure / Zone Summary", style)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=15)

    if style["tight_layout"]:
        fig.tight_layout()
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
