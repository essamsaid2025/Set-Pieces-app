# set_piece_charts.py
import io
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch

from data_utils import bool01
from ui_theme import build_chart_style

try:
    from mplsoccer import Pitch as MplsoccerPitch, VerticalPitch
except Exception:
    MplsoccerPitch = None
    VerticalPitch = None

# =========================================================
# PITCH DIMENSIONS
# =========================================================
PL, PW = 100.0, 64.0
BOX_X0, BOX_X1 = 83.5, 100.0
BOX_Y0, BOX_Y1 = 13.84, 50.16
SIX_X0 = 94.5
SIX_Y0, SIX_Y1 = 24.84, 39.16
GOAL_Y0, GOAL_Y1 = 28.34, 35.66
BOX_W = BOX_X1 - BOX_X0
BOX_H = BOX_Y1 - BOX_Y0

# =========================================================
# ZONES — aligned with six-yard and penalty box
# =========================================================
def _barca_zones(corner_side: str):
    if corner_side == "right":
        return [
            ("Near Post Short", BOX_X0, BOX_Y1, BOX_W, PW - BOX_Y1),
            ("Near Post", SIX_X0, SIX_Y1, BOX_X1 - SIX_X0, BOX_Y1 - SIX_Y1),
            ("Small Area", SIX_X0, GOAL_Y1, BOX_X1 - SIX_X0, BOX_Y1 - GOAL_Y1),
            ("Penalty Spot", BOX_X0, SIX_Y0, SIX_X0 - BOX_X0, SIX_Y1 - SIX_Y0),
            ("Small Area", SIX_X0, SIX_Y0, BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
            ("Far Post", BOX_X0, BOX_Y0, SIX_X0 - BOX_X0, SIX_Y0 - BOX_Y0),
            ("Small Area", SIX_X0, BOX_Y0, BOX_X1 - SIX_X0, GOAL_Y0 - BOX_Y0),
            ("Far Post Long", BOX_X0, 0.0, BOX_W, BOX_Y0),
            ("Box Front", 72.0, BOX_Y0, BOX_X0 - 72.0, BOX_H),
        ]
    else:
        return [
            ("Near Post Short", BOX_X0, 0.0, BOX_W, BOX_Y0),
            ("Near Post", SIX_X0, BOX_Y0, BOX_X1 - SIX_X0, GOAL_Y0 - BOX_Y0),
            ("Small Area", SIX_X0, BOX_Y0, BOX_X1 - SIX_X0, GOAL_Y0 - BOX_Y0),
            ("Penalty Spot", BOX_X0, SIX_Y0, SIX_X0 - BOX_X0, SIX_Y1 - SIX_Y0),
            ("Small Area", SIX_X0, SIX_Y0, BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
            ("Far Post", BOX_X0, SIX_Y1, SIX_X0 - BOX_X0, BOX_Y1 - SIX_Y1),
            ("Small Area", SIX_X0, GOAL_Y1, BOX_X1 - SIX_X0, BOX_Y1 - GOAL_Y1),
            ("Far Post Long", BOX_X0, BOX_Y1, BOX_W, PW - BOX_Y1),
            ("Box Front", 72.0, BOX_Y0, BOX_X0 - 72.0, BOX_H),
        ]

# =========================================================
# STYLE UTILS
# =========================================================
def resolve_style(tn, ov=None):
    return build_chart_style(tn, ov or {})


def apply_rcparams(s):
    mpl.rcParams["font.family"] = s["font_family"]
    mpl.rcParams["axes.titlesize"] = s["title_size"]


def make_pitch(s, vertical=False):
    if MplsoccerPitch:
        try:
            cls = VerticalPitch if (vertical and VerticalPitch) else MplsoccerPitch
            return cls(
                pitch_type="custom",
                pitch_length=100,
                pitch_width=64,
                linewidth=s["line_width"],
                pitch_color=s["pitch"],
                line_color=s["pitch_lines"],
            )
        except Exception:
            pass
    return None


def _base_fig(s, figsize=(8, 6)):
    apply_rcparams(s)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["panel"])
    return fig, ax


def chart_title(ax, t, s):
    ax.set_title(t, color=s["text"], fontsize=s["title_size"], fontweight="bold", pad=12)


# =========================================================
# PREP DATA
# =========================================================
def _prep(df):
    out = df.copy()
    for c in ["x", "y", "x2", "y2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================================================
# TRAJECTORY CHARTS
# =========================================================
def _traj_chart(df, theme_name, style_overrides, title, corner_side):
    s = resolve_style(theme_name, style_overrides or {})
    pitch = make_pitch(s, False)
    dff = _prep(df)
    if "side" in dff.columns:
        dff = dff[dff["side"].str.lower() == corner_side]
    fig, ax = _base_fig(s, (8, 6))
    if pitch:
        pitch.draw(ax=ax)
    zones = _barca_zones(corner_side)
    for (_, zx, zy, zw, zh) in zones:
        ax.add_patch(Rectangle((zx, zy), zw, zh, fill=False, edgecolor="gray", alpha=0.2))
    dd = dff.dropna(subset=["x", "y", "x2", "y2"])
    for _, r in dd.iterrows():
        ax.add_patch(
            FancyArrowPatch(
                (r["x"], r["y"]),
                (r["x2"], r["y2"]),
                arrowstyle="-|>",
                color=s["accent"],
                alpha=0.7,
            )
        )
    chart_title(ax, title, s)
    fig.tight_layout()
    return fig


def chart_delivery_trajectories_left(df, theme_name=None, flip_y=False, style_overrides=None):
    return _traj_chart(
        df,
        theme_name or "The Athletic Dark",
        style_overrides or {},
        "Delivery Trajectories — Left Corner",
        "left",
    )


def chart_delivery_trajectories_right(df, theme_name=None, flip_y=False, style_overrides=None):
    return _traj_chart(
        df,
        theme_name or "The Athletic Dark",
        style_overrides or {},
        "Delivery Trajectories — Right Corner",
        "right",
    )


# =========================================================
# DELIVERY END SCATTER (dots only)
# =========================================================
def _scatter_chart(df, theme_name, style_overrides, corner_side):
    s = resolve_style(theme_name, style_overrides or {})
    pitch = make_pitch(s, False)
    dff = _prep(df)
    if "side" in dff.columns:
        dff = dff[dff["side"].str.lower() == corner_side]
    fig, ax = _base_fig(s, (10, 6))
    if pitch:
        pitch.draw(ax=ax)
    dd = dff.dropna(subset=["x2", "y2"])
    ax.scatter(
        dd["x2"],
        dd["y2"],
        s=s["marker_size"] * 0.5,
        color=s["scatter_dot_color"],
        edgecolors=s["pitch_lines"],
        alpha=s["alpha"],
    )
    chart_title(ax, f"Delivery End Scatter — {corner_side.title()} Corner", s)
    fig.tight_layout()
    return fig


def chart_delivery_end_scatter_left(df, theme_name=None, flip_y=False, style_overrides=None):
    return _scatter_chart(df, theme_name or "The Athletic Dark", style_overrides or {}, "left")


def chart_delivery_end_scatter_right(df, theme_name=None, flip_y=False, style_overrides=None):
    return _scatter_chart(df, theme_name or "The Athletic Dark", style_overrides or {}, "right")


# =========================================================
# ZONE PLAYER STATS
# =========================================================
def compute_zone_player_stats(df, corner_side="right"):
    """
    Returns a DataFrame with columns:
      - zone: zone label
      - count: integer count of players (deliveries' end points) in that zone
      - pct: proportion of total deliveries that fall in that zone
      - col: canonical column name mapping for downstream use
    """
    dff = _prep(df)
    dd = dff.dropna(subset=["x2", "y2"])
    zones = _barca_zones(corner_side)
    total = len(dd)
    rows = []
    label_to_col = {
        "Near Post Short": "players_near_post_short",
        "Near Post": "players_near_post",
        "Small Area": "players_6yard",
        "Penalty Spot": "players_penalty",
        "Far Post": "players_far_post",
        "Far Post Long": "players_far_post_long",
        "Box Front": "players_box",
    }
    for label, zx, zy, zw, zh in zones:
        mask = (dd["x2"] >= zx) & (dd["x2"] < zx + zw) & (dd["y2"] >= zy) & (dd["y2"] < zy + zh)
        count = int(mask.sum())
        pct = float(count / total) if total else 0.0
        rows.append({"zone": label, "count": count, "pct": pct, "col": label_to_col.get(label, "other")})
    stats = pd.DataFrame(rows)
    # Keep original zone order
    stats["zone"] = pd.Categorical(stats["zone"], categories=[z[0] for z in zones], ordered=True)
    stats = stats.sort_values("zone")
    stats = stats.reset_index(drop=True)
    return stats


def chart_zone_player_stats(df, corner_side="right", theme_name=None, style_overrides=None):
    """
    Bar chart showing counts per zone. Returns a matplotlib Figure.
    """
    stats = compute_zone_player_stats(df, corner_side)
    s = resolve_style(theme_name or "The Athletic Dark", style_overrides or {})
    fig, ax = _base_fig(s, (10, 6))
    bars = ax.bar(stats["zone"].astype(str), stats["count"], color=s["accent"], edgecolor=s["pitch_lines"], alpha=0.9)
    ax.set_ylabel("Count", color=s["text"])
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelcolor=s["text"])
    ax.tick_params(axis="y", labelcolor=s["text"])
    # Annotate counts above bars
    for bar, cnt in zip(bars, stats["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(1, h * 0.02), str(int(cnt)), ha="center", va="bottom", color=s["text"])
    chart_title(ax, f"Zone Player Stats — {corner_side.title()} Corner", s)
    fig.tight_layout()
    return fig


# =========================================================
# CHART BUILDERS REGISTRY
# =========================================================
CHART_BUILDERS = {
    "delivery_left": chart_delivery_trajectories_left,
    "delivery_right": chart_delivery_trajectories_right,
    "scatter_left": chart_delivery_end_scatter_left,
    "scatter_right": chart_delivery_end_scatter_right,
    "zone_stats_left": lambda df, **kwargs: chart_zone_player_stats(df, "left", **kwargs),
    "zone_stats_right": lambda df, **kwargs: chart_zone_player_stats(df, "right", **kwargs),
}
