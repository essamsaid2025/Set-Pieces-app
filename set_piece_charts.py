import io
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.font_manager import FontProperties

from data_utils import bool01
from ui_theme import build_chart_style

try:
    from mplsoccer import Pitch as MplsoccerPitch, VerticalPitch
except Exception:
    MplsoccerPitch = None
    VerticalPitch = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# =========================================================
# PITCH DIMENSIONS (100×64 space)
# =========================================================
PITCH_LENGTH   = 100.0
PITCH_WIDTH    = 64.0
BOX_X_START    = 83.5       # penalty box starts here
BOX_X_END      = 100.0
BOX_Y_START    = 13.84      # penalty box top edge
BOX_Y_END      = 50.16      # penalty box bottom edge
SIX_YD_X_START = 94.5
SIX_YD_Y_START = 24.84
SIX_YD_Y_END   = 39.16
GOAL_Y_START   = 28.34
GOAL_Y_END     = 35.66
PENALTY_X      = 88.5


# =========================================================
# BARCELONA ZONES  (exact geometry)
# =========================================================
# Each zone: (label, x0, y0, width, height)
BARCA_ZONES = [
    ("Near Post\nShort",   BOX_X_START, 0.0,           BOX_X_END - BOX_X_START, BOX_Y_START),
    ("Near\nPost",         BOX_X_START, BOX_Y_START,   SIX_YD_X_START - BOX_X_START, SIX_YD_Y_START - BOX_Y_START),
    ("Small\nArea",        SIX_YD_X_START, BOX_Y_START, BOX_X_END - SIX_YD_X_START,  GOAL_Y_START - BOX_Y_START),
    ("Penalty\nSpot",      BOX_X_START, SIX_YD_Y_START, SIX_YD_X_START - BOX_X_START, SIX_YD_Y_END - SIX_YD_Y_START),
    ("Small\nArea",        SIX_YD_X_START, SIX_YD_Y_START, BOX_X_END - SIX_YD_X_START, SIX_YD_Y_END - SIX_YD_Y_START),
    ("Far\nPost",          BOX_X_START, SIX_YD_Y_END,  SIX_YD_X_START - BOX_X_START, BOX_Y_END - SIX_YD_Y_END),
    ("Small\nArea",        SIX_YD_X_START, GOAL_Y_END,  BOX_X_END - SIX_YD_X_START,  BOX_Y_END - GOAL_Y_END),
    ("Far Post\nLong",     BOX_X_START, BOX_Y_END,     BOX_X_END - BOX_X_START, PITCH_WIDTH - BOX_Y_END),
    ("Box\nFront",         72.0, BOX_Y_START,           BOX_X_START - 72.0, BOX_Y_END - BOX_Y_START),
]

ZONE_COLOUR_KEYS = [
    "accent",    # Near Post Short
    "accent_2",  # Near Post
    "warning",   # Small Area (top)
    "success",   # Penalty Spot
    "warning",   # Small Area (mid)
    "accent_2",  # Far Post
    "warning",   # Small Area (bot)
    "accent",    # Far Post Long
    "muted",     # Box Front
]


# =========================================================
# SAFE PITCH FALLBACK
# =========================================================
class SimplePitch:
    def __init__(self, pitch_length=100, pitch_width=64, pitch_color="#1F5F3B",
                 line_color="#FFFFFF", linewidth=1.4, stripe=False, stripe_color=None,
                 line_zorder=2, vertical=False):
        self.pitch_length = pitch_length
        self.pitch_width  = pitch_width
        self.pitch_color  = pitch_color
        self.line_color   = line_color
        self.linewidth    = linewidth
        self.stripe       = stripe
        self.stripe_color = stripe_color
        self.line_zorder  = line_zorder
        self.vertical     = vertical

    def draw(self, ax):
        ax.set_facecolor(self.pitch_color)
        W = self.pitch_width  if self.vertical else self.pitch_length
        H = self.pitch_length if self.vertical else self.pitch_width

        if self.stripe and self.stripe_color:
            sw = W / 10.0
            for i in range(10):
                if i % 2 == 0:
                    ax.add_patch(Rectangle((i*sw,0), sw, H, facecolor=self.stripe_color,
                                           edgecolor="none", alpha=0.35, zorder=0))
        ax.add_patch(Rectangle((0,0), W, H, fill=False, edgecolor=self.line_color,
                                linewidth=self.linewidth, zorder=self.line_zorder))

        def _hline(y):
            ax.plot([0,W],[y,y], color=self.line_color, lw=self.linewidth, zorder=self.line_zorder)
        def _vline(x):
            ax.plot([x,x],[0,H], color=self.line_color, lw=self.linewidth, zorder=self.line_zorder)
        def _rect(x,y,w,h):
            ax.add_patch(Rectangle((x,y),w,h, fill=False, edgecolor=self.line_color,
                                    linewidth=self.linewidth, zorder=self.line_zorder))
        def _circle(cx,cy,r):
            ax.add_patch(plt.Circle((cx,cy),r, fill=False, color=self.line_color,
                                     lw=self.linewidth, zorder=self.line_zorder))

        if self.vertical:
            PL, PW = self.pitch_length, self.pitch_width
            _hline(PL/2)
            _circle(PW/2, PL/2, 9.15)
            _rect((PW-40.32)/2, 0, 40.32, 16.5)
            _rect((PW-18.32)/2, 0, 18.32, 5.5)
            _rect((PW-40.32)/2, PL-16.5, 40.32, 16.5)
            _rect((PW-18.32)/2, PL-5.5,  18.32, 5.5)
            ax.scatter([PW/2,PW/2],[11,PL-11], c=self.line_color, s=10, zorder=self.line_zorder)
            ax.add_patch(Arc((PW/2,11),    18.3,18.3, angle=0, theta1=40,  theta2=140,  color=self.line_color, lw=self.linewidth, zorder=self.line_zorder))
            ax.add_patch(Arc((PW/2,PL-11), 18.3,18.3, angle=0, theta1=220, theta2=320, color=self.line_color, lw=self.linewidth, zorder=self.line_zorder))
            gd,gw = 1.5,7.32
            _rect((PW-gw)/2,-gd,gw,gd)
            _rect((PW-gw)/2,PL, gw,gd)
            ax.set_xlim(0,PW); ax.set_ylim(0,PL)
        else:
            PL, PW = self.pitch_length, self.pitch_width
            _vline(PL/2)
            _circle(PL/2, PW/2, 9.15)
            _rect(0,          (PW-40.32)/2, 16.5, 40.32)
            _rect(0,          (PW-18.32)/2,  5.5, 18.32)
            _rect(PL-16.5,    (PW-40.32)/2, 16.5, 40.32)
            _rect(PL-5.5,     (PW-18.32)/2,  5.5, 18.32)
            ax.scatter([11,PL-11],[PW/2,PW/2], c=self.line_color, s=10, zorder=self.line_zorder)
            ax.add_patch(Arc((11,     PW/2), 18.3,18.3, angle=0, theta1=310, theta2=50,  color=self.line_color, lw=self.linewidth, zorder=self.line_zorder))
            ax.add_patch(Arc((PL-11,  PW/2), 18.3,18.3, angle=0, theta1=130, theta2=230, color=self.line_color, lw=self.linewidth, zorder=self.line_zorder))
            gd,gw = 1.5,7.32
            _rect(-gd, PW/2-gw/2, gd, gw)
            _rect(PL,  PW/2-gw/2, gd, gw)
            ax.set_xlim(0,PL); ax.set_ylim(0,PW)
        return ax

    def scatter(self, x, y, ax, **kwargs): return ax.scatter(x, y, **kwargs)

    def kdeplot(self, x, y, ax, fill=True, levels=40, alpha=0.72, cmap="Blues"):
        if self.vertical:
            return ax.hist2d(y, x, bins=[22,14], range=[[0,self.pitch_length],[0,self.pitch_width]], cmap=cmap, alpha=alpha)
        return ax.hist2d(x, y, bins=[22,14], range=[[0,self.pitch_length],[0,self.pitch_width]], cmap=cmap, alpha=alpha)


# =========================================================
# CHART REQUIREMENTS
# =========================================================
CHART_REQUIREMENTS: Dict[str, List[str]] = {
    "Delivery Start Map":                  ["x", "y"],
    "Delivery Heatmap":                    ["x2", "y2"],
    "Delivery End Scatter":                ["x2", "y2"],
    "Delivery Trajectories":               ["x", "y", "x2", "y2"],
    "Delivery Trajectories - Left Corners":["x", "y", "x2", "y2"],
    "Delivery Trajectories - Right Corners":["x","y","x2","y2"],
    "Average Delivery Path":               ["x", "y", "x2", "y2"],
    "Heat + Trajectories":                 ["x", "y", "x2", "y2"],
    "Trajectory Clusters":                 ["x", "y", "x2", "y2"],
    "Delivery Length Distribution":        ["x", "y", "x2", "y2"],
    "Delivery Direction Map":              ["x", "y", "x2", "y2"],
    "Outcome Distribution":                ["set_piece_type"],
    "Target Zone Breakdown":               ["x2", "y2"],
    "First Contact Win By Zone":           ["x2", "y2"],
    "Routine Breakdown":                   ["set_piece_type"],
    "Shot Map":                            ["x", "y"],
    "Second Ball Map":                     ["x", "y"],
    "Defensive Vulnerability Map":         ["x", "y"],
    "Taker Profile":                       ["set_piece_type"],
    "Structure Zone Averages":             [],
    "Set Piece Landing Heatmap":           ["x2", "y2"],
    "Taker Stats Table":                   ["taker"],
}


# =========================================================
# STYLE HELPERS
# =========================================================
def resolve_style(theme_name, style_overrides=None):
    return build_chart_style(theme_name, style_overrides or {})

def apply_global_rcparams(style):
    mpl.rcParams["font.family"]      = style["font_family"]
    mpl.rcParams["axes.titlesize"]   = style["title_size"]
    mpl.rcParams["axes.labelsize"]   = style["label_size"]
    mpl.rcParams["xtick.labelsize"]  = style["tick_size"]
    mpl.rcParams["ytick.labelsize"]  = style["tick_size"]
    mpl.rcParams["legend.fontsize"]  = style["legend_size"]

def make_pitch(style, vertical=False):
    stripe = bool(style.get("pitch_stripe"))
    if MplsoccerPitch is not None:
        try:
            cls = VerticalPitch if (vertical and VerticalPitch) else MplsoccerPitch
            return cls(pitch_type="custom", pitch_length=100, pitch_width=64,
                       line_zorder=2, linewidth=style["line_width"],
                       pitch_color=style["pitch"], line_color=style["pitch_lines"],
                       stripe=stripe, stripe_color=style.get("pitch_stripe"))
        except Exception:
            pass
    return SimplePitch(pitch_length=100, pitch_width=64, pitch_color=style["pitch"],
                       line_color=style["pitch_lines"], linewidth=style["line_width"],
                       stripe=stripe, stripe_color=style.get("pitch_stripe"),
                       line_zorder=2, vertical=vertical)

def apply_flip_y(df, flip_y=False):
    out = df.copy()
    if not flip_y: return out
    for c in ["y","y2","y3"]:
        if c in out.columns:
            out[c] = 64 - pd.to_numeric(out[c], errors="coerce")
    return out

def themed_bar(ax, style):
    ax.set_facecolor(style["panel"])
    for spine in ax.spines.values():
        spine.set_color(style["lines"]); spine.set_linewidth(1.0)
    ax.tick_params(colors=style["muted"], labelsize=style["tick_size"])
    ax.yaxis.label.set_color(style["muted"])
    ax.xaxis.label.set_color(style["muted"])
    ax.title.set_color(style["text"])
    if style.get("show_grid", True):
        ax.grid(axis="y", alpha=style["grid_alpha"], color=style["lines"], linestyle="--", lw=0.8)
        ax.set_axisbelow(True)

def style_pitch_axes(ax, style, vertical=False):
    ax.set_facecolor(style["pitch"])
    px, py = style["pitch_pad_x"], style["pitch_pad_y"]
    if vertical:
        ax.set_xlim(-py, 64+py); ax.set_ylim(-px, 100+px)
    else:
        ax.set_xlim(-px, 100+px); ax.set_ylim(-py, 64+py)
    if not style.get("show_ticks", False):
        ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

def set_chart_title(ax, title, style):
    if style.get("show_title", True):
        ax.set_title(title, color=style["text"], fontsize=style["title_size"],
                     fontweight=style["title_weight"], pad=12)

def style_legend(leg, style):
    if leg is None: return
    frame = leg.get_frame()
    if frame:
        frame.set_facecolor(style["panel"]); frame.set_edgecolor(style["lines"]); frame.set_alpha(0.95)
    for txt in leg.get_texts(): txt.set_color(style["text"])

def fig_to_png_bytes(fig, dpi=260):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    buf.seek(0)
    return buf.getvalue()

def save_report_pdf(figures, filename="set_piece_report.pdf"):
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, filename)
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
    with open(pdf_path, "rb") as f: return f.read()

def _base_figure(style, figsize=(8,6)):
    apply_global_rcparams(style)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(style["bg"])
    ax.set_facecolor(style["panel"])
    return fig, ax


# =========================================================
# COORDINATE SCALING  (global-max approach to fix y/y2 mismatch)
# =========================================================
def _clean_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _auto_scale_coordinates(df):
    out = df.copy()
    out = _clean_numeric(out, ["x","y","x2","y2","x3","y3"])

    # scale y using the GLOBAL max across all y-columns together
    all_y = pd.concat([out[c].dropna() for c in ["y","y2","y3"] if c in out.columns], ignore_index=True)
    if len(all_y):
        gy_max = all_y.max()
        if pd.notna(gy_max) and gy_max > 64.5:
            ys = 64.0 / gy_max
            for c in ["y","y2","y3"]:
                if c in out.columns: out[c] = out[c] * ys

    # scale x using the GLOBAL max across all x-columns together
    all_x = pd.concat([out[c].dropna() for c in ["x","x2","x3"] if c in out.columns], ignore_index=True)
    if len(all_x):
        gx_max = all_x.max()
        if pd.notna(gx_max) and gx_max > 100.5:
            xs = 100.0 / gx_max
            for c in ["x","x2","x3"]:
                if c in out.columns: out[c] = out[c] * xs

    for c in ["x","x2","x3"]:
        if c in out.columns: out[c] = out[c].clip(0, 100)
    for c in ["y","y2","y3"]:
        if c in out.columns: out[c] = out[c].clip(0, 64)
    return out


# =========================================================
# ZONE HELPERS
# =========================================================
def infer_zone_from_xy(x, y):
    try: x,y = float(x), float(y)
    except: return "unknown"
    if y < BOX_Y_START:    return "near_post_short"
    if y > BOX_Y_END:      return "far_post_long"
    if x < BOX_X_START:    return "box_front"
    if x >= SIX_YD_X_START:
        if y < GOAL_Y_START: return "small_area"
        if y > GOAL_Y_END:   return "small_area"
        return "small_area"
    if y < SIX_YD_Y_START: return "near_post"
    if y > SIX_YD_Y_END:   return "far_post"
    return "penalty_spot"

def get_set_piece_series(df):
    for c in ["set_piece_type","event","outcome"]:
        if c in df.columns: return df[c].fillna("unknown").astype(str)
    return pd.Series(["unknown"]*len(df), index=df.index)

def get_target_zone_series(df):
    if "target_zone" in df.columns: return df["target_zone"].fillna("unknown").astype(str)
    if "x2" in df.columns and "y2" in df.columns:
        return df.apply(lambda r: infer_zone_from_xy(r.get("x2"), r.get("y2")), axis=1)
    return pd.Series(["unknown"]*len(df), index=df.index)

def get_first_contact_win_series(df):
    if "first_contact_win" in df.columns: return bool01(df["first_contact_win"])
    spt = get_set_piece_series(df).str.lower()
    return spt.isin(["successful","win","won","first_contact"]).astype(int)

def get_second_ball_win_series(df):
    if "second_ball_win" in df.columns: return bool01(df["second_ball_win"])
    spt = get_set_piece_series(df).str.lower()
    return spt.isin(["second_ball_win","second ball win","won_second_ball"]).astype(int)


# =========================================================
# DELIVERY HELPERS
# =========================================================
def _get_delivery_color_map(style):
    cm = style.get("arrow_colors", {})
    return {
        "inswing":  cm.get("inswing",  style["accent"]),
        "outswing": cm.get("outswing", style["warning"]),
        "straight": cm.get("straight", style["accent_2"]),
        "driven":   cm.get("driven",   style["success"]),
        "short":    cm.get("short",    style["danger"]),
    }

def _corner_anchor(x, y):
    if pd.isna(x) or pd.isna(y): return x, y, "unknown"
    side = "right" if x >= 50 else "left"
    half = "top"   if y <= 32  else "bottom"
    cx = 99.5 if side == "right" else 0.5
    cy = 0.5  if half == "top"   else 63.5
    return cx, cy, f"{side}_{half}"

def _curve_rad_for_delivery(dtype, corner_label):
    dtype = str(dtype).lower()
    if corner_label == "right_top":
        if dtype == "inswing":   return  0.30
        if dtype == "outswing":  return -0.30
    elif corner_label == "right_bottom":
        if dtype == "inswing":   return -0.30
        if dtype == "outswing":  return  0.30
    elif corner_label == "left_top":
        if dtype == "inswing":   return -0.30
        if dtype == "outswing":  return  0.30
    elif corner_label == "left_bottom":
        if dtype == "inswing":   return  0.30
        if dtype == "outswing":  return -0.30
    return 0.0

def _prepare_delivery_df(df, flip_y=False):
    dff = apply_flip_y(df, flip_y)
    dff = _auto_scale_coordinates(dff)
    if "x" in dff.columns and "y" in dff.columns:
        cd = dff.apply(lambda r: _corner_anchor(r.get("x"), r.get("y")), axis=1, result_type="expand")
        dff["x_start_plot"] = cd[0]; dff["y_start_plot"] = cd[1]; dff["corner_label"] = cd[2]
        dff["corner_side"]  = dff["corner_label"].astype(str).str.split("_").str[0]
    else:
        dff["x_start_plot"] = dff.get("x"); dff["y_start_plot"] = dff.get("y")
        dff["corner_label"] = "unknown";    dff["corner_side"]  = "unknown"
    return dff

def _draw_curved_arrow(ax, x1, y1, x2, y2, color, style, rad=0.0, lw_mult=1.0, alpha_mult=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1,y1),(x2,y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=style["trajectory_headwidth"]*4.0,
        linewidth=style["trajectory_width"]*lw_mult,
        color=color, alpha=style["trajectory_alpha"]*alpha_mult,
        clip_on=True, zorder=6,
    ))

def _safe_cluster_labels(dd, n_clusters=3):
    features = dd[["x_start_plot","y_start_plot","x2","y2"]].dropna().copy()
    if len(features) < max(n_clusters,3): return pd.Series([0]*len(dd), index=dd.index)
    if KMeans is not None:
        try:
            n  = min(n_clusters, max(2, len(features)))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            lb = km.fit_predict(features)
            return pd.Series(index=features.index, data=lb).reindex(dd.index).fillna(0).astype(int)
        except: pass
    bx = pd.qcut(features["x2"], q=min(3,len(features)), duplicates="drop", labels=False)
    by = pd.qcut(features["y2"], q=min(3,len(features)), duplicates="drop", labels=False)
    return (bx.fillna(0).astype(int)*10+by.fillna(0).astype(int)).reindex(dd.index).fillna(0).astype(int)


# =========================================================
# BARCELONA ZONE OVERLAY  (exact pitch-geometry version)
# =========================================================
def _draw_box_zone_overlay(ax, style, alpha=0.18, vertical=False):
    """
    Draw Barcelona-style zones that match Figure 8 exactly.
    Zones are defined in 100×64 space; vertical mode swaps x↔y.
    """
    for (label, zx, zy, zw, zh), ckey in zip(BARCA_ZONES, ZONE_COLOUR_KEYS):
        color = style.get(ckey, style["accent"])
        if vertical:
            rx, ry, rw, rh = zy, zx, zh, zw
        else:
            rx, ry, rw, rh = zx, zy, zw, zh

        ax.add_patch(Rectangle((rx,ry), rw, rh,
                                facecolor=color, edgecolor=style["pitch_lines"],
                                linewidth=0.7, alpha=alpha, zorder=1))
        fs = max(style["tick_size"]-2, 6)
        ax.text(rx+rw/2, ry+rh/2, label,
                ha="center", va="center", fontsize=fs,
                color=style["text"], alpha=0.92, zorder=2, linespacing=1.2)


# =========================================================
# THIRDS OVERLAY
# =========================================================
def _draw_thirds_lines(ax, style, vertical=False):
    lc    = style.get("pitch_lines","#FFFFFF")
    alpha = 0.55
    lw    = max(style["line_width"]*0.75, 0.9)
    fs    = max(style["tick_size"]-1, 7)
    for pos in [100/3, 200/3]:
        if vertical: ax.axhline(pos, color=lc, lw=lw, alpha=alpha, linestyle="--", zorder=3)
        else:        ax.axvline(pos, color=lc, lw=lw, alpha=alpha, linestyle="--", zorder=3)
    for pos, label in [(100/6,"Def Third"),(100/2,"Mid Third"),(5*100/6,"Att Third")]:
        if vertical:
            ax.text(64+style["pitch_pad_y"]*0.3, pos, label, ha="left",  va="center",
                    fontsize=fs, color=lc, alpha=0.70, zorder=4)
        else:
            ax.text(pos, 64+style["pitch_pad_y"]*0.3, label, ha="center", va="bottom",
                    fontsize=fs, color=lc, alpha=0.70, zorder=4)


# =========================================================
# SHIRT DRAWING HELPER
# =========================================================
def _draw_shirt(ax, cx, cy, size, body_color, sleeve_color, number, number_color,
                text_color, font_size=7, number_font_size=9):
    """Draw a football shirt with a number inside it."""
    s = size
    # Body (trapezoid approximation using rectangle + patches)
    body_w  = s * 0.55
    body_h  = s * 0.65
    sleeve_w = s * 0.22
    sleeve_h = s * 0.28
    collar_r = s * 0.09

    bx = cx - body_w/2
    by = cy - body_h/2 - s*0.04

    # Main body
    ax.add_patch(FancyBboxPatch((bx, by), body_w, body_h,
                                boxstyle="round,pad=0.01",
                                facecolor=body_color, edgecolor=text_color,
                                linewidth=0.6, zorder=4))
    # Left sleeve
    ax.add_patch(FancyBboxPatch((bx - sleeve_w, by + body_h*0.62), sleeve_w, sleeve_h*0.7,
                                boxstyle="round,pad=0.01",
                                facecolor=sleeve_color, edgecolor=text_color,
                                linewidth=0.5, zorder=4))
    # Right sleeve
    ax.add_patch(FancyBboxPatch((bx + body_w, by + body_h*0.62), sleeve_w, sleeve_h*0.7,
                                boxstyle="round,pad=0.01",
                                facecolor=sleeve_color, edgecolor=text_color,
                                linewidth=0.5, zorder=4))
    # Collar
    ax.add_patch(plt.Circle((cx, by+body_h-collar_r*0.3), collar_r,
                             facecolor=sleeve_color, edgecolor=text_color,
                             linewidth=0.5, zorder=5))
    # Number on shirt
    ax.text(cx, by + body_h*0.38, str(number),
            ha="center", va="center", fontsize=number_font_size,
            fontweight="bold", color=number_color, zorder=6)


# =========================================================
# TRAJECTORY CORE
# =========================================================
def _plot_delivery_trajectories_core(df, theme_name, flip_y=False, style_overrides=None,
                                      title="Delivery Trajectories", corner_filter=None, zone_overlay=True):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)
    if corner_filter in ["left","right"]:
        dff = dff[dff["corner_side"]==corner_filter].copy()

    figsize = (6.4,8.4) if vertical else (8.4,6.4)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    if zone_overlay: _draw_box_zone_overlay(ax, style, vertical=vertical)

    dd        = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    color_map = _get_delivery_color_map(style)

    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dtype, grp in dd.groupby("delivery_type"):
            dl = str(dtype).lower()
            for _, r in grp.iterrows():
                rad = _curve_rad_for_delivery(dl, str(r.get("corner_label","unknown")))
                x1 = r["y_start_plot"] if vertical else r["x_start_plot"]
                y1 = r["x_start_plot"] if vertical else r["y_start_plot"]
                x2 = r["y2"] if vertical else r["x2"]
                y2 = r["x2"] if vertical else r["y2"]
                _draw_curved_arrow(ax, x1,y1,x2,y2, color_map.get(dl, style["accent"]), style, rad=rad)
        if style.get("show_legend",True):
            handles,labels = [],[]
            for k,v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    handles.append(Line2D([0],[0],color=v,lw=style["trajectory_width"]+1)); labels.append(k.title())
            if handles:
                leg = ax.legend(handles,labels,frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
                style_legend(leg, style)
    else:
        for _, r in dd.iterrows():
            x1 = r["y_start_plot"] if vertical else r["x_start_plot"]
            y1 = r["x_start_plot"] if vertical else r["y_start_plot"]
            x2 = r["y2"] if vertical else r["x2"]
            y2 = r["x2"] if vertical else r["y2"]
            _draw_curved_arrow(ax, x1,y1,x2,y2, style["accent"], style, rad=0.0)

    set_chart_title(ax, title, style)
    if style["tight_layout"]: fig.tight_layout()
    return fig


# =========================================================
# PITCH CHARTS
# =========================================================
def _pitch_chart(df, theme_name, flip_y, style_overrides, title,
                 x_col, y_col, color_by_col=None, scatter_size_col=None,
                 show_kde=False, show_scatter=True, figsize_h=(8,6), figsize_v=(6,8),
                 zone_overlay=True):
    """Generic pitch chart builder used by scatter, heatmap, shot map etc."""
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = figsize_v if vertical else figsize_h
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    if zone_overlay: _draw_box_zone_overlay(ax, style, vertical=vertical)

    dd = dff.dropna(subset=[x_col, y_col]).copy()
    if not len(dd):
        set_chart_title(ax, title, style); fig.tight_layout(); return fig

    def _px(col): return dd["y"+col[1:]] if (vertical and col.startswith("x")) else dd[col]
    def _py(col): return dd["x"+col[1:]] if (vertical and col.startswith("y")) else dd[col]

    # Wait: for vertical, swap x and y data
    if vertical:
        px = dd[y_col]; py = dd[x_col]
    else:
        px = dd[x_col]; py = dd[y_col]

    if show_kde:
        try:
            pitch.kdeplot(px, py, ax=ax, fill=True, levels=50,
                          alpha=style["kde_alpha"], cmap=style["heatmap_cmap"])
        except Exception:
            pass

    if show_scatter:
        color_map = _get_delivery_color_map(style)
        if color_by_col and color_by_col in dd.columns:
            for dtype, grp in dd.groupby(color_by_col):
                gx = grp[y_col] if vertical else grp[x_col]
                gy = grp[x_col] if vertical else grp[y_col]
                pitch.scatter(gx, gy, ax=ax, s=style["marker_size"],
                              color=color_map.get(str(dtype).lower(), style["accent"]),
                              edgecolors=style["pitch_lines"],
                              linewidth=style["marker_edge_width"],
                              label=str(dtype).title(), alpha=style["alpha"])
            if style.get("show_legend",True):
                leg = ax.legend(frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
                style_legend(leg, style)
        else:
            sz = style["marker_size"]
            if scatter_size_col and scatter_size_col in dd.columns:
                xg = pd.to_numeric(dd[scatter_size_col], errors="coerce").fillna(0)
                sz = 35 + xg*500
            pitch.scatter(px, py, ax=ax, s=sz,
                          color=style["accent"],
                          edgecolors=style["pitch_lines"],
                          linewidth=style["marker_edge_width"],
                          alpha=style["alpha"])

    set_chart_title(ax, title, style)
    if style["tight_layout"]: fig.tight_layout()
    return fig


def chart_delivery_start_map(df, theme_name, flip_y=False, style_overrides=None):
    return _pitch_chart(df, theme_name, flip_y, style_overrides,
                        "Delivery Start Map", "x_start_plot","y_start_plot")

def chart_delivery_heatmap(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6,8) if vertical else (8,6)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.12, vertical=vertical)

    dd = dff.dropna(subset=["x2","y2"]).copy()
    if len(dd):
        px = dd["y2"] if vertical else dd["x2"]
        py = dd["x2"] if vertical else dd["y2"]
        try:
            pitch.kdeplot(px, py, ax=ax, fill=True, levels=50,
                          alpha=style["kde_alpha"], cmap=style["heatmap_cmap"])
        except Exception:
            pitch.scatter(px, py, ax=ax, s=style["marker_size"]*0.65,
                          color=style["accent"], alpha=style["alpha"])

    set_chart_title(ax, "Delivery End Location Heatmap", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_end_scatter(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6,8) if vertical else (8,6)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, vertical=vertical)

    dd = dff.dropna(subset=["x2","y2"]).copy()
    if not len(dd):
        set_chart_title(ax, "Delivery End Scatter", style)
        if style["tight_layout"]: fig.tight_layout()
        return fig

    color_map = _get_delivery_color_map(style)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dtype, grp in dd.groupby("delivery_type"):
            px = grp["y2"] if vertical else grp["x2"]
            py = grp["x2"] if vertical else grp["y2"]
            pitch.scatter(px, py, ax=ax, s=style["marker_size"],
                          color=color_map.get(str(dtype).lower(), style["accent"]),
                          edgecolors=style["pitch_lines"],
                          linewidth=style["marker_edge_width"],
                          label=str(dtype).title(), alpha=style["alpha"])
        if style.get("show_legend",True):
            leg = ax.legend(frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
            style_legend(leg, style)
    else:
        px = dd["y2"] if vertical else dd["x2"]
        py = dd["x2"] if vertical else dd["y2"]
        pitch.scatter(px, py, ax=ax, s=style["marker_size"],
                      color=style["accent"],
                      edgecolors=style["pitch_lines"],
                      linewidth=style["marker_edge_width"],
                      alpha=style["alpha"])

    set_chart_title(ax, "Delivery End Scatter", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_trajectories(df, theme_name, flip_y=False, style_overrides=None):
    return _plot_delivery_trajectories_core(df, theme_name, flip_y, style_overrides, "Delivery Trajectories")

def chart_delivery_trajectories_left(df, theme_name, flip_y=False, style_overrides=None):
    return _plot_delivery_trajectories_core(df, theme_name, flip_y, style_overrides,
                                            "Delivery Trajectories - Left Corners", corner_filter="left")

def chart_delivery_trajectories_right(df, theme_name, flip_y=False, style_overrides=None):
    return _plot_delivery_trajectories_core(df, theme_name, flip_y, style_overrides,
                                            "Delivery Trajectories - Right Corners", corner_filter="right")

def chart_average_delivery_path(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6.2,8.2) if vertical else (8.2,6.2)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, vertical=vertical)

    dd        = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    color_map = _get_delivery_color_map(style)

    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dtype, grp in dd.groupby("delivery_type"):
            if not len(grp): continue
            ax1,ay1,ax2,ay2 = grp["x_start_plot"].mean(), grp["y_start_plot"].mean(), grp["x2"].mean(), grp["y2"].mean()
            mc = grp["corner_label"].mode().iloc[0] if grp["corner_label"].notna().any() else "unknown"
            rad = _curve_rad_for_delivery(str(dtype).lower(), str(mc))
            x1p = ay1 if vertical else ax1; y1p = ax1 if vertical else ay1
            x2p = ay2 if vertical else ax2; y2p = ax2 if vertical else ay2
            c = color_map.get(str(dtype).lower(), style["accent"])
            _draw_curved_arrow(ax, x1p,y1p,x2p,y2p, c, style, rad=rad, lw_mult=2.2, alpha_mult=1.15)
            pitch.scatter([x2p],[y2p], ax=ax, s=style["marker_size"]*1.2, color=c,
                          edgecolors=style["pitch_lines"], linewidth=1.2)
        if style.get("show_legend",True):
            handles,labels = [],[]
            for k,v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    handles.append(Line2D([0],[0],color=v,lw=style["trajectory_width"]*2)); labels.append(f"{k.title()} Avg")
            if handles:
                leg = ax.legend(handles,labels,frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
                style_legend(leg, style)
    else:
        ax1,ay1 = dd["x_start_plot"].mean(), dd["y_start_plot"].mean()
        ax2,ay2 = dd["x2"].mean(), dd["y2"].mean()
        x1p = ay1 if vertical else ax1; y1p = ax1 if vertical else ay1
        x2p = ay2 if vertical else ax2; y2p = ax2 if vertical else ay2
        _draw_curved_arrow(ax, x1p,y1p,x2p,y2p, style["accent"], style, rad=0.0, lw_mult=2.2)

    set_chart_title(ax, "Average Delivery Path", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_heat_plus_trajectories(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6.4,8.4) if vertical else (8.4,6.4)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.10, vertical=vertical)

    dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if len(dd):
        px = dd["y2"] if vertical else dd["x2"]
        py = dd["x2"] if vertical else dd["y2"]
        try: pitch.kdeplot(px,py,ax=ax,fill=True,levels=40,alpha=style["kde_alpha"]*0.65,cmap=style["heatmap_cmap"])
        except: pass

    color_map = _get_delivery_color_map(style)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dtype, grp in dd.groupby("delivery_type"):
            dl = str(dtype).lower()
            for _,r in grp.iterrows():
                rad = _curve_rad_for_delivery(dl, str(r.get("corner_label","unknown")))
                x1p = r["y_start_plot"] if vertical else r["x_start_plot"]
                y1p = r["x_start_plot"] if vertical else r["y_start_plot"]
                x2p = r["y2"] if vertical else r["x2"]
                y2p = r["x2"] if vertical else r["y2"]
                _draw_curved_arrow(ax, x1p,y1p,x2p,y2p, color_map.get(dl,style["accent"]),style,rad=rad,lw_mult=0.9,alpha_mult=0.9)
        if style.get("show_legend",True):
            handles,labels = [],[]
            for k,v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    handles.append(Line2D([0],[0],color=v,lw=style["trajectory_width"]+0.6)); labels.append(k.title())
            if handles:
                leg = ax.legend(handles,labels,frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
                style_legend(leg, style)

    set_chart_title(ax, "Heat + Trajectories", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_trajectory_clusters(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6.4,8.4) if vertical else (8.4,6.4)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.10, vertical=vertical)

    dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if not len(dd):
        set_chart_title(ax, "Trajectory Clusters", style)
        if style["tight_layout"]: fig.tight_layout()
        return fig

    dd["cluster"] = _safe_cluster_labels(dd, n_clusters=3)
    palette  = [style["accent"],style["warning"],style["success"],style["accent_2"],style["danger"]]
    handles,labels = [],[]
    for i,(cid,grp) in enumerate(dd.groupby("cluster")):
        color = palette[i % len(palette)]
        for _,r in grp.iterrows():
            rad = _curve_rad_for_delivery(str(r.get("delivery_type","")).lower(), str(r.get("corner_label","unknown")))
            x1p = r["y_start_plot"] if vertical else r["x_start_plot"]
            y1p = r["x_start_plot"] if vertical else r["y_start_plot"]
            x2p = r["y2"] if vertical else r["x2"]
            y2p = r["x2"] if vertical else r["y2"]
            _draw_curved_arrow(ax, x1p,y1p,x2p,y2p, color, style, rad=rad, lw_mult=0.85, alpha_mult=0.65)
        mc = grp["corner_label"].mode().iloc[0] if grp["corner_label"].notna().any() else "unknown"
        md = grp["delivery_type"].mode().iloc[0] if ("delivery_type" in grp.columns and grp["delivery_type"].notna().any()) else ""
        ar = _curve_rad_for_delivery(str(md).lower(), str(mc))
        gx1 = grp["y_start_plot"].mean() if vertical else grp["x_start_plot"].mean()
        gy1 = grp["x_start_plot"].mean() if vertical else grp["y_start_plot"].mean()
        gx2 = grp["y2"].mean() if vertical else grp["x2"].mean()
        gy2 = grp["x2"].mean() if vertical else grp["y2"].mean()
        _draw_curved_arrow(ax, gx1,gy1,gx2,gy2, color, style, rad=ar, lw_mult=2.5, alpha_mult=1.1)
        handles.append(Line2D([0],[0],color=color,lw=style["trajectory_width"]*2))
        labels.append(f"Cluster {int(cid)+1} ({len(grp)})")

    if style.get("show_legend",True) and handles:
        leg = ax.legend(handles,labels,frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
        style_legend(leg, style)

    set_chart_title(ax, "Trajectory Clusters", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig


# =========================================================
# BAR CHARTS
# =========================================================
def chart_delivery_length_distribution(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6,4.8))
    dff = _prepare_delivery_df(df, flip_y)
    dd  = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    lengths = (((dd["x2"]-dd["x_start_plot"])**2 + (dd["y2"]-dd["y_start_plot"])**2)**0.5
               if len(dd) else pd.Series(dtype=float))
    bar_color = style.get("bar_colors",{}).get("default", style["accent"])
    ax.hist(lengths, bins=12, color=bar_color, edgecolor=style["lines"], linewidth=0.8, alpha=0.92)
    themed_bar(ax, style)
    set_chart_title(ax, "Delivery Length Distribution", style)
    ax.set_xlabel("Length"); ax.set_ylabel("Count")
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_direction_map(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6,4.8))
    dff = _prepare_delivery_df(df, flip_y)
    dd  = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if not len(dd):
        summary = pd.Series(dtype=float)
    else:
        dx = dd["x2"]-dd["x_start_plot"]; dy = dd["y2"]-dd["y_start_plot"]
        angles  = dy.combine(dx, lambda yv,xv: math.degrees(math.atan2(yv,xv)))
        lbls    = pd.cut(angles, bins=[-181,-60,-10,10,60,181],
                         labels=["Down","Down-In","Straight","Up-In","Up"], include_lowest=True)
        summary = lbls.value_counts().reindex(["Down","Down-In","Straight","Up-In","Up"]).fillna(0)
    bar_color = style.get("bar_colors",{}).get("default", style["accent_2"])
    ax.bar(summary.index.astype(str), summary.values, color=bar_color, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Delivery Direction Map", style)
    ax.set_ylabel("Count")
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_outcome_distribution(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.4,4.6))
    counts = get_set_piece_series(df).str.lower().value_counts()
    bar_colors_map = style.get("bar_colors",{})
    colors = []
    for x in counts.index:
        if x in ["successful","corner","free_kick"]: colors.append(bar_colors_map.get("success", style["success"]))
        elif x in ["unsuccessful","failed","loss"]:  colors.append(bar_colors_map.get("danger",  style["danger"]))
        else:                                         colors.append(bar_colors_map.get("default", style["accent"]))
    ax.bar(counts.index, counts.values, color=colors, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Set Piece Type / Outcome Distribution", style)
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=25)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_target_zone_breakdown(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.4,4.6))
    dff    = _prepare_delivery_df(df, flip_y)
    counts = get_target_zone_series(dff).value_counts()
    bar_color = style.get("bar_colors",{}).get("default", style["accent"])
    ax.bar(counts.index, counts.values, color=bar_color, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Target Zone Breakdown", style)
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=25)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_first_contact_win_by_zone(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6,4.8))
    dff = _prepare_delivery_df(df, flip_y); dd = dff.copy()
    dd["zone_calc"]   = get_target_zone_series(dd)
    dd["fc_win_calc"] = get_first_contact_win_series(dd)
    summary   = dd.groupby("zone_calc", dropna=False)["fc_win_calc"].mean().sort_values(ascending=False)*100
    bar_color = style.get("bar_colors",{}).get("default", style["accent"])
    ax.bar(summary.index.astype(str), summary.values, color=bar_color, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "First Contact Win % By Zone", style)
    ax.set_ylabel("Win %"); ax.tick_params(axis="x", rotation=25)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_routine_breakdown(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6,4.8))
    if "routine_type" in df.columns and df["routine_type"].notna().any():
        counts = df["routine_type"].fillna("unclassified").value_counts().head(10)
    else:
        counts = get_target_zone_series(df).value_counts().head(10)
    bar_color = style.get("bar_colors",{}).get("default", style["warning"])
    ax.barh(counts.index[::-1], counts.values[::-1], color=bar_color, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Routine Breakdown", style)
    ax.set_xlabel("Count")
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_taker_profile(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.8,4.8))
    if "taker" in df.columns and "sequence_id" in df.columns:
        seq_counts = df.groupby("taker")["sequence_id"].nunique().sort_values(ascending=False).head(10)
    elif "taker" in df.columns:
        seq_counts = df["taker"].value_counts().head(10)
    else:
        seq_counts = get_set_piece_series(df).value_counts().head(10)
    bar_color = style.get("bar_colors",{}).get("default", style["accent"])
    ax.barh(seq_counts.index[::-1], seq_counts.values[::-1], color=bar_color, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Taker / Event Profile", style)
    ax.set_xlabel("Count")
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_structure_zone_averages(df, theme_name, flip_y=False, style_overrides=None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6,4.8))
    cols = ["players_near_post","players_far_post","players_6yard","players_penalty"]
    existing = [c for c in cols if c in df.columns]
    if existing:
        means = df[existing].apply(pd.to_numeric,errors="coerce").mean().fillna(0)
        lmap  = {"players_near_post":"Near Post","players_far_post":"Far Post","players_6yard":"6 Yard","players_penalty":"Penalty"}
        lbls  = [lmap[c] for c in existing]; vals = means.values
    else:
        vc = get_target_zone_series(df).value_counts(); lbls = vc.index.tolist(); vals = vc.values
    colors = [style["accent"],style["warning"],style["success"],style["accent_2"]][:len(vals)]
    # Override with bar_colors if set
    bc = style.get("bar_colors",{}).get("default")
    if bc: colors = [bc]*len(vals)
    ax.bar(lbls, vals, color=colors, edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Structure / Zone Summary", style)
    ax.set_ylabel("Value"); ax.tick_params(axis="x", rotation=15)
    if style["tight_layout"]: fig.tight_layout()
    return fig


# =========================================================
# PITCH CHARTS (continued)
# =========================================================
def chart_shot_map(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)
    figsize  = (6,8) if vertical else (8,6)
    fig, ax  = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.10, vertical=vertical)
    dd = dff.dropna(subset=["x","y"]).copy()
    sz = style["marker_size"]*1.6
    if "xg" in dd.columns:
        xg = pd.to_numeric(dd["xg"],errors="coerce").fillna(0); sz = 35+xg*500
    px = dd["y"] if vertical else dd["x"]; py = dd["x"] if vertical else dd["y"]
    pitch.scatter(px, py, ax=ax, s=sz, color=style["success"],
                  edgecolors=style["pitch_lines"], linewidth=style["marker_edge_width"], alpha=style["alpha"])
    set_chart_title(ax, "Set Piece Shot / Event Map", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_second_ball_map(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)
    figsize  = (6,8) if vertical else (8,6)
    fig, ax  = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.10, vertical=vertical)
    dd = dff.dropna(subset=["x","y"]).copy()
    dd["sb"] = get_second_ball_win_series(dd)
    win = dd[dd["sb"]==1]; lose = dd[dd["sb"]==0]
    if len(win):
        px=win["y"] if vertical else win["x"]; py=win["x"] if vertical else win["y"]
        pitch.scatter(px,py,ax=ax,s=style["marker_size"]*1.25,color=style["success"],
                      edgecolors=style["pitch_lines"],linewidth=style["marker_edge_width"],label="Won",alpha=style["alpha"])
    if len(lose):
        px=lose["y"] if vertical else lose["x"]; py=lose["x"] if vertical else lose["y"]
        pitch.scatter(px,py,ax=ax,s=style["marker_size"]*1.25,facecolors="none",
                      edgecolors=style["danger"],linewidth=style["line_width"]+0.6,label="Lost",alpha=style["alpha"])
    if (len(win) or len(lose)) and style.get("show_legend",True):
        leg = ax.legend(frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=2); style_legend(leg, style)
    set_chart_title(ax, "Second Ball Map", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_defensive_vulnerability_map(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)
    figsize  = (6,8) if vertical else (8,6)
    fig, ax  = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    if style.get("show_thirds",False): _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.10, vertical=vertical)
    dd = dff.dropna(subset=["x","y"]).copy()
    if len(dd):
        px = dd["y"] if vertical else dd["x"]; py = dd["x"] if vertical else dd["y"]
        try: pitch.kdeplot(px,py,ax=ax,fill=True,levels=40,alpha=style["kde_alpha"],cmap=style["heatmap_cmap"])
        except: pitch.scatter(px,py,ax=ax,s=style["marker_size"]*0.8,color=style["danger"],
                               edgecolors=style["pitch_lines"],linewidth=style["marker_edge_width"],alpha=style["alpha"])
    set_chart_title(ax, "Defensive Vulnerability Map", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig

def chart_set_piece_landing_heatmap(df, theme_name, flip_y=False, style_overrides=None):
    style    = resolve_style(theme_name, style_overrides)
    vertical = style.get("pitch_vertical", False)
    pitch    = make_pitch(style, vertical=vertical)
    dff      = _prepare_delivery_df(df, flip_y)

    figsize = (6.2,8.8) if vertical else (10.0,6.8)
    fig, ax = _base_figure(style, figsize=figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax, style, vertical=vertical)
    _draw_thirds_lines(ax, style, vertical=vertical)
    _draw_box_zone_overlay(ax, style, alpha=0.14, vertical=vertical)

    dd = dff.dropna(subset=["x2","y2"]).copy()
    if not len(dd):
        set_chart_title(ax, "Set Piece Landing Heatmap", style)
        if style["tight_layout"]: fig.tight_layout()
        return fig

    px = dd["y2"] if vertical else dd["x2"]
    py = dd["x2"] if vertical else dd["y2"]
    try: pitch.kdeplot(px,py,ax=ax,fill=True,levels=60,alpha=style["kde_alpha"],cmap=style["heatmap_cmap"],zorder=4)
    except: pitch.scatter(px,py,ax=ax,s=style["marker_size"]*0.7,color=style["accent"],alpha=style["alpha"]*0.8)
    pitch.scatter(px,py,ax=ax,s=max(style["marker_size"]*0.35,12),color=style["text"],edgecolors="none",
                  alpha=min(style["alpha"]*0.55,0.6),zorder=5)

    src_col = py if vertical else px
    for (lo,hi),tname in zip([(0,100/3),(100/3,200/3),(200/3,100)],["Def Third","Mid Third","Att Third"]):
        cnt  = int(((src_col>=lo)&(src_col<hi)).sum())
        pct  = cnt/max(len(dd),1)*100
        mid  = (lo+hi)/2
        tx   = (32) if vertical else mid
        ty   = mid if vertical else (-style["pitch_pad_y"]*0.55)
        ax.text(tx,ty,f"{tname}\n{cnt} ({pct:.0f}%)",ha="center",
                va="center" if vertical else "top",fontsize=max(style["tick_size"]-1,7),
                color=style["muted"],alpha=0.85,zorder=6)

    set_chart_title(ax, "Set Piece Landing Heatmap", style)
    if style["tight_layout"]: fig.tight_layout()
    return fig


# =========================================================
# TAKER STATS TABLE  (shirt-mockup style)
# =========================================================
def chart_taker_stats_table(df, theme_name, flip_y=False, style_overrides=None):
    """
    Infographic table inspired by the EPL crosses chart.
    Rows = takers, columns = key stats, left icon = shirt mockup with sequence number.
    """
    style = resolve_style(theme_name, style_overrides)

    # ── build stats ───────────────────────────────────────────────────────
    if "taker" not in df.columns:
        fig, ax = _base_figure(style, figsize=(9, 4))
        ax.text(0.5, 0.5, "No 'taker' column found", ha="center", va="center",
                color=style["text"], fontsize=12, transform=ax.transAxes)
        return fig

    dff = df.copy()
    stats_rows = []
    for taker, grp in dff.groupby("taker"):
        if str(taker).lower() in ("nan","none",""): continue
        n_seq = int(grp["sequence_id"].nunique()) if "sequence_id" in grp.columns else len(grp)
        n_ins = int((grp["delivery_type"].astype(str).str.lower()=="inswing").sum()) if "delivery_type" in grp.columns else 0
        n_out = int((grp["delivery_type"].astype(str).str.lower()=="outswing").sum()) if "delivery_type" in grp.columns else 0
        n_suc = int((grp["outcome"].astype(str).str.lower().isin(["successful","success","won"])).sum()) if "outcome" in grp.columns else 0
        suc_r = round(n_suc / max(len(grp),1)*100, 1)
        n_left  = int((grp["side"].astype(str).str.lower()=="left").sum())  if "side" in grp.columns else 0
        n_right = int((grp["side"].astype(str).str.lower()=="right").sum()) if "side" in grp.columns else 0
        stats_rows.append({
            "taker": str(taker).title(),
            "sequences": n_seq,
            "inswing":   n_ins,
            "outswing":  n_out,
            "left":      n_left,
            "right":     n_right,
            "success_rate": suc_r,
        })

    if not stats_rows:
        fig, ax = _base_figure(style, figsize=(9, 4))
        ax.text(0.5, 0.5, "No taker data available", ha="center", va="center",
                color=style["text"], fontsize=12, transform=ax.transAxes)
        return fig

    stats_df = pd.DataFrame(stats_rows).sort_values("sequences", ascending=False).head(12).reset_index(drop=True)
    n_rows   = len(stats_df)

    # ── layout ─────────────────────────────────────────────────────────────
    row_h    = 0.72          # height per row in inches
    header_h = 1.0
    fig_h    = header_h + n_rows * row_h + 0.4
    fig_w    = 9.5

    apply_global_rcparams(style)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(style["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.set_facecolor(style["bg"])
    ax.axis("off")

    # colours
    body_col   = style.get("shirt_body_color",   style["accent"])
    sleeve_col = style.get("shirt_sleeve_color",  style["panel"])
    num_col    = style.get("shirt_number_color",  style["bg"])
    max_bar_w  = 2.2

    # ── title ───────────────────────────────────────────────────────────────
    ax.text(fig_w/2, fig_h - 0.28, "Set Piece Taker Stats",
            ha="center", va="top", fontsize=style["title_size"]+2,
            fontweight="bold", color=style["text"])
    ax.text(fig_w/2, fig_h - 0.62,
            "Sorted by number of set piece sequences taken",
            ha="center", va="top", fontsize=style["tick_size"],
            color=style["muted"])

    # ── column headers ───────────────────────────────────────────────────────
    col_x = {"shirt":0.40, "name":1.05, "seq":3.55, "ins":4.40, "out":5.25,
              "left":6.05, "right":6.85, "rate":7.80}
    col_labels = {"seq":"SEQ","ins":"INSWING","out":"OUTSWING",
                  "left":"LEFT","right":"RIGHT","rate":"SUCCESS %"}
    header_y = fig_h - header_h + 0.05
    for k, label in col_labels.items():
        ax.text(col_x[k], header_y, label, ha="center", va="bottom",
                fontsize=style["tick_size"]-1, color=style["muted"],
                fontweight="bold")

    # ── separator under header ────────────────────────────────────────────────
    ax.axhline(header_y - 0.02, xmin=0.02, xmax=0.98,
               color=style["lines"], linewidth=0.8, alpha=0.6)

    # ── rows ──────────────────────────────────────────────────────────────────
    for i, row in stats_df.iterrows():
        y_center = fig_h - header_h - (i + 0.5) * row_h

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.1, y_center - row_h/2 + 0.04),
                                    fig_w - 0.2, row_h - 0.08,
                                    facecolor=style["panel"], edgecolor="none",
                                    alpha=0.35, zorder=0))

        # ── shirt icon ──────────────────────────────────────────────────────
        _draw_shirt(ax, cx=col_x["shirt"], cy=y_center,
                    size=row_h*0.78,
                    body_color=body_col, sleeve_color=sleeve_col,
                    number=row["sequences"],
                    number_color=num_col,
                    text_color=style["lines"],
                    number_font_size=max(style["tick_size"]-1, 7))

        # ── name ────────────────────────────────────────────────────────────
        ax.text(col_x["name"], y_center, row["taker"],
                ha="left", va="center", fontsize=style["tick_size"]+0.5,
                fontweight="bold", color=style["text"])

        # ── numeric stats ────────────────────────────────────────────────────
        for key in ["seq","ins","out","left","right"]:
            val = {"seq":row["sequences"],"ins":row["inswing"],"out":row["outswing"],
                   "left":row["left"],"right":row["right"]}[key]
            ax.text(col_x[key], y_center, str(val),
                    ha="center", va="center", fontsize=style["tick_size"],
                    color=style["text"])

        # ── success rate bar ─────────────────────────────────────────────────
        rate  = row["success_rate"] / 100.0
        bar_x = col_x["rate"] - max_bar_w/2
        bar_h = row_h * 0.30
        # background track
        ax.add_patch(Rectangle((bar_x, y_center - bar_h/2), max_bar_w, bar_h,
                                facecolor=style["lines"], edgecolor="none",
                                alpha=0.35, zorder=1))
        # filled portion
        bar_color_filled = style.get("bar_colors",{}).get("default", style["accent"])
        ax.add_patch(Rectangle((bar_x, y_center - bar_h/2), max_bar_w*rate, bar_h,
                                facecolor=bar_color_filled, edgecolor="none",
                                alpha=0.90, zorder=2))
        # label
        ax.text(bar_x + max_bar_w*rate + 0.06, y_center,
                f"{row['success_rate']:.1f}%",
                ha="left", va="center",
                fontsize=style["tick_size"]-1, color=style["text"])

        # separator
        ax.axhline(y_center - row_h/2 + 0.04, xmin=0.02, xmax=0.98,
                   color=style["lines"], linewidth=0.4, alpha=0.3)

    if style["tight_layout"]:
        fig.tight_layout(pad=0.3)
    return fig


# =========================================================
# CHART BUILDERS REGISTRY
# =========================================================
CHART_BUILDERS = {
    "Delivery Start Map":                   chart_delivery_start_map,
    "Delivery Heatmap":                     chart_delivery_heatmap,
    "Delivery End Scatter":                 chart_delivery_end_scatter,
    "Delivery Trajectories":                chart_delivery_trajectories,
    "Delivery Trajectories - Left Corners": chart_delivery_trajectories_left,
    "Delivery Trajectories - Right Corners":chart_delivery_trajectories_right,
    "Average Delivery Path":                chart_average_delivery_path,
    "Heat + Trajectories":                  chart_heat_plus_trajectories,
    "Trajectory Clusters":                  chart_trajectory_clusters,
    "Delivery Length Distribution":         chart_delivery_length_distribution,
    "Delivery Direction Map":               chart_delivery_direction_map,
    "Outcome Distribution":                 chart_outcome_distribution,
    "Target Zone Breakdown":                chart_target_zone_breakdown,
    "First Contact Win By Zone":            chart_first_contact_win_by_zone,
    "Routine Breakdown":                    chart_routine_breakdown,
    "Shot Map":                             chart_shot_map,
    "Second Ball Map":                      chart_second_ball_map,
    "Defensive Vulnerability Map":          chart_defensive_vulnerability_map,
    "Taker Profile":                        chart_taker_profile,
    "Structure Zone Averages":              chart_structure_zone_averages,
    "Set Piece Landing Heatmap":            chart_set_piece_landing_heatmap,
    "Taker Stats Table":                    chart_taker_stats_table,
}
