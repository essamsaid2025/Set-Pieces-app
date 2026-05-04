import io, math, os, tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpl_pe
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, FancyBboxPatch

from data_utils import bool01
from ui_theme import build_chart_style

try:
    from mplsoccer import Pitch as MplsoccerPitch, VerticalPitch
    _MPLSOCCER = True
except Exception:
    MplsoccerPitch = None; VerticalPitch = None; _MPLSOCCER = False

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# ─────────────────────────────────────────────────────────────────────────────
# PITCH CONSTANTS  (100×64 coordinate space)
# ─────────────────────────────────────────────────────────────────────────────
PL, PW        = 100.0, 64.0
BOX_X0, BOX_X1 = 83.5,  100.0
BOX_Y0, BOX_Y1 = 13.84, 50.16
SIX_X0        = 94.5
SIX_Y0, SIX_Y1 = 24.84, 39.16
GOAL_Y0, GOAL_Y1 = 28.34, 35.66
BOX_W = BOX_X1 - BOX_X0   # 16.5
BOX_H = BOX_Y1 - BOX_Y0   # 36.32

# ─────────────────────────────────────────────────────────────────────────────
# BARCELONA ZONES
# ─────────────────────────────────────────────────────────────────────────────
def _barca_zones(corner_side: str):
    if corner_side == "left":
        return [
            ("Near Post\nShort",  BOX_X0, 0.0,     BOX_W, BOX_Y0),
            ("Near\nPost",        BOX_X0, BOX_Y0,  BOX_W, SIX_Y0 - BOX_Y0),
            ("Small\nArea",       SIX_X0, SIX_Y0,  BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
            ("Penalty\nSpot",     BOX_X0, SIX_Y0,  SIX_X0 - BOX_X0, SIX_Y1 - SIX_Y0),
            ("Far\nPost",         BOX_X0, SIX_Y1,  BOX_W, BOX_Y1 - SIX_Y1),
            ("Far Post\nLong",    BOX_X0, BOX_Y1,  BOX_W, PW - BOX_Y1),
            ("Box\nFront",        72.0,   BOX_Y0,  BOX_X0 - 72.0, BOX_H),
        ]
    else:
        return [
            ("Near Post\nShort",  BOX_X0, BOX_Y1,  BOX_W, PW - BOX_Y1),
            ("Near\nPost",        BOX_X0, SIX_Y1,  BOX_W, BOX_Y1 - SIX_Y1),
            ("Small\nArea",       SIX_X0, SIX_Y0,  BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
            ("Penalty\nSpot",     BOX_X0, SIX_Y0,  SIX_X0 - BOX_X0, SIX_Y1 - SIX_Y0),
            ("Far\nPost",         BOX_X0, BOX_Y0,  BOX_W, SIX_Y0 - BOX_Y0),
            ("Far Post\nLong",    BOX_X0, 0.0,     BOX_W, BOX_Y0),
            ("Box\nFront",        72.0,   BOX_Y0,  BOX_X0 - 72.0, BOX_H),
        ]

ZONE_CKEYS = ["accent", "accent_2", "warning", "success", "accent_2", "accent", "muted"]

# ─────────────────────────────────────────────────────────────────────────────
# CHART REQUIREMENTS
# ─────────────────────────────────────────────────────────────────────────────
CHART_REQUIREMENTS: Dict[str, List[str]] = {
    "Delivery Start Map":                      ["x", "y"],
    "Delivery Heatmap":                        ["x2", "y2"],
    "Delivery End Scatter - Left Corner":      ["x2", "y2"],
    "Delivery End Scatter - Right Corner":     ["x2", "y2"],
    "Delivery Trajectories - Left Corners":    ["x", "y", "x2", "y2"],
    "Delivery Trajectories - Right Corners":   ["x", "y", "x2", "y2"],
    "Attack Free Kick Trajectories":           ["x", "y", "x2", "y2"],
    "Average Delivery Path":                   ["x", "y", "x2", "y2"],
    "Heat + Trajectories":                     ["x", "y", "x2", "y2"],
    "Trajectory Clusters":                     ["x", "y", "x2", "y2"],
    "Delivery Length Distribution":            ["x", "y", "x2", "y2"],
    "Delivery Direction Map":                  ["x", "y", "x2", "y2"],
    "Outcome Distribution":                    ["set_piece_type"],
    "Target Zone Breakdown":                   ["x2", "y2"],
    "Zone Delivery Count Map - Left Corner":   ["x2", "y2"],
    "Zone Delivery Count Map - Right Corner":  ["x2", "y2"],
    "Avg Players Per Zone - Left Corner":      ["players_near_post"],
    "Avg Players Per Zone - Right Corner":     ["players_near_post"],
    "First Contact Win By Zone":               ["x2", "y2"],
    "Routine Breakdown":                       ["set_piece_type"],
    "Shot Map":                                ["x", "y"],
    "Second Ball Map":                         ["x", "y"],
    "Defensive Vulnerability Map":             ["x", "y"],
    "Taker Profile":                           ["set_piece_type"],
    "Structure Zone Averages":                 [],
    "Set Piece Landing Heatmap":               ["x2", "y2"],
    "Taker Stats Table":                       ["taker"],
    "First Contact Location Map":              ["x2", "y2"],
    "First Contact Players by Shirt Number":   ["first_contact_player"],
    "Players Who Made First Contact":          ["first_contact_player"],
    "Players That Lost First Contact":         ["lost_first_contact_player"],
    "Box Marking Scheme":                      ["man_marking_in_box", "zonal_marking_in_box"],
    # ── NEW DEFENSIVE CHARTS ─────────────────────────────────────────────────
    "Defensive Shape Map":                     ["x2", "y2"],
    "Defender vs Attacker Zone Matchup":       ["x2", "y2"],
    "Clearance Outcome Map":                   ["x2", "y2"],
    "Set Piece Conceded Heatmap":              ["x2", "y2"],
    "Defensive Success Rate By Zone":          ["x2", "y2"],
    "First Contact Win Rate Trend":            [],
    "Second Ball Recovery Map":                ["x2", "y2"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE PITCH
# ─────────────────────────────────────────────────────────────────────────────
class SimplePitch:
    def __init__(self, pitch_length=100, pitch_width=64, pitch_color="#1F5F3B",
                 line_color="#FFFFFF", linewidth=1.4, stripe=False, stripe_color=None,
                 line_zorder=2, vertical=False):
        self.pl=pitch_length; self.pw=pitch_width; self.pitch_color=pitch_color
        self.line_color=line_color; self.linewidth=linewidth
        self.stripe=stripe; self.stripe_color=stripe_color
        self.lz=line_zorder; self.vertical=vertical

    def draw(self, ax):
        ax.set_facecolor(self.pitch_color)
        W = self.pw if self.vertical else self.pl
        H = self.pl if self.vertical else self.pw
        lc, lw, lz = self.line_color, self.linewidth, self.lz

        def _r(x,y,w,h): ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=lc,linewidth=lw,zorder=lz))
        def _c(cx,cy,r): ax.add_patch(plt.Circle((cx,cy),r,fill=False,color=lc,lw=lw,zorder=lz))

        if self.stripe and self.stripe_color:
            sw=W/10
            for i in range(10):
                if i%2==0: ax.add_patch(Rectangle((i*sw,0),sw,H,facecolor=self.stripe_color,edgecolor="none",alpha=0.35,zorder=0))

        _r(0,0,W,H)
        if self.vertical:
            ax.plot([0,W],[H/2,H/2],color=lc,lw=lw,zorder=lz)
            _c(W/2,H/2,9.15)
            _r((W-40.32)/2,0,40.32,16.5); _r((W-18.32)/2,0,18.32,5.5)
            _r((W-40.32)/2,H-16.5,40.32,16.5); _r((W-18.32)/2,H-5.5,18.32,5.5)
            ax.scatter([W/2,W/2],[11,H-11],c=lc,s=10,zorder=lz)
            ax.add_patch(Arc((W/2,11),18.3,18.3,angle=0,theta1=40,theta2=140,color=lc,lw=lw,zorder=lz))
            ax.add_patch(Arc((W/2,H-11),18.3,18.3,angle=0,theta1=220,theta2=320,color=lc,lw=lw,zorder=lz))
            _r((W-7.32)/2,-1.5,7.32,1.5); _r((W-7.32)/2,H,7.32,1.5)
            ax.set_xlim(0,W); ax.set_ylim(0,H)
        else:
            ax.plot([W/2,W/2],[0,H],color=lc,lw=lw,zorder=lz)
            _c(W/2,H/2,9.15)
            _r(0,(H-40.32)/2,16.5,40.32); _r(0,(H-18.32)/2,5.5,18.32)
            _r(W-16.5,(H-40.32)/2,16.5,40.32); _r(W-5.5,(H-18.32)/2,5.5,18.32)
            ax.scatter([11,W-11],[H/2,H/2],c=lc,s=10,zorder=lz)
            ax.add_patch(Arc((11,H/2),18.3,18.3,angle=0,theta1=310,theta2=50,color=lc,lw=lw,zorder=lz))
            ax.add_patch(Arc((W-11,H/2),18.3,18.3,angle=0,theta1=130,theta2=230,color=lc,lw=lw,zorder=lz))
            _r(-1.5,H/2-3.66,1.5,7.32); _r(W,H/2-3.66,1.5,7.32)
            ax.set_xlim(0,W); ax.set_ylim(0,H)
        return ax

    def kdeplot(self, x, y, ax, fill=True, levels=40, alpha=0.72, cmap="Blues"):
        if self.vertical:
            return ax.hist2d(list(y), list(x), bins=[22,14], range=[[0,self.pl],[0,self.pw]], cmap=cmap, alpha=alpha)
        return ax.hist2d(list(x), list(y), bins=[22,14], range=[[0,self.pl],[0,self.pw]], cmap=cmap, alpha=alpha)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE UTILS
# ─────────────────────────────────────────────────────────────────────────────
def resolve_style(theme_name, style_overrides=None): return build_chart_style(theme_name, style_overrides or {})

def apply_rcparams(s):
    mpl.rcParams.update({
        "font.family": s["font_family"], "axes.titlesize": s["title_size"],
        "axes.labelsize": s["label_size"], "xtick.labelsize": s["tick_size"],
        "ytick.labelsize": s["tick_size"], "legend.fontsize": s["legend_size"],
    })

def make_pitch(s, vertical=False):
    stripe = bool(s.get("pitch_stripe"))
    if _MPLSOCCER:
        try:
            cls = VerticalPitch if (vertical and VerticalPitch) else MplsoccerPitch
            return cls(pitch_type="custom", pitch_length=100, pitch_width=64, line_zorder=2,
                       linewidth=s["line_width"], pitch_color=s["pitch"],
                       line_color=s["pitch_lines"], stripe=stripe,
                       stripe_color=s.get("pitch_stripe"))
        except Exception:
            pass
    return SimplePitch(pitch_length=100, pitch_width=64, pitch_color=s["pitch"],
                       line_color=s["pitch_lines"], linewidth=s["line_width"],
                       stripe=stripe, stripe_color=s.get("pitch_stripe"),
                       line_zorder=2, vertical=vertical)

def apply_flip_y(df, flip_y=False):
    out = df.copy()
    if not flip_y: return out
    for c in ["y","y2","y3"]:
        if c in out.columns: out[c] = 64 - pd.to_numeric(out[c], errors="coerce")
    return out

def themed_bar(ax, s):
    ax.set_facecolor(s["panel"])
    for sp in ax.spines.values(): sp.set_color(s["lines"]); sp.set_linewidth(1.0)
    ax.tick_params(colors=s["muted"], labelsize=s["tick_size"])
    ax.yaxis.label.set_color(s["muted"]); ax.xaxis.label.set_color(s["muted"])
    ax.title.set_color(s["text"])
    if s.get("show_grid", True):
        ax.grid(axis="y", alpha=s["grid_alpha"], color=s["lines"], linestyle="--", lw=0.8)
        ax.set_axisbelow(True)

def _setup_pitch_axes(ax, s, vertical=False):
    px, py = s["pitch_pad_x"], s["pitch_pad_y"]
    if vertical:
        ax.set_xlim(-py, PW + py); ax.set_ylim(-px, PL + px)
    else:
        ax.set_xlim(-px, PL + px); ax.set_ylim(-py, PW + py)
    ax.set_facecolor(s["pitch"])
    if not s.get("show_ticks", False): ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_autoscale_on(False)

def chart_title(ax, t, s):
    if s.get("show_title", True):
        ax.set_title(t, color=s["text"], fontsize=s["title_size"],
                     fontweight=s["title_weight"], pad=12)

def style_legend(leg, s):
    if not leg: return
    f = leg.get_frame()
    if f:
        f.set_facecolor(s.get("legend_bg", s["panel"]))
        f.set_edgecolor(s.get("legend_border", s["lines"]))
        f.set_alpha(0.95)
    for t in leg.get_texts():
        t.set_color(s.get("legend_text", s["text"]))

def fig_to_png_bytes(fig, dpi=260):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    buf.seek(0); return buf.getvalue()

def save_report_pdf(figures, filename="set_piece_report.pdf"):
    td = tempfile.mkdtemp(); pp = os.path.join(td, filename)
    with PdfPages(pp) as pdf:
        for f in figures: pdf.savefig(f, bbox_inches="tight", pad_inches=0.25)
    with open(pp, "rb") as f: return f.read()

def _base_fig(s, figsize=(8,6)):
    apply_rcparams(s)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(s["bg"]); ax.set_facecolor(s["panel"])
    return fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE SCALING
# ─────────────────────────────────────────────────────────────────────────────
def _clean_num(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _auto_scale(df):
    out = _clean_num(df.copy(), ["x","y","x2","y2","x3","y3"])
    ay = pd.concat([out[c].dropna() for c in ["y","y2","y3"] if c in out.columns], ignore_index=True)
    if len(ay):
        gmy = ay.max()
        if pd.notna(gmy) and gmy > 64.5:
            ys = 64.0 / gmy
            for c in ["y","y2","y3"]:
                if c in out.columns: out[c] = out[c] * ys
    ax_ = pd.concat([out[c].dropna() for c in ["x","x2","x3"] if c in out.columns], ignore_index=True)
    if len(ax_):
        gmx = ax_.max()
        if pd.notna(gmx) and gmx > 100.5:
            xs = 100.0 / gmx
            for c in ["x","x2","x3"]:
                if c in out.columns: out[c] = out[c] * xs
    for c in ["x","x2","x3"]:
        if c in out.columns: out[c] = out[c].clip(0, 100)
    for c in ["y","y2","y3"]:
        if c in out.columns: out[c] = out[c].clip(0, 64)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# ZONE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def _draw_zones(ax, s, corner_side="right", alpha=0.18, vertical=False, show_labels=False):
    zones = _barca_zones(corner_side)
    for (label, zx, zy, zw, zh), ck in zip(zones, ZONE_CKEYS):
        color = s.get(ck, s["accent"])
        if vertical:
            rx, ry, rw, rh = zy, zx, zh, zw
        else:
            rx, ry, rw, rh = zx, zy, zw, zh
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=color,
                                edgecolor=s["pitch_lines"], linewidth=0.7,
                                alpha=alpha, zorder=1))
        if show_labels:
            fs = max(s["tick_size"] - 2, 6)
            ax.text(rx + rw/2, ry + rh/2, label, ha="center", va="center",
                    fontsize=fs, color=s["text"], alpha=0.95, zorder=2,
                    linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=s["bg"],
                              edgecolor="none", alpha=0.45))

def _draw_thirds(ax, s, vertical=False):
    lc = s.get("pitch_lines", "#FFFFFF")
    lw = max(s["line_width"] * 0.75, 0.9)
    fs = max(s["tick_size"] - 1, 7)
    for pos in [PL/3, 2*PL/3]:
        if vertical: ax.axhline(pos, color=lc, lw=lw, alpha=0.55, linestyle="--", zorder=3)
        else:        ax.axvline(pos, color=lc, lw=lw, alpha=0.55, linestyle="--", zorder=3)
    for pos, lbl in [(PL/6,"Def Third"),(PL/2,"Mid Third"),(5*PL/6,"Att Third")]:
        if vertical:
            ax.text(PW + s["pitch_pad_y"]*0.3, pos, lbl, ha="left", va="center",
                    fontsize=fs, color=lc, alpha=0.70, zorder=4)
        else:
            ax.text(pos, PW + s["pitch_pad_y"]*0.3, lbl, ha="center", va="bottom",
                    fontsize=fs, color=lc, alpha=0.70, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _cmap_delivery(s):
    cm = s.get("arrow_colors", {})
    return {
        "inswing":  cm.get("inswing",  s["accent"]),
        "outswing": cm.get("outswing", s["warning"]),
        "straight": cm.get("straight", s["accent_2"]),
        "driven":   cm.get("driven",   s["success"]),
        "short":    cm.get("short",    s["danger"]),
    }

def _corner_anchor(x, y):
    if pd.isna(x) or pd.isna(y): return x, y, "unknown"
    side = "right" if x >= 50 else "left"
    half = "top"   if y <= 32  else "bottom"
    cx = 99.5 if side == "right" else 0.5
    cy = 0.5  if half == "top"   else 63.5
    return cx, cy, f"{side}_{half}"

def _curve_rad(dtype, corner_label):
    d = str(dtype).lower()
    if corner_label == "right_top":
        if d == "inswing": return 0.30
        if d == "outswing": return -0.15
    elif corner_label == "right_bottom":
        if d == "inswing": return -0.30
        if d == "outswing": return 0.15
    elif corner_label == "left_top":
        if d == "inswing": return -0.30
        if d == "outswing": return 0.15
    elif corner_label == "left_bottom":
        if d == "inswing": return 0.30
        if d == "outswing": return -0.15
    return 0.0

def _prep(df, flip_y=False):
    dff = apply_flip_y(df, flip_y)
    dff = _auto_scale(dff)
    if "x" in dff.columns and "y" in dff.columns:
        cd = dff.apply(lambda r: _corner_anchor(r.get("x"), r.get("y")), axis=1, result_type="expand")
        dff["x_start_plot"] = cd[0]; dff["y_start_plot"] = cd[1]
        dff["corner_label"] = cd[2]
        dff["corner_side"]  = dff["corner_label"].astype(str).str.split("_").str[0]
    else:
        dff["x_start_plot"] = dff.get("x"); dff["y_start_plot"] = dff.get("y")
        dff["corner_label"] = "unknown";    dff["corner_side"]  = "unknown"
    return dff

def _arrow(ax, x1, y1, x2, y2, color, s, rad=0.0, lm=1.0, am=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1,y1), (x2,y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=s["trajectory_headwidth"] * 4.0,
        linewidth=s["trajectory_width"] * lm,
        color=color,
        alpha=s["trajectory_alpha"] * am,
        clip_on=True, zorder=6,
    ))

def _side_dominant(df):
    if "side" in df.columns:
        vc = df["side"].dropna().astype(str).str.lower().value_counts()
        if len(vc): return vc.index[0]
    return "right"

def get_set_piece_series(df):
    for c in ["set_piece_type","event","outcome"]:
        if c in df.columns: return df[c].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)

def get_target_zone_series(df):
    if "target_zone" in df.columns: return df["target_zone"].fillna("unknown").astype(str)
    if "x2" in df.columns and "y2" in df.columns:
        return df.apply(lambda r: _infer_zone(r.get("x2"), r.get("y2")), axis=1)
    return pd.Series(["unknown"] * len(df), index=df.index)

def _infer_zone(x, y):
    try: x, y = float(x), float(y)
    except: return "unknown"
    if y < BOX_Y0: return "near_post_short"
    if y > BOX_Y1: return "far_post_long"
    if x < BOX_X0: return "box_front"
    if x >= SIX_X0: return "small_area"
    if y < SIX_Y0:  return "near_post"
    if y > SIX_Y1:  return "far_post"
    return "penalty_spot"

def _get_fcw(df):
    if "first_contact_win" in df.columns:
        col = df["first_contact_win"]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().any():
            return (num > 0).astype(int)
        return col.astype(str).str.strip().str.lower().isin(
            ["yes","1","true","won","win","successful","success"]
        ).astype(int)
    return get_set_piece_series(df).str.lower().isin(
        ["successful","success","won","win"]
    ).astype(int)

def _get_sbw(df):
    if "second_ball_win" in df.columns: return bool01(df["second_ball_win"])
    return get_set_piece_series(df).str.lower().isin(["second_ball_win","won_second_ball"]).astype(int)

def _get_outcome_success(df):
    """Returns 1/0 series based on outcome column."""
    if "outcome" in df.columns:
        return df["outcome"].astype(str).str.lower().str.strip().isin(
            ["successful", "success", "won", "win", "goal"]
        ).astype(int)
    return pd.Series([0]*len(df), index=df.index)

# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY TRAJECTORY CHARTS  (with zone labels)
# ─────────────────────────────────────────────────────────────────────────────
def _traj_chart(df, theme_name, flip_y, style_overrides, title, corner_side):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    figsize = (6.4, 8.4) if vert else (8.4, 6.4)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, corner_side, alpha=0.20, vertical=vert, show_labels=True)

    dd   = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    cmap = _cmap_delivery(s)

    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt, grp in dd.groupby("delivery_type"):
            dl = str(dt).lower()
            for _, r in grp.iterrows():
                rad = _curve_rad(dl, str(r.get("corner_label","unknown")))
                x1 = r["y_start_plot"] if vert else r["x_start_plot"]
                y1 = r["x_start_plot"] if vert else r["y_start_plot"]
                x2 = r["y2"]           if vert else r["x2"]
                y2 = r["x2"]           if vert else r["y2"]
                _arrow(ax, x1, y1, x2, y2, cmap.get(dl, s["accent"]), s, rad=rad)
        if s.get("show_legend", True):
            h, l = [], []
            for k, v in cmap.items():
                if (dd["delivery_type"].astype(str).str.lower() == k).any():
                    h.append(Line2D([0],[0], color=v, lw=s["trajectory_width"]+1))
                    l.append(k.title())
            if h:
                leg = ax.legend(h, l, frameon=True, loc="upper center",
                                bbox_to_anchor=(0.5,-0.03), ncol=3)
                style_legend(leg, s)
    else:
        for _, r in dd.iterrows():
            x1 = r["y_start_plot"] if vert else r["x_start_plot"]
            y1 = r["x_start_plot"] if vert else r["y_start_plot"]
            x2 = r["y2"]           if vert else r["x2"]
            y2 = r["x2"]           if vert else r["y2"]
            _arrow(ax, x1, y1, x2, y2, s["accent"], s, rad=0.0)

    chart_title(ax, title, s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_trajectories_left(df, theme_name, flip_y=False, style_overrides=None):
    return _traj_chart(df, theme_name, flip_y, style_overrides, "Delivery Trajectories — Left Corner", "left")

def chart_delivery_trajectories_right(df, theme_name, flip_y=False, style_overrides=None):
    return _traj_chart(df, theme_name, flip_y, style_overrides, "Delivery Trajectories — Right Corner", "right")

# ─────────────────────────────────────────────────────────────────────────────
# ATTACK FREE KICK TRAJECTORY MAP
# ─────────────────────────────────────────────────────────────────────────────
def chart_attack_freekick_trajectories(df, theme_name, flip_y=False, style_overrides=None):
    """
    Delivery trajectory arrows for attacking free kicks only.
    Filters: set_piece contains 'free kick' AND Type == 'Attack'.
    Arrow colour follows delivery_type colour-map identical to corner charts.
    """
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    # ── filter: free kicks only ───────────────────────────────────────────────
    for sp_col in ["set_piece_type", "set_piece", "Type"]:
        if sp_col in dff.columns:
            mask_fk = dff[sp_col].astype(str).str.lower().str.contains("free kick", na=False)
            if mask_fk.any():
                dff = dff[mask_fk].copy()
                break

    # filter for attacking type
    for att_col in ["Type", "attack_type", "type"]:
        if att_col in dff.columns:
            mask_att = dff[att_col].astype(str).str.lower().str.strip() == "attack"
            if mask_att.any():
                dff = dff[mask_att].copy()
                break

    figsize = (6.4, 8.4) if vert else (11.0, 7.0)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)

    dominant_side = _side_dominant(dff) if len(dff) else "right"
    # Show zone labels on free kick chart for spatial reference
    _draw_zones(ax, s, dominant_side, alpha=0.16, vertical=vert, show_labels=True)

    dd   = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()
    cmap = _cmap_delivery(s)

    if len(dd) == 0:
        ax.text(0.5, 0.5, "No attacking free kick\ndata found in this file",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=s["label_size"], color=s["muted"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor=s["bg"],
                          edgecolor=s["lines"], alpha=0.75))
        chart_title(ax, "Attack Free Kick Trajectories", s)
        if s["tight_layout"]: fig.tight_layout()
        return fig

    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt, grp in dd.groupby("delivery_type"):
            dl = str(dt).lower()
            for _, r in grp.iterrows():
                rad = _curve_rad(dl, str(r.get("corner_label", "unknown"))) * 0.6
                x1 = r["y_start_plot"] if vert else r["x_start_plot"]
                y1 = r["x_start_plot"] if vert else r["y_start_plot"]
                x2 = r["y2"]           if vert else r["x2"]
                y2 = r["x2"]           if vert else r["y2"]
                _arrow(ax, x1, y1, x2, y2, cmap.get(dl, s["accent"]), s, rad=rad)

        if s.get("show_legend", True):
            h, l = [], []
            for k, v in cmap.items():
                if (dd["delivery_type"].astype(str).str.lower() == k).any():
                    h.append(Line2D([0],[0], color=v, lw=s["trajectory_width"]+1))
                    l.append(k.title())
            if h:
                leg = ax.legend(h, l, frameon=True, loc="upper center",
                                bbox_to_anchor=(0.5,-0.03), ncol=min(len(h),5))
                style_legend(leg, s)
    else:
        for _, r in dd.iterrows():
            x1 = r["y_start_plot"] if vert else r["x_start_plot"]
            y1 = r["x_start_plot"] if vert else r["y_start_plot"]
            x2 = r["y2"]           if vert else r["x2"]
            y2 = r["x2"]           if vert else r["y2"]
            _arrow(ax, x1, y1, x2, y2, s["accent"], s, rad=0.0)

    # scatter end points
    base_color = s.get("scatter_dot_color", s["accent"])
    end_x = dd["y2"].values if vert else dd["x2"].values
    end_y = dd["x2"].values if vert else dd["y2"].values
    ax.scatter(end_x, end_y,
               s=max(s["marker_size"] * 0.9, 30),
               color=base_color,
               edgecolors=s["pitch_lines"],
               linewidths=s["marker_edge_width"],
               alpha=min(s["alpha"] * 0.85, 0.90),
               zorder=10, clip_on=False)

    # count badge
    bx_ = 5 if not vert else 2
    by_ = PW - 2 if not vert else PL - 2
    ax.text(bx_, by_, f"n = {len(dd)} free kicks",
            ha="left", va="top",
            fontsize=max(s["tick_size"] - 1, 7),
            color=s["muted"], zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=s["bg"],
                      edgecolor=s["lines"], alpha=0.65))

    chart_title(ax, "Attack Free Kick Trajectories", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY END SCATTER
# ─────────────────────────────────────────────────────────────────────────────
def _scatter_chart(df, theme_name, flip_y, style_overrides, corner_side):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    figsize = (6, 8) if vert else (10, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, corner_side, alpha=0.18, vertical=vert, show_labels=False)

    title = f"Delivery End Scatter — {'Left' if corner_side=='left' else 'Right'} Corner"
    dd    = dff.dropna(subset=["x2","y2"]).copy()

    if len(dd):
        if vert:
            px = dd["y2"].values; py = dd["x2"].values
        else:
            px = dd["x2"].values; py = dd["y2"].values

        base_color = s.get("scatter_dot_color", s["accent"])
        cmap       = _cmap_delivery(s)
        msz        = max(s["marker_size"] * 1.4, 60)
        mew        = s["marker_edge_width"]
        malpha     = s["alpha"]

        if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
            for dt, grp in dd.groupby("delivery_type"):
                gx = grp["y2"].values if vert else grp["x2"].values
                gy = grp["x2"].values if vert else grp["y2"].values
                c  = cmap.get(str(dt).lower(), base_color)
                ax.scatter(gx, gy, s=msz, color=c,
                           edgecolors=s["pitch_lines"], linewidths=mew,
                           alpha=malpha, zorder=9, clip_on=False,
                           label=str(dt).title())
            if s.get("show_legend", True):
                leg = ax.legend(frameon=True, loc="upper center",
                                bbox_to_anchor=(0.5,-0.03), ncol=3)
                style_legend(leg, s)
        else:
            ax.scatter(px, py, s=msz, color=base_color,
                       edgecolors=s["pitch_lines"], linewidths=mew,
                       alpha=malpha, zorder=9, clip_on=False)

        zones = _barca_zones(corner_side)
        for label, zx, zy, zw, zh in zones:
            mask = (dd["x2"] >= zx) & (dd["x2"] < zx+zw) & (dd["y2"] >= zy) & (dd["y2"] < zy+zh)
            cnt  = int(mask.sum())
            if cnt:
                cx_ = (zy+zh/2) if vert else (zx+zw/2)
                cy_ = (zx+zw/2) if vert else (zy+zh/2)
                ax.text(cx_, cy_, str(cnt), ha="center", va="center",
                        fontsize=max(s["tick_size"]+1, 10), fontweight="bold",
                        color=s["text"], zorder=11,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=s["bg"],
                                  edgecolor="none", alpha=0.65))

    chart_title(ax, title, s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_end_scatter_left(df, theme_name, flip_y=False, style_overrides=None):
    return _scatter_chart(df, theme_name, flip_y, style_overrides, "left")

def chart_delivery_end_scatter_right(df, theme_name, flip_y=False, style_overrides=None):
    return _scatter_chart(df, theme_name, flip_y, style_overrides, "right")

# ─────────────────────────────────────────────────────────────────────────────
# ZONE DELIVERY COUNT MAP
# ─────────────────────────────────────────────────────────────────────────────
def _zone_count_map(df, theme_name, flip_y, style_overrides, corner_side):
    s     = resolve_style(theme_name, style_overrides)
    vert  = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    figsize = (6, 7) if vert else (9, 6)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if not vert: ax.set_xlim(65, 103); ax.set_ylim(-2, 66)
    else:        ax.set_xlim(-2, 66);  ax.set_ylim(65, 103)

    dd    = dff.dropna(subset=["x2","y2"]).copy()
    total = max(len(dd), 1)
    zones = _barca_zones(corner_side)

    counts = {}
    for label, zx, zy, zw, zh in zones:
        mask = (dd["x2"] >= zx) & (dd["x2"] < zx+zw) & (dd["y2"] >= zy) & (dd["y2"] < zy+zh)
        counts[(label, zx, zy, zw, zh)] = int(mask.sum())

    max_cnt = max(counts.values()) if counts else 1

    for (label, zx, zy, zw, zh), ck in zip(zones, ZONE_CKEYS):
        cnt  = counts[(label, zx, zy, zw, zh)]
        inten = cnt / max(max_cnt, 1)
        color = s.get(ck, s["accent"])

        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh

        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=color,
                                edgecolor=s["pitch_lines"], linewidth=0.8,
                                alpha=0.10 + inten * 0.70, zorder=1))
        cx_ = rx + rw/2; cy_ = ry + rh/2
        pct = cnt / total * 100
        fs  = max(s["tick_size"] - 1, 7)
        ax.text(cx_, cy_ + rh*0.20, label.replace("\n"," "), ha="center", va="center",
                fontsize=max(fs-1, 5), color=s["muted"], alpha=0.85, zorder=3)
        ax.text(cx_, cy_ - rh*0.05, str(cnt), ha="center", va="center",
                fontsize=max(fs+3, 10), fontweight="bold", color=s["text"], zorder=4)
        ax.text(cx_, cy_ - rh*0.28, f"{pct:.0f}%", ha="center", va="center",
                fontsize=max(fs-1, 6), color=s["muted"], zorder=4)

    cols = ["players_near_post","players_far_post","players_6yard","players_penalty"]
    avg  = next((dff[c].mean() for c in cols if c in dff.columns and dff[c].notna().any()), None)
    if avg is None:
        in_box = dd[(dd["x2"] >= BOX_X0) & (dd["y2"] >= BOX_Y0) & (dd["y2"] <= BOX_Y1)]
        avg = round(len(in_box) / total * 5, 1)

    bx_, by_ = (32, 97) if vert else (60, 63.5)
    ax.add_patch(plt.Circle((bx_, by_), 2.5, facecolor=s["danger"],
                             edgecolor=s["pitch_lines"], linewidth=1, zorder=5))
    ax.text(bx_, by_, f"{avg:.1f}", ha="center", va="center",
            fontsize=max(s["tick_size"], 9), fontweight="bold", color="white", zorder=6)
    lby = 96 if vert else 60.5
    ax.text(bx_, lby, "Avg. players\nin box", ha="center", va="top",
            fontsize=max(s["tick_size"]-2, 6), color=s["muted"], zorder=6)

    lbl = "Right Side Corners" if corner_side == "right" else "Left Side Corners"
    chart_title(ax, lbl, s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_zone_count_left(df, theme_name, flip_y=False, style_overrides=None):  return _zone_count_map(df, theme_name, flip_y, style_overrides, "left")
def chart_zone_count_right(df, theme_name, flip_y=False, style_overrides=None): return _zone_count_map(df, theme_name, flip_y, style_overrides, "right")

# ─────────────────────────────────────────────────────────────────────────────
# PITCH SETUP HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _pitch_setup(df, theme_name, flip_y, style_overrides, figsize_h=(8,6), figsize_v=(6,8)):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)
    fig, ax = _base_fig(s, figsize_v if vert else figsize_h)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.16, vertical=vert, show_labels=False)
    return s, vert, pitch, dff, fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING PITCH CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_delivery_start_map(df, theme_name, flip_y=False, style_overrides=None):
    s, vert, pitch, dff, fig, ax = _pitch_setup(df, theme_name, flip_y, style_overrides)
    dd = dff.dropna(subset=["x_start_plot","y_start_plot"]).copy()
    px = dd["y_start_plot"].values if vert else dd["x_start_plot"].values
    py = dd["x_start_plot"].values if vert else dd["y_start_plot"].values
    ax.scatter(px, py, s=s["marker_size"], color=s["accent"],
               edgecolors=s["pitch_lines"], linewidths=s["marker_edge_width"],
               alpha=s["alpha"], zorder=7, clip_on=False)
    chart_title(ax, "Delivery Start Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_heatmap(df, theme_name, flip_y=False, style_overrides=None):
    s, vert, pitch, dff, fig, ax = _pitch_setup(df, theme_name, flip_y, style_overrides)
    dd = dff.dropna(subset=["x2","y2"]).copy()
    if len(dd):
        px = dd["y2"].values if vert else dd["x2"].values
        py = dd["x2"].values if vert else dd["y2"].values
        try:
            pitch.kdeplot(px, py, ax=ax, fill=True, levels=50,
                          alpha=s["kde_alpha"], cmap=s["heatmap_cmap"])
        except Exception:
            ax.scatter(px, py, s=s["marker_size"]*0.65, color=s["accent"],
                       alpha=s["alpha"], zorder=7, clip_on=False)
    chart_title(ax, "Delivery End Location Heatmap", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_average_delivery_path(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)
    fig, ax = _base_fig(s, (6.2,8.2) if vert else (8.2,6.2))
    pitch.draw(ax=ax); _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.16, vertical=vert, show_labels=False)
    dd   = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    cmap = _cmap_delivery(s)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt, grp in dd.groupby("delivery_type"):
            if not len(grp): continue
            ax1,ay1 = grp["x_start_plot"].mean(), grp["y_start_plot"].mean()
            ax2,ay2 = grp["x2"].mean(), grp["y2"].mean()
            mc = grp["corner_label"].mode().iloc[0] if grp["corner_label"].notna().any() else "unknown"
            rad = _curve_rad(str(dt).lower(), str(mc))
            x1p = ay1 if vert else ax1; y1p = ax1 if vert else ay1
            x2p = ay2 if vert else ax2; y2p = ax2 if vert else ay2
            c   = cmap.get(str(dt).lower(), s["accent"])
            _arrow(ax, x1p, y1p, x2p, y2p, c, s, rad=rad, lm=2.2, am=1.15)
            ax.scatter([x2p],[y2p], s=s["marker_size"]*1.2, color=c,
                       edgecolors=s["pitch_lines"], linewidths=1.2, zorder=8, clip_on=False)
        if s.get("show_legend", True):
            h, l = [], []
            for k, v in cmap.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    h.append(Line2D([0],[0],color=v,lw=s["trajectory_width"]*2)); l.append(f"{k.title()} Avg")
            if h:
                leg = ax.legend(h, l, frameon=True, loc="upper center",
                                bbox_to_anchor=(0.5,-0.03), ncol=3)
                style_legend(leg, s)
    else:
        ax1,ay1=dd["x_start_plot"].mean(),dd["y_start_plot"].mean()
        ax2,ay2=dd["x2"].mean(),dd["y2"].mean()
        x1p=ay1 if vert else ax1; y1p=ax1 if vert else ay1
        x2p=ay2 if vert else ax2; y2p=ax2 if vert else ay2
        _arrow(ax, x1p, y1p, x2p, y2p, s["accent"], s, rad=0.0, lm=2.2)
    chart_title(ax, "Average Delivery Path", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_heat_plus_trajectories(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)
    fig, ax = _base_fig(s, (6.4,8.4) if vert else (8.4,6.4))
    pitch.draw(ax=ax); _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)
    dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if len(dd):
        px=dd["y2"].values if vert else dd["x2"].values
        py=dd["x2"].values if vert else dd["y2"].values
        try: pitch.kdeplot(px, py, ax=ax, fill=True, levels=40, alpha=s["kde_alpha"]*0.65, cmap=s["heatmap_cmap"])
        except: pass
    cmap = _cmap_delivery(s)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt, grp in dd.groupby("delivery_type"):
            dl = str(dt).lower()
            for _,r in grp.iterrows():
                rad=_curve_rad(dl,str(r.get("corner_label","unknown")))
                x1=r["y_start_plot"] if vert else r["x_start_plot"]
                y1=r["x_start_plot"] if vert else r["y_start_plot"]
                x2=r["y2"]           if vert else r["x2"]
                y2=r["x2"]           if vert else r["y2"]
                _arrow(ax, x1, y1, x2, y2, cmap.get(dl,s["accent"]), s, rad=rad, lm=0.9, am=0.9)
        if s.get("show_legend", True):
            h, l = [], []
            for k, v in cmap.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    h.append(Line2D([0],[0],color=v,lw=s["trajectory_width"]+0.6)); l.append(k.title())
            if h:
                leg = ax.legend(h, l, frameon=True, loc="upper center",
                                bbox_to_anchor=(0.5,-0.03), ncol=3)
                style_legend(leg, s)
    chart_title(ax, "Heat + Trajectories", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_trajectory_clusters(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)
    fig, ax = _base_fig(s, (6.4,8.4) if vert else (8.4,6.4))
    pitch.draw(ax=ax); _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)
    dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if not len(dd):
        chart_title(ax, "Trajectory Clusters", s); fig.tight_layout(); return fig
    feats = dd[["x_start_plot","y_start_plot","x2","y2"]].dropna()
    if KMeans and len(feats) >= 3:
        try:
            n = min(3, len(feats))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            dd.loc[feats.index, "cluster"] = km.fit_predict(feats)
        except: dd["cluster"] = 0
    else: dd["cluster"] = 0
    pal = [s["accent"],s["warning"],s["success"],s["accent_2"],s["danger"]]
    h, l = [], []
    for i, (cid, grp) in enumerate(dd.groupby("cluster")):
        color = pal[i % len(pal)]
        for _,r in grp.iterrows():
            rad=_curve_rad(str(r.get("delivery_type","")).lower(),str(r.get("corner_label","unknown")))
            x1=r["y_start_plot"] if vert else r["x_start_plot"]
            y1=r["x_start_plot"] if vert else r["y_start_plot"]
            x2=r["y2"]           if vert else r["x2"]
            y2=r["x2"]           if vert else r["y2"]
            _arrow(ax, x1, y1, x2, y2, color, s, rad=rad, lm=0.85, am=0.65)
        mc=grp["corner_label"].mode().iloc[0] if grp["corner_label"].notna().any() else "unknown"
        md=grp["delivery_type"].mode().iloc[0] if ("delivery_type" in grp.columns and grp["delivery_type"].notna().any()) else ""
        ar=_curve_rad(str(md).lower(), str(mc))
        gx1=grp["y_start_plot"].mean() if vert else grp["x_start_plot"].mean()
        gy1=grp["x_start_plot"].mean() if vert else grp["y_start_plot"].mean()
        gx2=grp["y2"].mean() if vert else grp["x2"].mean()
        gy2=grp["x2"].mean() if vert else grp["y2"].mean()
        _arrow(ax, gx1, gy1, gx2, gy2, color, s, rad=ar, lm=2.5, am=1.1)
        h.append(Line2D([0],[0],color=color,lw=s["trajectory_width"]*2))
        l.append(f"Cluster {int(cid)+1} ({len(grp)})")
    if s.get("show_legend", True) and h:
        leg = ax.legend(h, l, frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5,-0.03), ncol=3)
        style_legend(leg, s)
    chart_title(ax, "Trajectory Clusters", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_shot_map(df, theme_name, flip_y=False, style_overrides=None):
    s, vert, pitch, dff, fig, ax = _pitch_setup(df, theme_name, flip_y, style_overrides)
    dd = dff.dropna(subset=["x","y"]).copy()
    sz = s["marker_size"]*1.6
    if "xg" in dd.columns:
        xg = pd.to_numeric(dd["xg"],errors="coerce").fillna(0); sz = 35+xg*500
    px = dd["y"].values if vert else dd["x"].values
    py = dd["x"].values if vert else dd["y"].values
    ax.scatter(px, py, s=sz, color=s["success"], edgecolors=s["pitch_lines"],
               linewidths=s["marker_edge_width"], alpha=s["alpha"], zorder=7, clip_on=False)
    chart_title(ax, "Set Piece Shot / Event Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_second_ball_map(df, theme_name, flip_y=False, style_overrides=None):
    s, vert, pitch, dff, fig, ax = _pitch_setup(df, theme_name, flip_y, style_overrides)
    dd = dff.dropna(subset=["x","y"]).copy()
    dd["sb"] = _get_sbw(dd)
    win = dd[dd["sb"]==1]; lose = dd[dd["sb"]==0]
    if len(win):
        px=win["y"].values if vert else win["x"].values
        py=win["x"].values if vert else win["y"].values
        ax.scatter(px,py,s=s["marker_size"]*1.25,color=s["success"],
                   edgecolors=s["pitch_lines"],linewidths=s["marker_edge_width"],
                   alpha=s["alpha"],zorder=7,clip_on=False,label="Won")
    if len(lose):
        px=lose["y"].values if vert else lose["x"].values
        py=lose["x"].values if vert else lose["y"].values
        ax.scatter(px,py,s=s["marker_size"]*1.25,facecolors="none",
                   edgecolors=s["danger"],linewidths=s["line_width"]+0.6,
                   alpha=s["alpha"],zorder=7,clip_on=False,label="Lost")
    if (len(win) or len(lose)) and s.get("show_legend",True):
        leg = ax.legend(frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=2)
        style_legend(leg, s)
    chart_title(ax, "Second Ball Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_defensive_vulnerability_map(df, theme_name, flip_y=False, style_overrides=None):
    s, vert, pitch, dff, fig, ax = _pitch_setup(df, theme_name, flip_y, style_overrides)
    dd = dff.dropna(subset=["x","y"]).copy()
    if len(dd):
        px=dd["y"].values if vert else dd["x"].values
        py=dd["x"].values if vert else dd["y"].values
        try: pitch.kdeplot(px,py,ax=ax,fill=True,levels=40,alpha=s["kde_alpha"],cmap=s["heatmap_cmap"])
        except: ax.scatter(px,py,s=s["marker_size"]*0.8,color=s["danger"],edgecolors=s["pitch_lines"],linewidths=s["marker_edge_width"],alpha=s["alpha"],zorder=7,clip_on=False)
    chart_title(ax, "Defensive Vulnerability Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_set_piece_landing_heatmap(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)
    fig, ax = _base_fig(s, (6.2,8.8) if vert else (10.0,6.8))
    pitch.draw(ax=ax); _setup_pitch_axes(ax, s, vert)
    _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.14, vertical=vert, show_labels=False)
    dd = dff.dropna(subset=["x2","y2"]).copy()
    if len(dd):
        px=dd["y2"].values if vert else dd["x2"].values
        py=dd["x2"].values if vert else dd["y2"].values
        try: pitch.kdeplot(px,py,ax=ax,fill=True,levels=60,alpha=s["kde_alpha"],cmap=s["heatmap_cmap"],zorder=4)
        except: pass
        ax.scatter(px, py, s=max(s["marker_size"]*0.4, 15), color=s["text"],
                   edgecolors="none", alpha=min(s["alpha"]*0.6, 0.65),
                   zorder=8, clip_on=False)
    chart_title(ax, "Set Piece Landing Heatmap", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ── bar charts ──────────────────────────────────────────────────────────────
def chart_delivery_length_distribution(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.6,4.8))
    dff = _prep(df, flip_y); dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    lengths = (((dd["x2"]-dd["x_start_plot"])**2+(dd["y2"]-dd["y_start_plot"])**2)**0.5 if len(dd) else pd.Series(dtype=float))
    bc = s.get("bar_colors",{}).get("default", s["accent"])
    ax.hist(lengths, bins=12, color=bc, edgecolor=s["lines"], linewidth=0.8, alpha=0.92)
    themed_bar(ax, s); chart_title(ax, "Delivery Length Distribution", s)
    ax.set_xlabel("Length"); ax.set_ylabel("Count")
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_direction_map(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.6,4.8))
    dff = _prep(df, flip_y); dd = dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    if not len(dd): summary = pd.Series(dtype=float)
    else:
        dx=dd["x2"]-dd["x_start_plot"]; dy=dd["y2"]-dd["y_start_plot"]
        angles=dy.combine(dx, lambda yv,xv: math.degrees(math.atan2(yv,xv)))
        lbls=pd.cut(angles,bins=[-181,-60,-10,10,60,181],labels=["Down","Down-In","Straight","Up-In","Up"],include_lowest=True)
        summary=lbls.value_counts().reindex(["Down","Down-In","Straight","Up-In","Up"]).fillna(0)
    bc = s.get("bar_colors",{}).get("default", s["accent_2"])
    ax.bar(summary.index.astype(str), summary.values, color=bc, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Delivery Direction Map", s); ax.set_ylabel("Count")
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_outcome_distribution(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.4,4.6))
    counts = get_set_piece_series(df).str.lower().value_counts()
    bcm = s.get("bar_colors",{})
    colors = [bcm.get("success",s["success"]) if x in ["successful","corner","free_kick"]
              else bcm.get("danger",s["danger"]) if x in ["unsuccessful","failed","loss"]
              else bcm.get("default",s["accent"]) for x in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Outcome Distribution", s)
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=25)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_target_zone_breakdown(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.4,4.6))
    dff = _prep(df, flip_y); counts = get_target_zone_series(dff).value_counts()
    bc = s.get("bar_colors",{}).get("default", s["accent"])
    ax.bar(counts.index, counts.values, color=bc, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Target Zone Breakdown", s)
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=25)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_first_contact_win_by_zone(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.6,4.8))
    dff = _prep(df, flip_y); dd = dff.copy()
    dd["zone_calc"] = get_target_zone_series(dd); dd["fc_calc"] = _get_fcw(dd)
    summary = dd.groupby("zone_calc",dropna=False)["fc_calc"].mean().sort_values(ascending=False)*100
    bc = s.get("bar_colors",{}).get("default", s["accent"])
    ax.bar(summary.index.astype(str), summary.values, color=bc, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "First Contact Win % By Zone", s)
    ax.set_ylabel("Win %"); ax.tick_params(axis="x", rotation=25)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_routine_breakdown(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.6,4.8))
    if "routine_type" in df.columns and df["routine_type"].notna().any():
        counts = df["routine_type"].fillna("unclassified").value_counts().head(10)
    else: counts = get_target_zone_series(df).value_counts().head(10)
    bc = s.get("bar_colors",{}).get("default", s["warning"])
    ax.barh(counts.index[::-1], counts.values[::-1], color=bc, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Routine Breakdown", s); ax.set_xlabel("Count")
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_taker_profile(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.8,4.8))
    if "taker" in df.columns and "sequence_id" in df.columns:
        seq = df.groupby("taker")["sequence_id"].nunique().sort_values(ascending=False).head(10)
    elif "taker" in df.columns: seq = df["taker"].value_counts().head(10)
    else: seq = get_set_piece_series(df).value_counts().head(10)
    bc = s.get("bar_colors",{}).get("default", s["accent"])
    ax.barh(seq.index[::-1], seq.values[::-1], color=bc, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Taker / Event Profile", s); ax.set_xlabel("Count")
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_structure_zone_averages(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides); fig, ax = _base_fig(s, (7.6,4.8))
    cols = ["players_near_post","players_far_post","players_6yard","players_penalty"]
    existing = [c for c in cols if c in df.columns]
    if existing:
        means = df[existing].apply(pd.to_numeric,errors="coerce").mean().fillna(0)
        lmap = {"players_near_post":"Near Post","players_far_post":"Far Post","players_6yard":"6 Yard","players_penalty":"Penalty"}
        lbls = [lmap[c] for c in existing]; vals = means.values
    else:
        vc = get_target_zone_series(df).value_counts(); lbls = vc.index.tolist(); vals = vc.values
    bc = s.get("bar_colors",{}).get("default")
    colors = [bc]*len(vals) if bc else [s["accent"],s["warning"],s["success"],s["accent_2"]][:len(vals)]
    ax.bar(lbls, vals, color=colors, edgecolor=s["lines"], linewidth=0.8)
    themed_bar(ax, s); chart_title(ax, "Structure / Zone Summary", s)
    ax.set_ylabel("Value"); ax.tick_params(axis="x", rotation=15)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_taker_stats_table(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides)
    if "taker" not in df.columns:
        fig,ax=_base_fig(s,(9,4)); ax.text(0.5,0.5,"No 'taker' column",ha="center",va="center",color=s["text"],fontsize=12,transform=ax.transAxes); return fig
    rows=[]
    for taker,grp in df.groupby("taker"):
        if str(taker).lower() in ("nan","none",""): continue
        ns=int(grp["sequence_id"].nunique()) if "sequence_id" in grp.columns else len(grp)
        ni=int((grp["delivery_type"].astype(str).str.lower()=="inswing").sum()) if "delivery_type" in grp.columns else 0
        no=int((grp["delivery_type"].astype(str).str.lower()=="outswing").sum()) if "delivery_type" in grp.columns else 0
        nsu=int((grp["outcome"].astype(str).str.lower().isin(["successful","success","won"])).sum()) if "outcome" in grp.columns else 0
        sr=round(nsu/max(len(grp),1)*100,1)
        nl=int((grp["side"].astype(str).str.lower()=="left").sum()) if "side" in grp.columns else 0
        nr=int((grp["side"].astype(str).str.lower()=="right").sum()) if "side" in grp.columns else 0
        try:
            taker_num = str(int(float(str(taker))))
        except Exception:
            taker_num = str(taker).strip()
        rows.append({"taker":taker_num,"sequences":ns,"inswing":ni,"outswing":no,"left":nl,"right":nr,"success_rate":sr,"taker_num":taker_num})
    if not rows:
        fig,ax=_base_fig(s,(9,4)); ax.text(0.5,0.5,"No taker data",ha="center",va="center",color=s["text"],fontsize=12,transform=ax.transAxes); return fig
    sdf=pd.DataFrame(rows).sort_values("sequences",ascending=False).head(12).reset_index(drop=True)
    n=len(sdf); row_h=0.72; hh=1.0; fw=9.0; fh=hh+n*row_h+0.4
    apply_rcparams(s)
    fig=plt.figure(figsize=(fw,fh)); fig.patch.set_facecolor(s["bg"])
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,fw); ax.set_ylim(0,fh)
    ax.set_facecolor(s["bg"]); ax.axis("off")
    bc_s=s.get("shirt_body_color",s["accent"]); sl_c=s.get("shirt_sleeve_color",s["panel"]); nm_c=s.get("shirt_number_color",s["bg"])
    mw=2.0
    ax.text(fw/2,fh-0.28,"Set Piece Taker Stats",ha="center",va="top",fontsize=s["title_size"]+2,fontweight="bold",color=s["text"])
    ax.text(fw/2,fh-0.62,"Sorted by number of set piece sequences taken",ha="center",va="top",fontsize=s["tick_size"],color=s["muted"])
    cx={"shirt":0.45,"seq":1.55,"ins":2.75,"out":3.80,"right":4.80,"left":5.75,"rate":7.30}
    hy=fh-hh+0.05
    for k,lbl in {"seq":"SEQ","ins":"INSWING","out":"OUTSWING","right":"RIGHT","left":"LEFT","rate":"SUCCESS %"}.items():
        ax.text(cx[k],hy,lbl,ha="center",va="bottom",fontsize=s["tick_size"]-1,color=s["muted"],fontweight="bold")
    ax.axhline(hy-0.02,xmin=0.02,xmax=0.98,color=s["lines"],linewidth=0.8,alpha=0.6)
    for i,row in sdf.iterrows():
        yc=fh-hh-(i+0.5)*row_h
        if i%2==0: ax.add_patch(Rectangle((0.1,yc-row_h/2+0.04),fw-0.2,row_h-0.08,facecolor=s["panel"],edgecolor="none",alpha=0.35,zorder=0))
        sz=row_h*0.78; bx=cx["shirt"]-sz*0.55/2; by=yc-sz*0.65/2-sz*0.04
        ax.add_patch(FancyBboxPatch((bx,by),sz*0.55,sz*0.65,boxstyle="round,pad=0.01",facecolor=bc_s,edgecolor=s["lines"],linewidth=0.6,zorder=4))
        ax.add_patch(FancyBboxPatch((bx-sz*0.22,by+sz*0.65*0.62),sz*0.22,sz*0.28*0.7,boxstyle="round,pad=0.01",facecolor=sl_c,edgecolor=s["lines"],linewidth=0.5,zorder=4))
        ax.add_patch(FancyBboxPatch((bx+sz*0.55,by+sz*0.65*0.62),sz*0.22,sz*0.28*0.7,boxstyle="round,pad=0.01",facecolor=sl_c,edgecolor=s["lines"],linewidth=0.5,zorder=4))
        ax.add_patch(plt.Circle((cx["shirt"],by+sz*0.65-sz*0.09*0.3),sz*0.09,facecolor=sl_c,edgecolor=s["lines"],linewidth=0.5,zorder=5))
        ax.text(cx["shirt"],by+sz*0.65*0.38,str(row["taker_num"]),ha="center",va="center",fontsize=max(s["tick_size"]-1,7),fontweight="bold",color=nm_c,zorder=6)
        for k in ["seq","ins","out","right","left"]:
            v={"seq":row["sequences"],"ins":row["inswing"],"out":row["outswing"],"right":row["right"],"left":row["left"]}[k]
            ax.text(cx[k],yc,str(v),ha="center",va="center",fontsize=s["tick_size"],color=s["text"])
        rate=row["success_rate"]/100.0; bx_=cx["rate"]-mw/2; bh=row_h*0.30
        ax.add_patch(Rectangle((bx_,yc-bh/2),mw,bh,facecolor=s["lines"],edgecolor="none",alpha=0.35,zorder=1))
        bfc=s.get("bar_colors",{}).get("default",s["accent"])
        ax.add_patch(Rectangle((bx_,yc-bh/2),mw*rate,bh,facecolor=bfc,edgecolor="none",alpha=0.90,zorder=2))
        ax.text(bx_+mw+0.10,yc,f"{row['success_rate']:.1f}%",ha="left",va="center",fontsize=s["tick_size"]-1,color=s["text"])
        ax.axhline(yc-row_h/2+0.04,xmin=0.02,xmax=0.98,color=s["lines"],linewidth=0.4,alpha=0.3)
    if s["tight_layout"]: fig.tight_layout(pad=0.3)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# AVG PLAYERS PER ZONE MAP
# ─────────────────────────────────────────────────────────────────────────────
_ZONE_PLAYER_COLS = {
    "Near\nPost":    ["players_near_post"],
    "Far\nPost":     ["players_far_post"],
    "Small\nArea":   ["players_small_area", "players_6yard"],
    "Penalty\nSpot": ["players_penalty", "players_penalty_area"],
}

def _get_zone_avg(df_side, col_candidates):
    for c in col_candidates:
        if c in df_side.columns:
            v = pd.to_numeric(df_side[c], errors="coerce").dropna()
            if len(v): return round(float(v.mean()), 1)
    return 0.0

def _get_avg_in_box(df_side):
    found_cols = []
    for candidates in _ZONE_PLAYER_COLS.values():
        for c in candidates:
            if c in df_side.columns:
                found_cols.append(c); break
    if not found_cols: return 0.0
    numeric = df_side[found_cols].apply(pd.to_numeric, errors="coerce")
    row_sums = numeric.dropna(how="all").fillna(0).sum(axis=1)
    return round(float(row_sums.mean()), 1) if len(row_sums) else 0.0

def _draw_gaussian_blob(ax, cx, cy, radius, peak_alpha, color="#ff2200"):
    steps = 10
    for i in range(steps, 0, -1):
        r = radius * i / steps
        a = peak_alpha * ((steps - i + 1) / steps) ** 2.2
        ax.add_patch(plt.Circle((cx, cy), r, facecolor=color,
                                 edgecolor="none", alpha=a, zorder=3))

def _avg_players_zone_map(df, theme_name, flip_y, style_overrides, corner_side):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    zone_avgs = {label: _get_zone_avg(dff, cols)
                 for label, cols in _ZONE_PLAYER_COLS.items()}
    avg_in_box = _get_avg_in_box(dff)
    max_avg    = max(zone_avgs.values(), default=1.0)
    max_avg    = max(max_avg, 0.01)

    figsize = (6, 5.5) if vert else (8, 5.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if not vert:
        ax.set_xlim(68, 103); ax.set_ylim(-1, 65)
    else:
        ax.set_xlim(-1, 65);  ax.set_ylim(68, 103)

    _RED_LOW  = np.array(mpl.colors.to_rgb("#1c0404"))
    _RED_HIGH = np.array(mpl.colors.to_rgb("#ff1a00"))

    zones = _barca_zones(corner_side)

    for (label, zx, zy, zw, zh) in zones:
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh

        avg_v = zone_avgs.get(label)
        if avg_v is not None:
            t     = avg_v / max_avg
            rgb   = _RED_LOW * (1 - t) + _RED_HIGH * t
            alpha = 0.30 + t * 0.58
        else:
            rgb   = np.array([0.07, 0.07, 0.07])
            alpha = 0.22

        ax.add_patch(Rectangle((rx, ry), rw, rh,
                                facecolor=rgb, edgecolor=s["pitch_lines"],
                                linewidth=0.8, alpha=alpha, zorder=2))

    max_label = max(zone_avgs, key=zone_avgs.get) if zone_avgs else None
    if max_label and zone_avgs.get(max_label, 0) > 0:
        for (label, zx, zy, zw, zh) in zones:
            if label == max_label:
                cx_z = zx + zw / 2; cy_z = zy + zh / 2
                if vert: cx_p, cy_p = cy_z, cx_z
                else:    cx_p, cy_p = cx_z, cy_z
                blob_r = min(zw, zh) * 0.70
                peak_a = 0.55 * (zone_avgs[max_label] / max_avg)
                _draw_gaussian_blob(ax, cx_p, cy_p, blob_r, peak_a)
                break

    sorted_zones = sorted(zone_avgs.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_zones) >= 2 and sorted_zones[1][1] > 0:
        sec_label = sorted_zones[1][0]
        for (label, zx, zy, zw, zh) in zones:
            if label == sec_label:
                cx_z = zx + zw / 2; cy_z = zy + zh / 2
                if vert: cx_p, cy_p = cy_z, cx_z
                else:    cx_p, cy_p = cx_z, cy_z
                blob_r = min(zw, zh) * 0.50
                peak_a = 0.35 * (sorted_zones[1][1] / max_avg)
                _draw_gaussian_blob(ax, cx_p, cy_p, blob_r, peak_a)
                break

    fs_num  = max(s["title_size"] - 1, 13)
    fs_lbl  = max(s["tick_size"] - 2, 6)

    for (label, zx, zy, zw, zh) in zones:
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh
        cx_ = rx + rw / 2
        cy_ = ry + rh / 2

        avg_v = zone_avgs.get(label)
        if avg_v is not None:
            ax.text(cx_, cy_, f"{avg_v:.1f}",
                    ha="center", va="center",
                    fontsize=fs_num, fontweight="bold",
                    color="white", zorder=9,
                    path_effects=[mpl_pe.withStroke(linewidth=2.5, foreground="black")])
            ax.text(cx_, ry + rh * 0.10, label.replace("\n", " "),
                    ha="center", va="bottom",
                    fontsize=fs_lbl, color=s["muted"], alpha=0.90, zorder=9)
        else:
            ax.text(cx_, ry + rh * 0.88, label.replace("\n", " "),
                    ha="center", va="top",
                    fontsize=fs_lbl, color=s["muted"], alpha=0.65, zorder=9)

    box_front_cx = (72.0 + BOX_X0) / 2.0
    arc_bottom_y  = 32.0 - 9.15
    if not vert:
        bx_ = box_front_cx + 5
        by_ = arc_bottom_y - 1.5
    else:
        bx_ = arc_bottom_y + 9.5
        by_ = box_front_cx - 2

    ax.add_patch(plt.Circle((bx_, by_), 2.5,
                             facecolor="#ff3b30", edgecolor="white",
                             linewidth=1.4, zorder=20, clip_on=False))
    ax.text(bx_, by_, f"{avg_in_box:.1f}",
            ha="center", va="center",
            fontsize=max(s["tick_size"] + 1, 10), fontweight="bold",
            color="white", zorder=21, clip_on=False)
    if not vert:
        ax.text(bx_, by_ - 3.2, "Avg. players\nin box",
                ha="center", va="top",
                fontsize=max(s["tick_size"] - 2, 6),
                color=s["muted"], zorder=21, clip_on=False)
    else:
        ax.text(bx_ - 3.5, by_, "Avg. players\nin box",
                ha="right", va="center",
                fontsize=max(s["tick_size"] - 2, 6),
                color=s["muted"], zorder=21, clip_on=False)

    side_lbl = f"{'Left' if corner_side=='left' else 'Right'} Side Corners — Avg Players per Zone"
    chart_title(ax, side_lbl, s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_avg_players_left(df, theme_name, flip_y=False, style_overrides=None):
    return _avg_players_zone_map(df, theme_name, flip_y, style_overrides, "left")

def chart_avg_players_right(df, theme_name, flip_y=False, style_overrides=None):
    return _avg_players_zone_map(df, theme_name, flip_y, style_overrides, "right")

# ─────────────────────────────────────────────────────────────────────────────
# FIRST CONTACT LOCATION MAP
# ─────────────────────────────────────────────────────────────────────────────
_ACTION_CONFIG = {
    "shot":        ("*",  "#FFD400", "Shot"),
    "clearance":   ("v",  "#FF4D4D", "Clearance"),
    "cleareance":  ("v",  "#FF4D4D", "Clearance"),
    "header":      ("^",  "#38BDF8", "Header"),
    "threat":      ("D",  "#A78BFA", "Threat"),
    "cross":       ("P",  "#22C55E", "Cross"),
    "foul":        ("s",  "#F97316", "Foul"),
    "other":       ("o",  "#94A3B8", "Other"),
}

def _normalise_action(val: str) -> str:
    v = str(val).strip().lower()
    if v in ("clearance","cleareance","cleared"): return "clearance"
    if v in ("shot","shots","shoot"): return "shot"
    if v in ("header","head","headed"): return "header"
    if v in ("threat","dangerous","chance"): return "threat"
    if v in ("cross","crossed"): return "cross"
    if v in ("foul","fouled","foul won"): return "foul"
    return v if v in _ACTION_CONFIG else "other"

def chart_first_contact_map(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6, 8.5) if vert else (10, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)

    dd = dff.dropna(subset=["x2", "y2"]).copy()

    action_col = None
    for c in ["result", "action", "event", "first_contact_result"]:
        if c in dd.columns and dd[c].notna().any():
            action_col = c; break

    dd["_fcw"] = _get_fcw(dd)

    user_colours  = s.get("action_colors", {})
    visible_types = s.get("action_types_visible", None)

    msz   = max(s["marker_size"] * 1.6, 80)
    mew   = s["marker_edge_width"] + 0.5

    if action_col and dd[action_col].notna().any():
        dd["_action"] = dd[action_col].apply(_normalise_action)
        if visible_types:
            dd = dd[dd["_action"].isin(visible_types)].copy()

        for act, grp in dd.groupby("_action"):
            cfg = _ACTION_CONFIG.get(act, ("o", "#94A3B8", act.title()))
            marker_, base_color, label_ = cfg
            color_ = user_colours.get(act, base_color)
            px = grp["y2"].values if vert else grp["x2"].values
            py = grp["x2"].values if vert else grp["y2"].values
            for xi, yi, fcw in zip(px, py, grp["_fcw"].values):
                edge_c = "#FFD400" if fcw == 1 else s["lines"]
                ax.scatter(xi, yi, s=msz, marker=marker_, color=color_,
                           edgecolors=edge_c, linewidths=mew,
                           alpha=s["alpha"], zorder=9, clip_on=False)
    else:
        px = dd["y2"].values if vert else dd["x2"].values
        py = dd["x2"].values if vert else dd["y2"].values
        ax.scatter(px, py, s=msz, color=s.get("scatter_dot_color", s["accent"]),
                   edgecolors=s["lines"], linewidths=mew,
                   alpha=s["alpha"], zorder=9, clip_on=False)

    if s.get("show_legend", True) and action_col and len(dd):
        present_actions = dd["_action"].unique() if "_action" in dd.columns else []
        handles, labels = [], []
        for act in present_actions:
            cfg = _ACTION_CONFIG.get(act, ("o", "#94A3B8", act.title()))
            marker_, base_color, label_ = cfg
            color_ = user_colours.get(act, base_color)
            handles.append(mpl.lines.Line2D([0],[0], marker=marker_, color="none",
                                             markerfacecolor=color_,
                                             markeredgecolor=s["lines"],
                                             markeredgewidth=0.8,
                                             markersize=9, label=label_))
            labels.append(label_)
        handles.append(mpl.lines.Line2D([0],[0], marker="o", color="none",
                                         markerfacecolor="none",
                                         markeredgecolor="#FFD400",
                                         markeredgewidth=2.2,
                                         markersize=9, label="FC Won (gold edge)"))
        labels.append("FC Won (gold edge)")
        leg = ax.legend(handles, labels, frameon=True,
                        loc="upper center", bbox_to_anchor=(0.5, -0.04),
                        ncol=min(4, len(labels)))
        style_legend(leg, s)

    chart_title(ax, "First Contact Location Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# ██████████████████████   DEFENSIVE CHARTS   ██████████████████████████████
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFENSIVE SHAPE MAP
# Shows HOW the defence is set up by plotting attacker end-positions (x2/y2)
# and overlaying defender counts per zone as big shaded badges.
# Insight: which zones are over/under-loaded by defenders vs attackers?
# ─────────────────────────────────────────────────────────────────────────────
def chart_defensive_shape_map(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6, 7.5) if vert else (9.5, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)
    if not vert: ax.set_xlim(65, 103); ax.set_ylim(-2, 66)
    else:        ax.set_xlim(-2, 66);  ax.set_ylim(65, 103)

    dominant_side = _side_dominant(dff)
    zones = _barca_zones(dominant_side)
    dd = dff.dropna(subset=["x2","y2"]).copy()

    # defender column candidates (near/far post)
    def_cols_near = ["defenders_near_post"]
    def_cols_far  = ["defenders_far_post"]
    has_def = any(c in dff.columns for c in def_cols_near + def_cols_far)

    # attacker counts per zone
    att_counts = {}
    for label, zx, zy, zw, zh in zones:
        mask = (dd["x2"] >= zx) & (dd["x2"] < zx+zw) & (dd["y2"] >= zy) & (dd["y2"] < zy+zh)
        att_counts[label] = int(mask.sum())
    max_att = max(att_counts.values(), default=1)

    for (label, zx, zy, zw, zh), ck in zip(zones, ZONE_CKEYS):
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh
        cnt   = att_counts.get(label, 0)
        inten = cnt / max(max_att, 1)
        # attacker pressure = warm colour
        ax.add_patch(Rectangle((rx, ry), rw, rh,
                                facecolor=s["danger"],
                                edgecolor=s["pitch_lines"], linewidth=0.8,
                                alpha=0.08 + inten * 0.55, zorder=1))
        cx_ = rx + rw/2; cy_ = ry + rh/2
        fs  = max(s["tick_size"] - 1, 7)
        ax.text(cx_, cy_ + rh*0.18, label.replace("\n"," "),
                ha="center", va="center", fontsize=max(fs-1,5),
                color=s["muted"], alpha=0.85, zorder=3)
        ax.text(cx_, cy_ - rh*0.06, str(cnt),
                ha="center", va="center",
                fontsize=max(fs+3,10), fontweight="bold",
                color=s["text"], zorder=4)
        ax.text(cx_, cy_ - rh*0.28, "att",
                ha="center", va="center",
                fontsize=max(fs-2,5), color=s["muted"], zorder=4)

    # defender average badges (red circles on near/far post zones)
    if has_def:
        avg_near = dff[def_cols_near[0]].apply(pd.to_numeric,errors="coerce").mean() if def_cols_near[0] in dff.columns else 0
        avg_far  = dff[def_cols_far[0]].apply(pd.to_numeric,errors="coerce").mean()  if def_cols_far[0]  in dff.columns else 0
        # place near-post badge
        for (label, zx, zy, zw, zh) in zones:
            if "Near\nPost" in label and "Short" not in label:
                bx_ = (zy+zh/2) if vert else (zx+zw/2)
                by_ = (zx+zw/2) if vert else (zy+zh/2)
                ax.add_patch(plt.Circle((bx_, by_+3.5 if not vert else by_), 2.2,
                             facecolor=s["success"], edgecolor="white", linewidth=1.2, zorder=10))
                ax.text(bx_, by_+3.5 if not vert else by_, f"{avg_near:.1f}",
                        ha="center", va="center",
                        fontsize=max(s["tick_size"],9), fontweight="bold", color="white", zorder=11)
                ax.text(bx_, (by_+3.5-2.8) if not vert else (by_-3.0), "def",
                        ha="center", va="top",
                        fontsize=max(s["tick_size"]-2,5), color=s["muted"], zorder=11)
                break
        for (label, zx, zy, zw, zh) in zones:
            if label == "Far\nPost":
                bx_ = (zy+zh/2) if vert else (zx+zw/2)
                by_ = (zx+zw/2) if vert else (zy+zh/2)
                ax.add_patch(plt.Circle((bx_, by_+3.5 if not vert else by_), 2.2,
                             facecolor=s["success"], edgecolor="white", linewidth=1.2, zorder=10))
                ax.text(bx_, by_+3.5 if not vert else by_, f"{avg_far:.1f}",
                        ha="center", va="center",
                        fontsize=max(s["tick_size"],9), fontweight="bold", color="white", zorder=11)
                ax.text(bx_, (by_+3.5-2.8) if not vert else (by_-3.0), "def",
                        ha="center", va="top",
                        fontsize=max(s["tick_size"]-2,5), color=s["muted"], zorder=11)
                break

    # legend
    if s.get("show_legend", True):
        h = [mpl.patches.Patch(facecolor=s["danger"], alpha=0.55, label="Attacker deliveries"),
             mpl.patches.Patch(facecolor=s["success"], label="Avg defenders (badge)")]
        leg = ax.legend(handles=h, frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5,-0.04), ncol=2)
        style_legend(leg, s)

    chart_title(ax, "Defensive Shape Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFENDER VS ATTACKER ZONE MATCHUP (grouped bar)
# Compares average attackers vs average defenders per zone side by side.
# Insight: which zones are numerically overloaded by attackers?
# ─────────────────────────────────────────────────────────────────────────────
def chart_defender_attacker_matchup(df, theme_name, flip_y=False, style_overrides=None):
    s   = resolve_style(theme_name, style_overrides)
    dff = _prep(df, flip_y)
    dd  = dff.dropna(subset=["x2","y2"]).copy()

    dominant_side = _side_dominant(dff)
    zones = _barca_zones(dominant_side)
    total = max(len(dd), 1)

    zone_labels = []
    att_vals    = []
    def_vals    = []

    att_zone_cols = {
        "Near\nPost":   ["players_near_post"],
        "Far\nPost":    ["players_far_post"],
        "Small\nArea":  ["players_small_area","players_6yard"],
        "Penalty\nSpot":["players_penalty","players_penalty_area"],
    }
    def_zone_cols = {
        "Near\nPost":   ["defenders_near_post"],
        "Far\nPost":    ["defenders_far_post"],
        "Small\nArea":  [],
        "Penalty\nSpot":[],
    }

    for (label, zx, zy, zw, zh) in zones:
        short = label.replace("\n"," ")
        # attacker count from x2/y2
        mask = (dd["x2"] >= zx) & (dd["x2"] < zx+zw) & (dd["y2"] >= zy) & (dd["y2"] < zy+zh)
        att_n = float(mask.sum()) / total * 10  # normalise to per-10-kicks scale

        # attacker players column fallback
        for c in att_zone_cols.get(label, []):
            if c in dff.columns:
                v = pd.to_numeric(dff[c], errors="coerce").dropna()
                if len(v): att_n = float(v.mean()); break

        # defender players
        def_n = 0.0
        for c in def_zone_cols.get(label, []):
            if c in dff.columns:
                v = pd.to_numeric(dff[c], errors="coerce").dropna()
                if len(v): def_n = float(v.mean()); break

        zone_labels.append(short)
        att_vals.append(round(att_n, 2))
        def_vals.append(round(def_n, 2))

    fig, ax = _base_fig(s, (9.0, 5.2))
    x   = np.arange(len(zone_labels))
    w   = 0.38
    bcm = s.get("bar_colors", {})
    att_c = bcm.get("danger",  s["danger"])
    def_c = bcm.get("success", s["success"])

    bars_a = ax.bar(x - w/2, att_vals, w, color=att_c, edgecolor=s["lines"],
                    linewidth=0.8, label="Avg Attackers", alpha=0.90)
    bars_d = ax.bar(x + w/2, def_vals, w, color=def_c, edgecolor=s["lines"],
                    linewidth=0.8, label="Avg Defenders", alpha=0.90)

    for bar in list(bars_a) + list(bars_d):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=max(s["tick_size"]-1,7), color=s["text"])

    ax.set_xticks(x); ax.set_xticklabels(zone_labels, fontsize=s["tick_size"])
    ax.set_ylabel("Avg players per kick")
    themed_bar(ax, s)
    if s.get("show_legend", True):
        leg = ax.legend(frameon=True); style_legend(leg, s)
    chart_title(ax, "Defender vs Attacker Zone Matchup", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEARANCE OUTCOME MAP
# Plots x2/y2 positions, marking only rows where result = clearance.
# Dot size encodes whether second_ball_win was achieved (won = large,
# lost = small hollow).
# Insight: where do clearances land and does the team recover the second ball?
# ─────────────────────────────────────────────────────────────────────────────
def chart_clearance_outcome_map(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6, 8.5) if vert else (10, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)

    dd = dff.dropna(subset=["x2","y2"]).copy()

    # isolate clearances
    if "result" in dd.columns:
        mask_cl = dd["result"].astype(str).str.lower().str.contains("clear", na=False)
        dd = dd[mask_cl].copy()

    dd["_sbw"] = _get_sbw(dd)

    won  = dd[dd["_sbw"] == 1]
    lost = dd[dd["_sbw"] == 0]
    msz  = max(s["marker_size"] * 1.3, 60)

    if len(won):
        px = won["y2"].values if vert else won["x2"].values
        py = won["x2"].values if vert else won["y2"].values
        ax.scatter(px, py, s=msz, color=s["success"],
                   edgecolors=s["pitch_lines"], linewidths=s["marker_edge_width"],
                   alpha=s["alpha"], zorder=9, clip_on=False, label="2nd Ball Won ✓")

    if len(lost):
        px = lost["y2"].values if vert else lost["x2"].values
        py = lost["x2"].values if vert else lost["y2"].values
        ax.scatter(px, py, s=msz*0.55, facecolors="none",
                   edgecolors=s["danger"], linewidths=s["line_width"]+0.5,
                   alpha=s["alpha"], zorder=9, clip_on=False, label="2nd Ball Lost ✗")

    if len(dd) == 0:
        ax.text(0.5, 0.5, "No clearance rows found\n(check 'result' column)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=s["label_size"], color=s["muted"])
    else:
        # count badge
        ax.text(2, PW-1 if not vert else PL-1,
                f"Clearances: {len(dd)}  |  2nd ball won: {len(won)}  ({len(won)/max(len(dd),1)*100:.0f}%)",
                ha="left", va="top", fontsize=max(s["tick_size"]-1,7),
                color=s["muted"], zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=s["bg"],
                          edgecolor=s["lines"], alpha=0.65))

    if s.get("show_legend", True) and len(dd) > 0:
        leg = ax.legend(frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5, -0.04), ncol=2)
        style_legend(leg, s)

    chart_title(ax, "Clearance Outcome Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 4. SET PIECE CONCEDED HEATMAP
# KDE heatmap of all delivery end points (x2/y2) for UNSUCCESSFUL outcomes.
# Shows the areas the defence is most frequently exposed to.
# Insight: which zones does the opposition keep threatening?
# ─────────────────────────────────────────────────────────────────────────────
def chart_set_piece_conceded_heatmap(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6.2, 8.8) if vert else (10.0, 6.8)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)
    _draw_thirds(ax, s, vert)
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)

    dd = dff.dropna(subset=["x2","y2"]).copy()

    # keep only unsuccessful / conceded deliveries
    if "outcome" in dd.columns:
        mask_bad = dd["outcome"].astype(str).str.lower().str.strip().isin(
            ["unsuccessful", "conceded", "failed", "loss", "no"]
        )
        if mask_bad.any():
            dd = dd[mask_bad].copy()

    if len(dd):
        px = dd["y2"].values if vert else dd["x2"].values
        py = dd["x2"].values if vert else dd["y2"].values
        try:
            pitch.kdeplot(px, py, ax=ax, fill=True, levels=55,
                          alpha=s["kde_alpha"], cmap="Reds", zorder=4)
        except:
            pass
        ax.scatter(px, py, s=max(s["marker_size"]*0.35, 12), color=s["danger"],
                   edgecolors="none", alpha=min(s["alpha"]*0.55, 0.60),
                   zorder=8, clip_on=False)
        ax.text(2, PW-1 if not vert else PL-1,
                f"Unsuccessful deliveries: {len(dd)}",
                ha="left", va="top", fontsize=max(s["tick_size"]-1,7),
                color=s["muted"], zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=s["bg"],
                          edgecolor=s["lines"], alpha=0.65))

    chart_title(ax, "Set Piece Conceded Heatmap", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5. DEFENSIVE SUCCESS RATE BY ZONE (bar chart)
# Per zone: % of deliveries where first_contact_win = yes.
# Also shows raw counts as text.
# Insight: which zones does the defence win / lose the first contact battle?
# ─────────────────────────────────────────────────────────────────────────────
def chart_defensive_success_rate_by_zone(df, theme_name, flip_y=False, style_overrides=None):
    s   = resolve_style(theme_name, style_overrides)
    dff = _prep(df, flip_y)
    dd  = dff.dropna(subset=["x2","y2"]).copy()
    dd["_fcw"]  = _get_fcw(dd)
    dd["_zone"] = dd.apply(lambda r: _infer_zone(r.get("x2"), r.get("y2")), axis=1)

    zone_order = ["near_post_short","near_post","small_area",
                  "penalty_spot","far_post","far_post_long","box_front"]
    zone_labels_map = {
        "near_post_short":"Near Post\nShort","near_post":"Near Post",
        "small_area":"Small Area","penalty_spot":"Penalty Spot",
        "far_post":"Far Post","far_post_long":"Far Post\nLong",
        "box_front":"Box Front"
    }

    records = []
    for z in zone_order:
        sub = dd[dd["_zone"] == z]
        n   = len(sub)
        rate = sub["_fcw"].mean() * 100 if n > 0 else 0.0
        records.append({"zone": z, "label": zone_labels_map[z],
                         "rate": round(rate,1), "n": n})
    rdf = pd.DataFrame(records)

    fig, ax = _base_fig(s, (9.5, 5.2))
    x   = np.arange(len(rdf))
    bcm = s.get("bar_colors", {})
    bar_colors = [bcm.get("success",s["success"]) if r >= 50
                  else bcm.get("danger",s["danger"])
                  for r in rdf["rate"]]

    bars = ax.bar(x, rdf["rate"], color=bar_colors,
                  edgecolor=s["lines"], linewidth=0.8, alpha=0.90)

    for bar, row in zip(bars, rdf.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f"{h:.0f}%\n(n={row.n})", ha="center", va="bottom",
                fontsize=max(s["tick_size"]-1,6), color=s["text"])

    ax.axhline(50, color=s["muted"], lw=1.0, ls="--", alpha=0.55, zorder=3)
    ax.text(len(rdf)-0.4, 51.5, "50%", color=s["muted"],
            fontsize=max(s["tick_size"]-1,6), ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(rdf["label"], fontsize=s["tick_size"])
    ax.set_ylabel("First Contact Win %")
    ax.set_ylim(0, max(rdf["rate"].max() + 15, 70))
    themed_bar(ax, s)

    if s.get("show_legend", True):
        h_ = [mpl.patches.Patch(facecolor=bcm.get("success",s["success"]), label="Win rate ≥ 50%"),
              mpl.patches.Patch(facecolor=bcm.get("danger",s["danger"]),   label="Win rate < 50%")]
        leg = ax.legend(handles=h_, frameon=True); style_legend(leg, s)

    chart_title(ax, "Defensive Success Rate By Zone", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIRST CONTACT WIN RATE TREND (line chart per opponent / match)
# Groups by opponent and shows first-contact win rate to track defensive
# improvement or regression across matches.
# Insight: is the team getting better or worse at defending set pieces?
# ─────────────────────────────────────────────────────────────────────────────
def chart_first_contact_win_rate_trend(df, theme_name, flip_y=False, style_overrides=None):
    s   = resolve_style(theme_name, style_overrides)
    dff = df.copy()
    dff["_fcw"] = _get_fcw(dff)

    # group column: prefer match_id → opponent → competition
    group_col = None
    for c in ["match_id", "opponent", "competition", "date"]:
        if c in dff.columns and dff[c].notna().any():
            group_col = c; break

    fig, ax = _base_fig(s, (9.5, 5.0))

    if group_col is None:
        ax.text(0.5, 0.5, "No match/opponent column found.\nAdd a 'match_id' or 'opponent' column.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=s["label_size"], color=s["muted"])
        chart_title(ax, "First Contact Win Rate Trend", s)
        if s["tight_layout"]: fig.tight_layout()
        return fig

    grp = dff.groupby(group_col, sort=False)["_fcw"].agg(["mean","count"]).reset_index()
    grp.columns = [group_col, "rate", "n"]
    grp["rate_pct"] = grp["rate"] * 100

    xs = np.arange(len(grp))
    line_c = s["accent"]

    ax.fill_between(xs, grp["rate_pct"], alpha=0.18, color=line_c, zorder=2)
    ax.plot(xs, grp["rate_pct"], color=line_c, lw=s["line_width"]+0.8,
            marker="o", markersize=7, zorder=4)

    for xi, row in zip(xs, grp.itertuples()):
        ax.text(xi, row.rate_pct + 1.5, f"{row.rate_pct:.0f}%\n(n={row.n})",
                ha="center", va="bottom", fontsize=max(s["tick_size"]-1,6),
                color=s["text"])

    avg_rate = grp["rate_pct"].mean()
    ax.axhline(avg_rate, color=s["warning"], lw=1.2, ls="--", alpha=0.70, zorder=3)
    ax.text(len(grp)-0.3, avg_rate+1.5, f"Avg {avg_rate:.0f}%",
            color=s["warning"], fontsize=max(s["tick_size"]-1,6), ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(grp[group_col].astype(str), fontsize=s["tick_size"], rotation=20, ha="right")
    ax.set_ylabel("FC Win Rate %")
    ax.set_ylim(0, min(grp["rate_pct"].max() + 20, 105))
    themed_bar(ax, s)
    ax.grid(axis="x", alpha=0.0)  # remove vertical grid

    chart_title(ax, "First Contact Win Rate Trend", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 7. SECOND BALL RECOVERY MAP
# Plots x2/y2 and colour-codes by whether second_ball_win was achieved.
# Zones are shaded by net recovery rate (green = high, red = low).
# Insight: after a set piece delivery, where does the defence recover / lose
# the second ball?
# ─────────────────────────────────────────────────────────────────────────────
def chart_second_ball_recovery_map(df, theme_name, flip_y=False, style_overrides=None):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6, 8.5) if vert else (10, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)
    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)

    dominant_side = _side_dominant(dff)
    zones = _barca_zones(dominant_side)
    dd = dff.dropna(subset=["x2","y2"]).copy()
    dd["_sbw"] = _get_sbw(dd)

    # per-zone recovery rate → fill colour
    for (label, zx, zy, zw, zh) in zones:
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh
        mask = (dd["x2"] >= zx) & (dd["x2"] < zx+zw) & (dd["y2"] >= zy) & (dd["y2"] < zy+zh)
        sub  = dd[mask]
        n    = len(sub)
        if n == 0:
            ax.add_patch(Rectangle((rx,ry),rw,rh,
                                    facecolor=s["muted"],edgecolor=s["pitch_lines"],
                                    linewidth=0.7,alpha=0.10,zorder=1))
            continue
        rate = sub["_sbw"].mean()  # 0→1
        # green for high recovery, red for low
        from_rgb = np.array(mpl.colors.to_rgb(s["danger"]))
        to_rgb   = np.array(mpl.colors.to_rgb(s["success"]))
        rgb = from_rgb * (1-rate) + to_rgb * rate
        ax.add_patch(Rectangle((rx,ry),rw,rh,
                                facecolor=rgb,edgecolor=s["pitch_lines"],
                                linewidth=0.7,alpha=0.35+rate*0.45,zorder=1))
        cx_ = rx+rw/2; cy_ = ry+rh/2
        fs  = max(s["tick_size"]-1,7)
        ax.text(cx_, cy_+rh*0.18, label.replace("\n"," "),
                ha="center",va="center",fontsize=max(fs-1,5),
                color=s["muted"],alpha=0.85,zorder=3)
        ax.text(cx_, cy_-rh*0.06, f"{rate*100:.0f}%",
                ha="center",va="center",
                fontsize=max(fs+2,10),fontweight="bold",color=s["text"],zorder=4)
        ax.text(cx_, cy_-rh*0.28, f"n={n}",
                ha="center",va="center",
                fontsize=max(fs-2,5),color=s["muted"],zorder=4)

    # scatter dots coloured by recovery
    won  = dd[dd["_sbw"]==1]
    lost = dd[dd["_sbw"]==0]
    msz  = max(s["marker_size"]*0.7, 30)
    if len(won):
        px=won["y2"].values if vert else won["x2"].values
        py=won["x2"].values if vert else won["y2"].values
        ax.scatter(px,py,s=msz,color=s["success"],edgecolors=s["pitch_lines"],
                   linewidths=s["marker_edge_width"],alpha=s["alpha"],
                   zorder=9,clip_on=False,label="2nd Ball Won")
    if len(lost):
        px=lost["y2"].values if vert else lost["x2"].values
        py=lost["x2"].values if vert else lost["y2"].values
        ax.scatter(px,py,s=msz,facecolors="none",
                   edgecolors=s["danger"],linewidths=s["line_width"]+0.4,
                   alpha=s["alpha"],zorder=9,clip_on=False,label="2nd Ball Lost")

    if s.get("show_legend", True):
        leg = ax.legend(frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5,-0.04), ncol=2)
        style_legend(leg, s)

    # overall badge
    overall_rate = dd["_sbw"].mean() * 100 if len(dd) else 0
    bx_ = 5 if not vert else 2
    by_ = PW-2 if not vert else PL-2
    ax.text(bx_, by_, f"Overall 2nd ball recovery: {overall_rate:.0f}%",
            ha="left",va="top",fontsize=max(s["tick_size"]-1,7),
            color=s["muted"],zorder=12,
            bbox=dict(boxstyle="round,pad=0.3",facecolor=s["bg"],
                      edgecolor=s["lines"],alpha=0.65))

    chart_title(ax, "Second Ball Recovery Map", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# NEW CONTACT / MARKING CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _shirt_counts(series):
    s = series.dropna().astype(str).str.strip()
    s = s[~s.isin(["", "nan", "none"])]
    return s.value_counts().sort_values(ascending=False)

def chart_first_contact_players_by_shirt(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides)
    dff = df.copy()

    if "play_type" in dff.columns:
        play = dff["play_type"].astype(str).str.lower().str.strip()
        off = dff[play.eq("attack")].copy()
        deff = dff[play.eq("defence")].copy()
    else:
        off = dff.copy()
        deff = dff.iloc[0:0].copy()

    off_counts = _shirt_counts(off.get("first_contact_player", pd.Series(dtype=object)))
    def_counts = _shirt_counts(deff.get("first_contact_player", pd.Series(dtype=object)))

    labels = sorted(set(off_counts.index.tolist()) | set(def_counts.index.tolist()), key=lambda x: (len(str(x)), str(x)))
    off_vals = [int(off_counts.get(lbl, 0)) for lbl in labels]
    def_vals = [int(def_counts.get(lbl, 0)) for lbl in labels]

    fig, ax = _base_fig(s, (10, 5.4))
    x = np.arange(len(labels))
    w = 0.38
    bcm = s.get("bar_colors", {})
    bars1 = ax.bar(x - w/2, off_vals, w, color=bcm.get("default", s["accent"]),
                   edgecolor=s["lines"], linewidth=0.8, label="Offensive")
    bars2 = ax.bar(x + w/2, def_vals, w, color=bcm.get("danger", s["danger"]),
                   edgecolor=s["lines"], linewidth=0.8, label="Defensive")

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + 0.05, f"{int(h)}",
                        ha="center", va="bottom", fontsize=max(s["tick_size"]-1, 7), color=s["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else ["No data"])
    ax.set_ylabel("First contacts")
    themed_bar(ax, s)
    if s.get("show_legend", True):
        leg = ax.legend(frameon=True)
        style_legend(leg, s)
    chart_title(ax, "First Contact Players by Shirt Number", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_players_made_first_contact(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides)
    counts = _shirt_counts(df.get("first_contact_player", pd.Series(dtype=object))).head(15)

    fig, ax = _base_fig(s, (8.6, 5.2))
    bc = s.get("bar_colors", {}).get("default", s["accent"])
    ax.barh(counts.index[::-1].astype(str), counts.values[::-1], color=bc,
            edgecolor=s["lines"], linewidth=0.8)
    for yi, val in enumerate(counts.values[::-1]):
        ax.text(val + 0.05, yi, str(int(val)), va="center", ha="left",
                fontsize=max(s["tick_size"]-1, 7), color=s["text"])
    ax.set_xlabel("Count")
    themed_bar(ax, s)
    chart_title(ax, "Players Who Made First Contact", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_players_lost_first_contact(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides)
    series = df.get("lost_first_contact_player", pd.Series(dtype=object))
    counts = _shirt_counts(series).head(15)

    fig, ax = _base_fig(s, (8.8, 5.2))
    bc = s.get("bar_colors", {}).get("danger", s["danger"])
    ax.barh(counts.index[::-1].astype(str), counts.values[::-1], color=bc,
            edgecolor=s["lines"], linewidth=0.8)
    for yi, val in enumerate(counts.values[::-1]):
        ax.text(val + 0.05, yi, str(int(val)), va="center", ha="left",
                fontsize=max(s["tick_size"]-1, 7), color=s["text"])
    ax.set_xlabel("Count")
    themed_bar(ax, s)
    chart_title(ax, "Players That Lost First Contact", s)
    if len(counts) == 0:
        ax.text(0.5, 0.5, "Add 'lost_first_contact_player' column to the CSV",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=s["label_size"], color=s["muted"])
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_box_marking_scheme(df, theme_name, flip_y=False, style_overrides=None):
    s = resolve_style(theme_name, style_overrides)
    fig, ax = _base_fig(s, (8.6, 5.2))
    man = pd.to_numeric(df.get("man_marking_in_box", pd.Series(dtype=float)), errors="coerce")
    zonal = pd.to_numeric(df.get("zonal_marking_in_box", pd.Series(dtype=float)), errors="coerce")

    vals = [float(man.mean()) if man.notna().any() else 0.0,
            float(zonal.mean()) if zonal.notna().any() else 0.0]
    labels = ["Man Marking", "Zonal"]
    colors = [s.get("bar_colors", {}).get("danger", s["danger"]),
              s.get("bar_colors", {}).get("success", s["success"])]

    bars = ax.bar(labels, vals, color=colors, edgecolor=s["lines"], linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.1f}",
                ha="center", va="bottom", fontsize=max(s["tick_size"]-1, 7), color=s["text"])

    total = sum(vals)
    if total > 0:
        ax.text(0.98, 0.95,
                f"Man: {vals[0]/total*100:.0f}%  |  Zonal: {vals[1]/total*100:.0f}%",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=max(s["tick_size"]-1, 7), color=s["muted"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor=s["bg"], edgecolor=s["lines"], alpha=0.7))

    ax.set_ylabel("Avg players inside box")
    themed_bar(ax, s)
    chart_title(ax, "Box Marking Scheme", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
CHART_BUILDERS = {
    "Delivery Start Map":                      chart_delivery_start_map,
    "Delivery Heatmap":                        chart_delivery_heatmap,
    "Delivery End Scatter - Left Corner":      chart_delivery_end_scatter_left,
    "Delivery End Scatter - Right Corner":     chart_delivery_end_scatter_right,
    "Delivery Trajectories - Left Corners":    chart_delivery_trajectories_left,
    "Delivery Trajectories - Right Corners":   chart_delivery_trajectories_right,
    "Attack Free Kick Trajectories":           chart_attack_freekick_trajectories,
    "Average Delivery Path":                   chart_average_delivery_path,
    "Heat + Trajectories":                     chart_heat_plus_trajectories,
    "Trajectory Clusters":                     chart_trajectory_clusters,
    "Delivery Length Distribution":            chart_delivery_length_distribution,
    "Delivery Direction Map":                  chart_delivery_direction_map,
    "Outcome Distribution":                    chart_outcome_distribution,
    "Target Zone Breakdown":                   chart_target_zone_breakdown,
    "Zone Delivery Count Map - Left Corner":   chart_zone_count_left,
    "Zone Delivery Count Map - Right Corner":  chart_zone_count_right,
    "Avg Players Per Zone - Left Corner":      chart_avg_players_left,
    "Avg Players Per Zone - Right Corner":     chart_avg_players_right,
    "First Contact Win By Zone":               chart_first_contact_win_by_zone,
    "Routine Breakdown":                       chart_routine_breakdown,
    "Shot Map":                                chart_shot_map,
    "Second Ball Map":                         chart_second_ball_map,
    "Defensive Vulnerability Map":             chart_defensive_vulnerability_map,
    "Taker Profile":                           chart_taker_profile,
    "Structure Zone Averages":                 chart_structure_zone_averages,
    "Set Piece Landing Heatmap":               chart_set_piece_landing_heatmap,
    "Taker Stats Table":                       chart_taker_stats_table,
    "First Contact Location Map":              chart_first_contact_map,
    "First Contact Players by Shirt Number":   chart_first_contact_players_by_shirt,
    "Players Who Made First Contact":          chart_players_made_first_contact,
    "Players That Lost First Contact":         chart_players_lost_first_contact,
    "Box Marking Scheme":                      chart_box_marking_scheme,
    # ── DEFENSIVE ────────────────────────────────────────────────────────────
    "Defensive Shape Map":                     chart_defensive_shape_map,
    "Defender vs Attacker Zone Matchup":       chart_defender_attacker_matchup,
    "Clearance Outcome Map":                   chart_clearance_outcome_map,
    "Set Piece Conceded Heatmap":              chart_set_piece_conceded_heatmap,
    "Defensive Success Rate By Zone":          chart_defensive_success_rate_by_zone,
    "First Contact Win Rate Trend":            chart_first_contact_win_rate_trend,
    "Second Ball Recovery Map":                chart_second_ball_recovery_map,
}
