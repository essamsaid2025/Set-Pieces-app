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
# BARCELONA ZONES  — exact match to Figure 8
#
#  LEFT corner (taker at y≈0, top):  Near=top(y low)  Far=bottom(y high)
#  RIGHT corner (taker at y≈64, bottom): Near=bottom(y high)  Far=top(y low)
#
#  Each zone: (label, x0, y0, width, height)
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
    else:   # right corner — mirror near/far
        return [
            ("Near Post\nShort",  BOX_X0, BOX_Y1,  BOX_W, PW - BOX_Y1),
            ("Near\nPost",        BOX_X0, SIX_Y1,  BOX_W, BOX_Y1 - SIX_Y1),
            ("Small\nArea",       SIX_X0, SIX_Y0,  BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
            ("Penalty\nSpot",     BOX_X0, SIX_Y0,  SIX_X0 - BOX_X0, SIX_Y1 - SIX_Y0),
            ("Far\nPost",         BOX_X0, BOX_Y0,  BOX_W, SIX_Y0 - BOX_Y0),
            ("Far Post\nLong",    BOX_X0, 0.0,     BOX_W, BOX_Y0),
            ("Box\nFront",        72.0,   BOX_Y0,  BOX_X0 - 72.0, BOX_H),
        ]

# colour key per zone (index-matched to list above)
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
    "First Contact Location Map":             ["x2", "y2"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE PITCH  (used when mplsoccer not installed)
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
    """Draw pitch then lock axis limits — call AFTER pitch.draw()."""
    px, py = s["pitch_pad_x"], s["pitch_pad_y"]
    if vertical:
        ax.set_xlim(-py, PW + py); ax.set_ylim(-px, PL + px)
    else:
        ax.set_xlim(-px, PL + px); ax.set_ylim(-py, PW + py)
    ax.set_facecolor(s["pitch"])
    if not s.get("show_ticks", False): ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_autoscale_on(False)   # ← prevent scatter from rescaling axes

def chart_title(ax, t, s):
    if s.get("show_title", True):
        ax.set_title(t, color=s["text"], fontsize=s["title_size"],
                     fontweight=s["title_weight"], pad=12)

def style_legend(leg, s):
    if not leg: return
    f = leg.get_frame()
    if f: f.set_facecolor(s["panel"]); f.set_edgecolor(s["lines"]); f.set_alpha(0.95)
    for t in leg.get_texts(): t.set_color(s["text"])

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
# COORDINATE SCALING  — global-max approach (fixes y / y2 mismatch)
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
# show_labels=True  → trajectory charts only
# show_labels=False → all other pitch charts (coloured areas, no text)
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
        if d == "outswing": return -0.30
    elif corner_label == "right_bottom":
        if d == "inswing": return -0.30
        if d == "outswing": return 0.30
    elif corner_label == "left_top":
        if d == "inswing": return -0.30
        if d == "outswing": return 0.30
    elif corner_label == "left_bottom":
        if d == "inswing": return 0.30
        if d == "outswing": return -0.30
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
        # numeric path (already 0/1)
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().any():
            return (num > 0).astype(int)
        # string path: 'Yes'/'No' survived or pre-normalization
        return col.astype(str).str.strip().str.lower().isin(
            ["yes","1","true","won","win","successful","success"]
        ).astype(int)
    # fallback: use outcome column
    return get_set_piece_series(df).str.lower().isin(
        ["successful","success","won","win"]
    ).astype(int)

def _get_sbw(df):
    if "second_ball_win" in df.columns: return bool01(df["second_ball_win"])
    return get_set_piece_series(df).str.lower().isin(["second_ball_win","won_second_ball"]).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY TRAJECTORY CHARTS  (with zone labels)
# ─────────────────────────────────────────────────────────────────────────────
def _traj_chart(df, theme_name, flip_y, style_overrides, title, corner_side):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    # filter by 'side' column
    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    figsize = (6.4, 8.4) if vert else (8.4, 6.4)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    # ← trajectory charts: show_labels=True
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
# DELIVERY END SCATTER  — direct ax.scatter to guarantee visibility
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
    # ← other charts: show_labels=False (areas only, no text)
    _draw_zones(ax, s, corner_side, alpha=0.18, vertical=vert, show_labels=False)

    title = f"Delivery End Scatter — {'Left' if corner_side=='left' else 'Right'} Corner"
    dd    = dff.dropna(subset=["x2","y2"]).copy()

    if len(dd):
        # ── DIRECT ax.scatter (bypasses any mplsoccer coordinate quirks) ──
        if vert:
            px = dd["y2"].values; py = dd["x2"].values
        else:
            px = dd["x2"].values; py = dd["y2"].values

        base_color = s.get("scatter_dot_color", s["accent"])
        cmap       = _cmap_delivery(s)
        msz        = max(s["marker_size"] * 1.4, 60)   # bigger for visibility
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

        # count badge per zone
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
# ZONE DELIVERY COUNT MAP  (like pic 2 — heatmap intensity + counts)
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

    # zoom to attacking area
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

    # avg-players badge (red circle, bottom)
    cols = ["players_near_post","players_far_post","players_6yard","players_penalty"]
    avg  = next((dff[c].mean() for c in cols if c in dff.columns and dff[c].notna().any()), None)
    if avg is None:
        in_box = dd[(dd["x2"] >= BOX_X0) & (dd["y2"] >= BOX_Y0) & (dd["y2"] <= BOX_Y1)]
        avg = round(len(in_box) / total * 5, 1)

    bx_, by_ = (32, 97) if vert else (84, 63.5)
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
# ALL OTHER PITCH CHARTS  — use show_labels=False
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

# ─── bar charts ──────────────────────────────────────────────────────────────
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
        # Extract player shirt number from taker column value
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
    mw=2.0   # bar width; % label placed after bar
    ax.text(fw/2,fh-0.28,"Set Piece Taker Stats",ha="center",va="top",fontsize=s["title_size"]+2,fontweight="bold",color=s["text"])
    ax.text(fw/2,fh-0.62,"Sorted by number of set piece sequences taken",ha="center",va="top",fontsize=s["tick_size"],color=s["muted"])
    # "right" column removed — success bar is the last element
    cx={"shirt":0.45,"seq":1.55,"ins":2.80,"out":3.85,"left":4.90,"rate":6.60}
    hy=fh-hh+0.05
    for k,lbl in {"seq":"SEQ","ins":"INSWING","out":"OUTSWING","left":"LEFT","rate":"SUCCESS %"}.items():
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
        # No duplicate taker number outside shirt — shirt number IS the identifier
        for k in ["seq","ins","out","left"]:
            v={"seq":row["sequences"],"ins":row["inswing"],"out":row["outswing"],"left":row["left"]}[k]
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
# AVG PLAYERS PER ZONE MAP  (matches reference image style)
# ─────────────────────────────────────────────────────────────────────────────
# Column name mapping (after normalize_set_piece_df pipeline):
#   players_near_post  → Near Post zone
#   players_far_post   → Far Post zone
#   players_small_area → Small Area (6-yard box)
#   players_penalty    → Penalty Spot area
#
# "Avg players in box" = mean of per-row sums across the 4 zones
# (matches the reference image showing 5.0 for this dataset)
# ─────────────────────────────────────────────────────────────────────────────

# Map zone labels → player column names (after normalization)
_ZONE_PLAYER_COLS = {
    "Near\nPost":    ["players_near_post"],
    "Far\nPost":     ["players_far_post"],
    "Small\nArea":   ["players_small_area", "players_6yard"],
    "Penalty\nSpot": ["players_penalty", "players_penalty_area"],
}

def _get_zone_avg(df_side, col_candidates):
    """Return mean of first matching column, or 0.0."""
    for c in col_candidates:
        if c in df_side.columns:
            v = pd.to_numeric(df_side[c], errors="coerce").dropna()
            if len(v): return round(float(v.mean()), 1)
    return 0.0

def _get_avg_in_box(df_side):
    """
    Average total players in box per corner kick.
    = mean of (sum of 4 zone counts per row).
    """
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
    """Simulate a Gaussian heatmap blob with concentric circles."""
    steps = 10
    for i in range(steps, 0, -1):
        r = radius * i / steps
        # alpha rises sharply at centre
        a = peak_alpha * ((steps - i + 1) / steps) ** 2.2
        ax.add_patch(plt.Circle((cx, cy), r, facecolor=color,
                                 edgecolor="none", alpha=a, zorder=3))

def _avg_players_zone_map(df, theme_name, flip_y, style_overrides, corner_side):
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    # ── filter by corner side ────────────────────────────────────────────────
    if "side" in dff.columns:
        mask = dff["side"].astype(str).str.lower() == corner_side
        if mask.any(): dff = dff[mask].copy()

    # ── compute per-zone averages ────────────────────────────────────────────
    zone_avgs = {label: _get_zone_avg(dff, cols)
                 for label, cols in _ZONE_PLAYER_COLS.items()}
    avg_in_box = _get_avg_in_box(dff)
    max_avg    = max(zone_avgs.values(), default=1.0)
    max_avg    = max(max_avg, 0.01)

    # ── figure & pitch ────────────────────────────────────────────────────────
    figsize = (6, 5.5) if vert else (8, 5.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    # zoom to attacking end — same box proportions as other charts
    if not vert:
        ax.set_xlim(68, 103); ax.set_ylim(-1, 65)
    else:
        ax.set_xlim(-1, 65);  ax.set_ylim(68, 103)

    # ── colour ramp: very dark red → bright red ───────────────────────────────
    _RED_LOW  = np.array(mpl.colors.to_rgb("#1c0404"))  # near-black red
    _RED_HIGH = np.array(mpl.colors.to_rgb("#ff1a00"))  # vivid red

    zones = _barca_zones(corner_side)

    # ── draw zone fills ───────────────────────────────────────────────────────
    for (label, zx, zy, zw, zh) in zones:
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh

        avg_v = zone_avgs.get(label)
        if avg_v is not None:
            t     = avg_v / max_avg                                # 0→1
            rgb   = _RED_LOW * (1 - t) + _RED_HIGH * t
            alpha = 0.30 + t * 0.58
        else:
            rgb   = np.array([0.07, 0.07, 0.07])
            alpha = 0.22

        ax.add_patch(Rectangle((rx, ry), rw, rh,
                                facecolor=rgb, edgecolor=s["pitch_lines"],
                                linewidth=0.8, alpha=alpha, zorder=2))

    # ── Gaussian blob on highest-value zone ───────────────────────────────────
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

    # ── also draw a smaller blob for the 2nd highest zone ────────────────────
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

    # ── zone labels & numbers ─────────────────────────────────────────────────
    fs_num  = max(s["title_size"] - 1, 13)   # big bold number
    fs_lbl  = max(s["tick_size"] - 2, 6)     # small zone label

    for (label, zx, zy, zw, zh) in zones:
        if vert: rx, ry, rw, rh = zy, zx, zh, zw
        else:    rx, ry, rw, rh = zx, zy, zw, zh
        cx_ = rx + rw / 2
        cy_ = ry + rh / 2

        avg_v = zone_avgs.get(label)
        if avg_v is not None:
            # big avg number
            ax.text(cx_, cy_, f"{avg_v:.1f}",
                    ha="center", va="center",
                    fontsize=fs_num, fontweight="bold",
                    color="white", zorder=9,
                    path_effects=[mpl_pe.withStroke(linewidth=2.5, foreground="black")])
            # small zone label inside zone (bottom)
            ax.text(cx_, ry + rh * 0.10, label.replace("\n", " "),
                    ha="center", va="bottom",
                    fontsize=fs_lbl, color=s["muted"], alpha=0.90, zorder=9)
        else:
            # zones without data: just the label
            ax.text(cx_, cy_, label.replace("\n", " "),
                    ha="center", va="center",
                    fontsize=fs_lbl, color=s["muted"], alpha=0.65, zorder=9)

    # ── "Avg players in box" badge — centred inside Box Front zone ──────────────
    # Box Front: x=72..83.5, y=BOX_Y0..BOX_Y1
    # Badge placed at horizontal centre of Box Front, lower portion of the zone
    box_front_cx = (72.0 + BOX_X0) / 2.0          # 77.75  (horizontal centre)
    box_front_badge_y = BOX_Y0 + (BOX_Y1 - BOX_Y0) * 0.30   # lower 30% of zone

    if not vert:
        bx_, by_ = box_front_cx, box_front_badge_y
    else:
        bx_, by_ = box_front_badge_y, box_front_cx

    # No connector lines — badge sits directly inside the Box Front zone

    ax.add_patch(plt.Circle((bx_, by_), 2.5,
                             facecolor="#ff3b30", edgecolor="white",
                             linewidth=1.4, zorder=20, clip_on=False))
    ax.text(bx_, by_, f"{avg_in_box:.1f}",
            ha="center", va="center",
            fontsize=max(s["tick_size"] + 1, 10), fontweight="bold",
            color="white", zorder=21, clip_on=False)
    ax.text(bx_, by_ + 3.0 if not vert else by_,
            "Avg. players\nin box",
            ha="center", va="bottom",
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
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FIRST CONTACT LOCATION MAP
# Shows x2/y2 delivery end positions, coloured & shaped by action type
# (result column: shot / clearance / threat / header / cross / other)
# ─────────────────────────────────────────────────────────────────────────────

# Action type config: marker, colour_key, display label
# Marker codes: 'o'=circle, '^'=triangle up, 'v'=tri down, 's'=square,
#               'D'=diamond, 'P'=plus-filled, 'X'=x-filled, '*'=star
_ACTION_CONFIG = {
    "shot":        ("*",  "#FFD400", "Shot"),
    "clearance":   ("v",  "#FF4D4D", "Clearance"),
    "cleareance":  ("v",  "#FF4D4D", "Clearance"),   # handle common typo
    "header":      ("^",  "#38BDF8", "Header"),
    "threat":      ("D",  "#A78BFA", "Threat"),
    "cross":       ("P",  "#22C55E", "Cross"),
    "foul":        ("s",  "#F97316", "Foul"),
    "other":       ("o",  "#94A3B8", "Other"),
}

def _normalise_action(val: str) -> str:
    v = str(val).strip().lower()
    # common spellings
    if v in ("clearance","cleareance","cleared"): return "clearance"
    if v in ("shot","shots","shoot"): return "shot"
    if v in ("header","head","headed"): return "header"
    if v in ("threat","dangerous","chance"): return "threat"
    if v in ("cross","crossed"): return "cross"
    if v in ("foul","fouled","foul won"): return "foul"
    return v if v in _ACTION_CONFIG else "other"

def chart_first_contact_map(df, theme_name, flip_y=False, style_overrides=None):
    """
    Scatter map at delivery end (x2/y2) coloured + shaped by action result.
    First-contact win shown via edgecolor (gold=won, grey=lost).
    """
    s    = resolve_style(theme_name, style_overrides)
    vert = s.get("pitch_vertical", False)
    pitch = make_pitch(s, vert)
    dff   = _prep(df, flip_y)

    figsize = (6, 8.5) if vert else (10, 6.5)
    fig, ax = _base_fig(s, figsize)
    pitch.draw(ax=ax)
    _setup_pitch_axes(ax, s, vert)

    if s.get("show_thirds", False): _draw_thirds(ax, s, vert)
    # silent zone overlay (no labels) for spatial context
    _draw_zones(ax, s, _side_dominant(dff), alpha=0.12, vertical=vert, show_labels=False)

    dd = dff.dropna(subset=["x2", "y2"]).copy()

    # resolve action column
    action_col = None
    for c in ["result", "action", "event", "first_contact_result"]:
        if c in dd.columns and dd[c].notna().any():
            action_col = c; break

    # resolve first_contact_win for edge highlight
    dd["_fcw"] = _get_fcw(dd)

    # allow user overrides from style
    user_colours  = s.get("action_colors", {})   # e.g. {"shot": "#ffff00"}
    visible_types = s.get("action_types_visible", None)  # None = show all

    msz   = max(s["marker_size"] * 1.6, 80)
    mew   = s["marker_edge_width"] + 0.5

    if action_col and dd[action_col].notna().any():
        dd["_action"] = dd[action_col].apply(_normalise_action)
        # filter visible types if user has limited them
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
                ax.scatter(xi, yi,
                           s=msz, marker=marker_, color=color_,
                           edgecolors=edge_c, linewidths=mew,
                           alpha=s["alpha"], zorder=9, clip_on=False)
    else:
        # no action column — just scatter all points
        px = dd["y2"].values if vert else dd["x2"].values
        py = dd["x2"].values if vert else dd["y2"].values
        ax.scatter(px, py, s=msz, color=s.get("scatter_dot_color", s["accent"]),
                   edgecolors=s["lines"], linewidths=mew,
                   alpha=s["alpha"], zorder=9, clip_on=False)

    # ── legend ────────────────────────────────────────────────────────────────
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
        # add first-contact-win indicator
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

CHART_BUILDERS = {
    "Delivery Start Map":                      chart_delivery_start_map,
    "Delivery Heatmap":                        chart_delivery_heatmap,
    "Delivery End Scatter - Left Corner":      chart_delivery_end_scatter_left,
    "Delivery End Scatter - Right Corner":     chart_delivery_end_scatter_right,
    "Delivery Trajectories - Left Corners":    chart_delivery_trajectories_left,
    "Delivery Trajectories - Right Corners":   chart_delivery_trajectories_right,
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
    "First Contact Location Map":             chart_first_contact_map,
}
