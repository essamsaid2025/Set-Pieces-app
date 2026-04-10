# set_piece_charts.py
import io, math, os, tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, FancyBboxPatch

from data_utils import bool01
from ui_theme import build_chart_style

try:
    from mplsoccer import Pitch as MplsoccerPitch, VerticalPitch
except Exception:
    MplsoccerPitch = None; VerticalPitch = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# =========================================================
# PITCH DIMENSIONS (100×64)
# =========================================================
PL, PW         = 100.0, 64.0
BOX_X0, BOX_X1 = 83.5, 100.0
BOX_Y0, BOX_Y1 = 13.84, 50.16
SIX_X0         = 94.5
SIX_Y0, SIX_Y1 = 24.84, 39.16
GOAL_Y0, GOAL_Y1 = 28.34, 35.66
PEN_X          = 88.5
BOX_W          = BOX_X1 - BOX_X0   # 16.5
BOX_H          = BOX_Y1 - BOX_Y0   # 36.32

# =========================================================
# ZONES — perspective-aware for LEFT vs RIGHT corner
# Labels updated to match requested names exactly
# Each zone: (label, x0, y0, w, h)
def _barca_zones(corner_side: str):
    """
    Return list of (label, x0, y0, w, h) for zones.
    corner_side: 'right' (taker at bottom y≈64) or 'left' (taker at top y≈0)
    """
    # Order of zones is important and index-matched to ZONE_CKEYS
    if corner_side == "right":
        # Right corner: taker at bottom (y≈64) -> Near = bottom half
        return [
            ("Near Post Short",  BOX_X0, BOX_Y1,        BOX_W, PW - BOX_Y1),   # below box (near taker)
            ("Near Post",        BOX_X0, SIX_Y1,         BOX_X0 - SIX_X0 + BOX_W - (BOX_X1 - SIX_X0),  BOX_Y1 - SIX_Y1),  # inner bottom strip
            ("Small Area",       SIX_X0, GOAL_Y1,        BOX_X1 - SIX_X0,       BOX_Y1 - GOAL_Y1),      # 6yd bottom
            ("Penalty Spot",     BOX_X0, SIX_Y0,         SIX_X0 - BOX_X0,       SIX_Y1 - SIX_Y0),       # central
            ("Small Area",       SIX_X0, SIX_Y0,         BOX_X1 - SIX_X0,       SIX_Y1 - SIX_Y0),       # 6yd middle
            ("Far Post",         BOX_X0, BOX_Y0,         SIX_X0 - BOX_X0,       SIX_Y0 - BOX_Y0),        # inner top strip
            ("Small Area",       SIX_X0, BOX_Y0,         BOX_X1 - SIX_X0,       GOAL_Y0 - BOX_Y0),       # 6yd top
            ("Far Post Long",    BOX_X0, 0.0,             BOX_W, BOX_Y0),                                  # above box (far from taker)
            ("Box Front",        72.0,   BOX_Y0,          BOX_X0 - 72.0,         BOX_H),
        ]
    else:
        # LEFT corner: taker at top → Near=top, Far=bottom (swap near/far)
        return [
            ("Near Post Short",  BOX_X0, 0.0,             BOX_W, BOX_Y0),                                  # above box (near taker)
            ("Near Post",        BOX_X0, BOX_Y0,          SIX_X0 - BOX_X0,       SIX_Y0 - BOX_Y0),        # inner top strip
            ("Small Area",       SIX_X0, BOX_Y0,          BOX_X1 - SIX_X0,       GOAL_Y0 - BOX_Y0),       # 6yd top
            ("Penalty Spot",     BOX_X0, SIX_Y0,          SIX_X0 - BOX_X0,       SIX_Y1 - SIX_Y0),        # central
            ("Small Area",       SIX_X0, SIX_Y0,          BOX_X1 - SIX_X0,       SIX_Y1 - SIX_Y0),        # 6yd middle
            ("Far Post",         BOX_X0, SIX_Y1,          SIX_X0 - BOX_X0,       BOX_Y1 - SIX_Y1),        # inner bottom strip
            ("Small Area",       SIX_X0, GOAL_Y1,         BOX_X1 - SIX_X0,       BOX_Y1 - GOAL_Y1),       # 6yd bottom
            ("Far Post Long",    BOX_X0, BOX_Y1,          BOX_W, PW - BOX_Y1),                             # below box (far from taker)
            ("Box Front",        72.0,   BOX_Y0,          BOX_X0 - 72.0,         BOX_H),
        ]

# Fixed colours per zone (index-matched to above list)
ZONE_CKEYS = ["accent","accent_2","warning","success","warning","accent_2","warning","accent","muted"]

# =========================================================
# CHART REQUIREMENTS
# =========================================================
CHART_REQUIREMENTS: Dict[str, List[str]] = {
    "Delivery Start Map":                      ["x","y"],
    "Delivery Heatmap":                        ["x2","y2"],
    "Delivery End Scatter - Left Corner":      ["x2","y2"],
    "Delivery End Scatter - Right Corner":     ["x2","y2"],
    "Delivery Trajectories - Left Corner":     ["x","y","x2","y2"],
    "Delivery Trajectories - Right Corner":    ["x","y","x2","y2"],
    "Average Delivery Path":                   ["x","y","x2","y2"],
    "Heat + Trajectories":                     ["x","y","x2","y2"],
    "Trajectory Clusters":                     ["x","y","x2","y2"],
    "Delivery Length Distribution":            ["x","y","x2","y2"],
    "Delivery Direction Map":                  ["x","y","x2","y2"],
    "Outcome Distribution":                    ["set_piece_type"],
    "Target Zone Breakdown":                   ["x2","y2"],
    "Zone Delivery Count Map - Left Corner":   ["x2","y2"],
    "Zone Delivery Count Map - Right Corner":  ["x2","y2"],
    "First Contact Win By Zone":               ["x2","y2"],
    "Routine Breakdown":                       ["set_piece_type"],
    "Shot Map":                                ["x","y"],
    "Second Ball Map":                         ["x","y"],
    "Defensive Vulnerability Map":             ["x","y"],
    "Taker Profile":                           ["set_piece_type"],
    "Structure Zone Averages":                 [],
    "Set Piece Landing Heatmap":               ["x2","y2"],
    "Taker Stats Table":                       ["taker"],
    # new
    "Zone Player Stats":                       ["x2","y2"],
}

# =========================================================
# SIMPLE PITCH FALLBACK
# (unchanged)
# =========================================================
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
        def rect(x,y,w,h): ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=lc,linewidth=lw,zorder=lz))
        def circle(cx,cy,r): ax.add_patch(plt.Circle((cx,cy),r,fill=False,color=lc,lw=lw,zorder=lz))
        if self.stripe and self.stripe_color:
            sw=W/10
            for i in range(10):
                if i%2==0: ax.add_patch(Rectangle((i*sw,0),sw,H,facecolor=self.stripe_color,edgecolor="none",alpha=0.35,zorder=0))
        rect(0,0,W,H)
        if self.vertical:
            ax.plot([0,W],[H/2,H/2],color=lc,lw=lw,zorder=lz)
            circle(W/2,H/2,9.15)
            rect((W-40.32)/2,0,40.32,16.5); rect((W-18.32)/2,0,18.32,5.5)
            rect((W-40.32)/2,H-16.5,40.32,16.5); rect((W-18.32)/2,H-5.5,18.32,5.5)
            ax.scatter([W/2,W/2],[11,H-11],c=lc,s=10,zorder=lz)
            ax.add_patch(Arc((W/2,11),18.3,18.3,angle=0,theta1=40,theta2=140,color=lc,lw=lw,zorder=lz))
            ax.add_patch(Arc((W/2,H-11),18.3,18.3,angle=0,theta1=220,theta2=320,color=lc,lw=lw,zorder=lz))
            rect((W-7.32)/2,-1.5,7.32,1.5); rect((W-7.32)/2,H,7.32,1.5)
            ax.set_xlim(0,W); ax.set_ylim(0,H)
        else:
            ax.plot([W/2,W/2],[0,H],color=lc,lw=lw,zorder=lz)
            circle(W/2,H/2,9.15)
            rect(0,(H-40.32)/2,16.5,40.32); rect(0,(H-18.32)/2,5.5,18.32)
            rect(W-16.5,(H-40.32)/2,16.5,40.32); rect(W-5.5,(H-18.32)/2,5.5,18.32)
            ax.scatter([11,W-11],[H/2,H/2],c=lc,s=10,zorder=lz)
            ax.add_patch(Arc((11,H/2),18.3,18.3,angle=0,theta1=310,theta2=50,color=lc,lw=lw,zorder=lz))
            ax.add_patch(Arc((W-11,H/2),18.3,18.3,angle=0,theta1=130,theta2=230,color=lc,lw=lw,zorder=lz))
            rect(-1.5,H/2-3.66,1.5,7.32); rect(W,H/2-3.66,1.5,7.32)
            ax.set_xlim(0,W); ax.set_ylim(0,H)
        return ax

    def scatter(self,x,y,ax,**kw): return ax.scatter(x,y,**kw)
    def kdeplot(self,x,y,ax,fill=True,levels=40,alpha=0.72,cmap="Blues"):
        if self.vertical: return ax.hist2d(y,x,bins=[22,14],range=[[0,self.pl],[0,self.pw]],cmap=cmap,alpha=alpha)
        return ax.hist2d(x,y,bins=[22,14],range=[[0,self.pl],[0,self.pw]],cmap=cmap,alpha=alpha)

# =========================================================
# STYLE UTILS (unchanged)
# =========================================================
def resolve_style(tn, ov=None): return build_chart_style(tn, ov or {})

def apply_rcparams(s):
    mpl.rcParams["font.family"]=s["font_family"]; mpl.rcParams["axes.titlesize"]=s["title_size"]
    mpl.rcParams["axes.labelsize"]=s["label_size"]; mpl.rcParams["xtick.labelsize"]=s["tick_size"]
    mpl.rcParams["ytick.labelsize"]=s["tick_size"]; mpl.rcParams["legend.fontsize"]=s["legend_size"]

def make_pitch(s, vertical=False):
    stripe=bool(s.get("pitch_stripe"))
    if MplsoccerPitch:
        try:
            cls=VerticalPitch if (vertical and VerticalPitch) else MplsoccerPitch
            return cls(pitch_type="custom",pitch_length=100,pitch_width=64,line_zorder=2,
                       linewidth=s["line_width"],pitch_color=s["pitch"],line_color=s["pitch_lines"],
                       stripe=stripe,stripe_color=s.get("pitch_stripe"))
        except: pass
    return SimplePitch(pitch_length=100,pitch_width=64,pitch_color=s["pitch"],
                       line_color=s["pitch_lines"],linewidth=s["line_width"],
                       stripe=stripe,stripe_color=s.get("pitch_stripe"),line_zorder=2,vertical=vertical)

def apply_flip_y(df,flip_y=False):
    out=df.copy()
    if not flip_y: return out
    for c in ["y","y2","y3"]:
        if c in out.columns: out[c]=64-pd.to_numeric(out[c],errors="coerce")
    return out

def themed_bar(ax,s):
    ax.set_facecolor(s["panel"])
    for sp in ax.spines.values(): sp.set_color(s["lines"]); sp.set_linewidth(1.0)
    ax.tick_params(colors=s["muted"],labelsize=s["tick_size"])
    ax.yaxis.label.set_color(s["muted"]); ax.xaxis.label.set_color(s["muted"]); ax.title.set_color(s["text"])
    if s.get("show_grid",True):
        ax.grid(axis="y",alpha=s["grid_alpha"],color=s["lines"],linestyle="--",lw=0.8); ax.set_axisbelow(True)

def style_pitch_axes(ax,s,vertical=False):
    ax.set_facecolor(s["pitch"]); px,py=s["pitch_pad_x"],s["pitch_pad_y"]
    if vertical: ax.set_xlim(-py,64+py); ax.set_ylim(-px,100+px)
    else: ax.set_xlim(-px,100+px); ax.set_ylim(-py,64+py)
    if not s.get("show_ticks",False): ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def chart_title(ax,t,s):
    if s.get("show_title",True): ax.set_title(t,color=s["text"],fontsize=s["title_size"],fontweight=s["title_weight"],pad=12)

def style_legend(leg,s):
    if not leg: return
    f=leg.get_frame()
    if f: f.set_facecolor(s["panel"]); f.set_edgecolor(s["lines"]); f.set_alpha(0.95)
    for t in leg.get_texts(): t.set_color(s["text"])

def fig_to_png_bytes(fig,dpi=260):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=dpi,bbox_inches="tight",pad_inches=0.25)
    buf.seek(0); return buf.getvalue()

def save_report_pdf(figures,filename="set_piece_report.pdf"):
    td=tempfile.mkdtemp(); pp=os.path.join(td,filename)
    with PdfPages(pp) as pdf:
        for f in figures: pdf.savefig(f,bbox_inches="tight",pad_inches=0.25)
    with open(pp,"rb") as f: return f.read()

def _base_fig(s,figsize=(8,6)):
    apply_rcparams(s); fig,ax=plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(s["bg"]); ax.set_facecolor(s["panel"]); return fig,ax

# =========================================================
# COORDINATE SCALING (unchanged)
# =========================================================
def _clean_num(df,cols):
    out=df.copy()
    for c in cols:
        if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")
    return out

def _auto_scale(df):
    out=_clean_num(df.copy(),["x","y","x2","y2","x3","y3"])
    # global y scale
    ay=pd.concat([out[c].dropna() for c in ["y","y2","y3"] if c in out.columns],ignore_index=True)
    if len(ay):
        gmy=ay.max()
        if pd.notna(gmy) and gmy>64.5:
            ys=64.0/gmy
            for c in ["y","y2","y3"]:
                if c in out.columns: out[c]=out[c]*ys
    # global x scale
    ax_=pd.concat([out[c].dropna() for c in ["x","x2","x3"] if c in out.columns],ignore_index=True)
    if len(ax_):
        gmx=ax_.max()
        if pd.notna(gmx) and gmx>100.5:
            xs=100.0/gmx
            for c in ["x","x2","x3"]:
                if c in out.columns: out[c]=out[c]*xs
    for c in ["x","x2","x3"]:
        if c in out.columns: out[c]=out[c].clip(0,100)
    for c in ["y","y2","y3"]:
        if c in out.columns: out[c]=out[c].clip(0,64)
    return out

# =========================================================
# DELIVERY HELPERS (unchanged except mapping fixes)
# =========================================================
def _cmap_delivery(s):
    cm=s.get("arrow_colors",{})
    return {"inswing":cm.get("inswing",s["accent"]),"outswing":cm.get("outswing",s["warning"]),
            "straight":cm.get("straight",s["accent_2"]),"driven":cm.get("driven",s["success"]),
            "short":cm.get("short",s["danger"])}

def _corner_anchor(x,y):
    if pd.isna(x) or pd.isna(y): return x,y,"unknown"
    side="right" if x>=50 else "left"
    half="top"   if y<=32  else "bottom"
    cx=99.5 if side=="right" else 0.5
    cy=0.5  if half=="top"  else 63.5
    return cx,cy,f"{side}_{half}"

def _curve_rad(dtype,corner_label):
    d=str(dtype).lower()
    if corner_label=="right_top":
        if d=="inswing": return 0.30
        if d=="outswing": return -0.30
    elif corner_label=="right_bottom":
        if d=="inswing": return -0.30
        if d=="outswing": return 0.30
    elif corner_label=="left_top":
        if d=="inswing": return -0.30
        if d=="outswing": return 0.30
    elif corner_label=="left_bottom":
        if d=="inswing": return 0.30
        if d=="outswing": return -0.30
    return 0.0

def _prep(df,flip_y=False):
    dff=apply_flip_y(df,flip_y); dff=_auto_scale(dff)
    if "x" in dff.columns and "y" in dff.columns:
        cd=dff.apply(lambda r: _corner_anchor(r.get("x"),r.get("y")),axis=1,result_type="expand")
        dff["x_start_plot"]=cd[0]; dff["y_start_plot"]=cd[1]; dff["corner_label"]=cd[2]
        dff["corner_side"]=dff["corner_label"].astype(str).str.split("_").str[0]
    else:
        dff["x_start_plot"]=dff.get("x"); dff["y_start_plot"]=dff.get("y")
        dff["corner_label"]="unknown"; dff["corner_side"]="unknown"
    return dff

def _arrow(ax,x1,y1,x2,y2,color,s,rad=0.0,lm=1.0,am=1.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",mutation_scale=s["trajectory_headwidth"]*4.0,
        linewidth=s["trajectory_width"]*lm,color=color,
        alpha=s["trajectory_alpha"]*am,clip_on=True,zorder=6))

def get_set_piece_series(df):
    for c in ["set_piece_type","event","outcome"]:
        if c in df.columns: return df[c].fillna("unknown").astype(str)
    return pd.Series(["unknown"]*len(df),index=df.index)

def get_target_zone_series(df):
    if "target_zone" in df.columns: return df["target_zone"].fillna("unknown").astype(str)
    if "x2" in df.columns and "y2" in df.columns:
        return df.apply(lambda r: _infer_zone(r.get("x2"),r.get("y2")),axis=1)
    return pd.Series(["unknown"]*len(df),index=df.index)

def _infer_zone(x,y):
    try: x,y=float(x),float(y)
    except: return "unknown"
    if y<BOX_Y0: return "near_post_short"
    if y>BOX_Y1: return "far_post_long"
    if x<BOX_X0: return "box_front"
    if x>=SIX_X0:
        if y<GOAL_Y0 or y>GOAL_Y1: return "small_area"
        return "small_area"
    if y<SIX_Y0: return "near_post"
    if y>SIX_Y1: return "far_post"
    return "penalty_spot"

def get_first_contact_series(df):
    if "first_contact_win" in df.columns: return bool01(df["first_contact_win"])
    return get_set_piece_series(df).str.lower().isin(["successful","win","won","first_contact"]).astype(int)

def get_second_ball_series(df):
    if "second_ball_win" in df.columns: return bool01(df["second_ball_win"])
    return get_set_piece_series(df).str.lower().isin(["second_ball_win","won_second_ball"]).astype(int)

# =========================================================
# ZONE OVERLAY  (perspective-aware)
# =========================================================
def _draw_zones(ax,s,corner_side="right",alpha=0.18,vertical=False):
    zones=_barca_zones(corner_side)
    for (label,zx,zy,zw,zh),ck in zip(zones,ZONE_CKEYS):
        color=s.get(ck,s["accent"])
        rx,ry,rw,rh=(zy,zx,zh,zw) if vertical else (zx,zy,zw,zh)
        ax.add_patch(Rectangle((rx,ry),rw,rh,facecolor=color,edgecolor=s["pitch_lines"],
                                linewidth=0.7,alpha=alpha,zorder=1))
        fs=max(s["tick_size"]-2,6)
        ax.text(rx+rw/2,ry+rh/2,label,ha="center",va="center",fontsize=fs,
                color=s["text"],alpha=0.92,zorder=2,linespacing=1.2)

def _draw_thirds(ax,s,vertical=False):
    lc=s.get("pitch_lines","#FFFFFF"); lw=max(s["line_width"]*0.75,0.9)
    fs=max(s["tick_size"]-1,7)
    for pos in [100/3,200/3]:
        if vertical: ax.axhline(pos,color=lc,lw=lw,alpha=0.55,linestyle="--",zorder=3)
        else: ax.axvline(pos,color=lc,lw=lw,alpha=0.55,linestyle="--",zorder=3)
    for pos,lbl in [(100/6,"Def Third"),(50,"Mid Third"),(500/6,"Att Third")]:
        if vertical: ax.text(64+s["pitch_pad_y"]*0.3,pos,lbl,ha="left",va="center",fontsize=fs,color=lc,alpha=0.70,zorder=4)
        else: ax.text(pos,64+s["pitch_pad_y"]*0.3,lbl,ha="center",va="bottom",fontsize=fs,color=lc,alpha=0.70,zorder=4)

# =========================================================
# TRAJECTORY CHART (left or right)
# =========================================================
def _traj_chart(df,theme_name,flip_y,style_overrides,title,corner_side):
    s=resolve_style(theme_name,style_overrides); vert=s.get("pitch_vertical",False)
    pitch=make_pitch(s,vert); dff=_prep(df,flip_y)
    # filter by side column in data
    if "side" in dff.columns and corner_side in ["left","right"]:
        mask=dff["side"].astype(str).str.lower()==corner_side
        if mask.any(): dff=dff[mask].copy()
    elif "corner_side" in dff.columns and corner_side in ["left","right"]:
        mask=dff["corner_side"]==corner_side
        if mask.any(): dff=dff[mask].copy()

    fig,ax=_base_fig(s,(6.4,8.4) if vert else (8.4,6.4))
    pitch.draw(ax=ax); style_pitch_axes(ax,s,vert)
    if s.get("show_thirds",False): _draw_thirds(ax,s,vert)
    _draw_zones(ax,s,corner_side,vertical=vert)

    dd=dff.dropna(subset=["x_start_plot","y_start_plot","x2","y2"]).copy()
    cmap=_cmap_delivery(s)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt,grp in dd.groupby("delivery_type"):
            dl=str(dt).lower()
            for _,r in grp.iterrows():
                rad=_curve_rad(dl,str(r.get("corner_label","unknown")))
                # mapping corrected: for horizontal pitch x->x, y->y; for vertical swap
                x1=r["x_start_plot"] if not vert else r["y_start_plot"]
                y1=r["y_start_plot"] if not vert else r["x_start_plot"]
                x2=r["x2"] if not vert else r["y2"]
                y2=r["y2"] if not vert else r["x2"]
                _arrow(ax,x1,y1,x2,y2,cmap.get(dl,s["accent"]),s,rad=rad)
        if s.get("show_legend",True):
            h,l=[],[]
            for k,v in cmap.items():
                if (dd["delivery_type"].astype(str).str.lower()==k).any():
                    h.append(Line2D([0],[0],color=v,lw=s["trajectory_width"]+1)); l.append(k.title())
            if h: leg=ax.legend(h,l,frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3); style_legend(leg,s)
    else:
        for _,r in dd.iterrows():
            x1=r["x_start_plot"] if not vert else r["y_start_plot"]
            y1=r["y_start_plot"] if not vert else r["x_start_plot"]
            x2=r["x2"] if not vert else r["y2"]
            y2=r["y2"] if not vert else r["x2"]
            _arrow(ax,x1,y1,x2,y2,s["accent"],s,rad=0.0)

    chart_title(ax,title,s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_trajectories_left(df,tn,flip_y=False,ov=None):
    return _traj_chart(df,tn,flip_y,ov,"Delivery Trajectories — Left Corner","left")
def chart_delivery_trajectories_right(df,tn,flip_y=False,ov=None):
    return _traj_chart(df,tn,flip_y,ov,"Delivery Trajectories — Right Corner","right")

# =========================================================
# DELIVERY END SCATTER  (left/right split, full pitch)
# Fixed mapping bug so x2,y2 appear correctly; color control added
# =========================================================
def _scatter_chart(df,theme_name,flip_y,style_overrides,corner_side):
    s=resolve_style(theme_name,style_overrides); vert=s.get("pitch_vertical",False)
    pitch=make_pitch(s,vert); dff=_prep(df,flip_y)

    # filter by side
    if "side" in dff.columns and corner_side in ["left","right"]:
        mask=dff["side"].astype(str).str.lower()==corner_side
        if mask.any(): dff=dff[mask].copy()
    elif "corner_side" in dff.columns and corner_side in ["left","right"]:
        mask=dff["corner_side"]==corner_side
        if mask.any(): dff=dff[mask].copy()

    # Use full pitch so points are contextualised like the Ball Receiving Map
    figsize=(6,8) if vert else (10,6.5)
    fig,ax=_base_fig(s,figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax,s,vert)
    if s.get("show_thirds",False): _draw_thirds(ax,s,vert)
    _draw_zones(ax,s,corner_side,alpha=0.14,vertical=vert)

    dd=dff.dropna(subset=["x2","y2"]).copy()
    title=f"Delivery End Scatter — {'Left' if corner_side=='left' else 'Right'} Corner"

    if not len(dd):
        chart_title(ax,title,s)
        if s["tight_layout"]: fig.tight_layout()
        return fig

    cmap=_cmap_delivery(s)
    scatter_color=s.get("scatter_dot_color",s["accent"])

    # corrected mapping: horizontal -> x2,y2 ; vertical -> y2,x2
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dt,grp in dd.groupby("delivery_type"):
            px=grp["x2"] if not vert else grp["y2"]
            py=grp["y2"] if not vert else grp["x2"]
            c=cmap.get(str(dt).lower(),scatter_color)
            pitch.scatter(px,py,ax=ax,s=s["marker_size"]*1.1,
                          color=c,edgecolors=s["pitch_lines"],
                          linewidth=s["marker_edge_width"],
                          label=str(dt).title(),alpha=s["alpha"],zorder=7)
        if s.get("show_legend",True):
            leg=ax.legend(frameon=True,loc="upper center",bbox_to_anchor=(0.5,-0.03),ncol=3)
            style_legend(leg,s)
    else:
        px=dd["x2"] if not vert else dd["y2"]
        py=dd["y2"] if not vert else dd["x2"]
        pitch.scatter(px,py,ax=ax,s=s["marker_size"]*1.1,
                      color=scatter_color,edgecolors=s["pitch_lines"],
                      linewidth=s["marker_edge_width"],alpha=s["alpha"],zorder=7)

    # count per zone overlay (labels)
    zones=_barca_zones(corner_side)
    for (label,zx,zy,zw,zh) in zones:
        if vert: cx_=zy+zh/2; cy_=zx+zw/2
        else: cx_=zx+zw/2; cy_=zy+zh/2
        in_zone=dd[(dd["x2"]>=zx)&(dd["x2"]<zx+zw)&(dd["y2"]>=zy)&(dd["y2"]<zy+zh)]
        if len(in_zone):
            ax.text(cx_,cy_+0.5,str(len(in_zone)),ha="center",va="center",
                    fontsize=max(s["tick_size"],9),fontweight="bold",
                    color=s["text"],zorder=10,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor=s["bg"],edgecolor="none",alpha=0.6))

    chart_title(ax,title,s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_delivery_end_scatter_left(df,tn,flip_y=False,ov=None):
    return _scatter_chart(df,tn,flip_y,ov,"left")
def chart_delivery_end_scatter_right(df,tn,flip_y=False,ov=None):
    return _scatter_chart(df,tn,flip_y,ov,"right")

# =========================================================
# ZONE DELIVERY COUNT MAP  (unchanged except small fixes)
# =========================================================
def _zone_count_map(df,theme_name,flip_y,style_overrides,corner_side):
    s=resolve_style(theme_name,style_overrides)
    vert=s.get("pitch_vertical",False)
    pitch=make_pitch(s,vert); dff=_prep(df,flip_y)

    # filter by corner side
    if "side" in dff.columns and corner_side in ["left","right"]:
        mask=dff["side"].astype(str).str.lower()==corner_side
        if mask.any(): dff=dff[mask].copy()

    figsize=(6,7) if vert else (9,6)
    fig,ax=_base_fig(s,figsize)
    pitch.draw(ax=ax); style_pitch_axes(ax,s,vert)

    # zoom to the attacking area
    if not vert:
        ax.set_xlim(65,102); ax.set_ylim(-2,66)
    else:
        ax.set_xlim(-2,66); ax.set_ylim(65,102)

    _draw_thirds(ax,s,vert)
    zones=_barca_zones(corner_side)
    dd=dff.dropna(subset=["x2","y2"]).copy()
    total=max(len(dd),1)

    # compute counts per zone
    zone_counts={}
    for label,zx,zy,zw,zh in zones:
        mask=(dd["x2"]>=zx)&(dd["x2"]<zx+zw)&(dd["y2"]>=zy)&(dd["y2"]<zy+zh)
        zone_counts[(zx,zy,zw,zh,label)]=int(mask.sum())

    max_count=max(zone_counts.values()) if zone_counts else 1

    for (zx,zy,zw,zh,label),cnt in zone_counts.items():
        intensity=cnt/max(max_count,1)
        color=s.get(ZONE_CKEYS[zones.index((label,zx,zy,zw,zh))] if (label,zx,zy,zw,zh) in zones else "accent",s["accent"])
        if vert: rx,ry,rw,rh=zy,zx,zh,zw
        else: rx,ry,rw,rh=zx,zy,zw,zh

        ax.add_patch(Rectangle((rx,ry),rw,rh,facecolor=color,edgecolor=s["pitch_lines"],
                                linewidth=0.8,alpha=0.10+intensity*0.65,zorder=1))
        cx_=rx+rw/2; cy_=ry+rh/2
        pct=cnt/total*100
        fs=max(s["tick_size"]-1,7)
        ax.text(cx_,cy_+rh*0.22,label.replace("\n"," "),ha="center",va="center",
                fontsize=max(fs-1,5),color=s["muted"],alpha=0.85,zorder=3)
        ax.text(cx_,cy_-rh*0.05,str(cnt),ha="center",va="center",
                fontsize=max(fs+3,10),fontweight="bold",color=s["text"],zorder=4)
        ax.text(cx_,cy_-rh*0.30,f"{pct:.0f}%",ha="center",va="center",
                fontsize=max(fs-1,6),color=s["muted"],zorder=4)

    # avg players in box annotation (bottom of chart)
    player_cols=["players_near_post","players_far_post","players_6yard","players_penalty","players_box"]
    avg_players=None
    for c in player_cols:
        if c in dff.columns and dff[c].notna().any():
            avg_players=dff[c].mean(); break
    if avg_players is None:
        total_in_box=len(dd[(dd["x2"]>=BOX_X0)&(dd["y2"]>=BOX_Y0)&(dd["y2"]<=BOX_Y1)])
        avg_players=round(total_in_box/max(total,1)*5,1)

    # draw the circle badge
    ax.add_patch(plt.Circle((84,63),2.2,facecolor=s["danger"],edgecolor=s["pitch_lines"],linewidth=1,zorder=5))
    ax.text(84,63,f"{avg_players:.1f}",ha="center",va="center",fontsize=max(s["tick_size"],9),
            fontweight="bold",color="white",zorder=6)
    ax.text(84,60.2,"Avg. players\nin box",ha="center",va="top",fontsize=max(s["tick_size"]-2,6),
            color=s["muted"],zorder=6)

    side_label="Right Side Corners" if corner_side=="right" else "Left Side Corners"
    chart_title(ax,side_label,s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

def chart_zone_count_left(df,tn,flip_y=False,ov=None):  return _zone_count_map(df,tn,flip_y,ov,"left")
def chart_zone_count_right(df,tn,flip_y=False,ov=None): return _zone_count_map(df,tn,flip_y,ov,"right")

# =========================================================
# ZONE PLAYER STATS (new)
# Compute total deliveries per zone, percent, and average players (sum and mean)
# =========================================================
def compute_zone_player_stats(df, corner_side="right"):
    """
    Returns a DataFrame with columns:
      zone_label, deliveries, pct_of_total, avg_players_in_zone
    Logic:
      - If explicit player columns exist (players_near_post, players_far_post, players_6yard, players_penalty, players_box),
        use them to compute averages.
      - Otherwise, compute deliveries per zone and estimate avg players in box using heuristic.
    """
    dff=_prep(df.copy(), flip_y=False)
    dd=dff.dropna(subset=["x2","y2"]).copy()
    zones=_barca_zones(corner_side)
    total=len(dd)
    rows=[]
    # try to find per-zone player columns mapping by name
    player_cols = {
        "Near Post Short": "players_near_post_short",
        "Near Post": "players_near_post",
        "Small Area": "players_6yard",
        "Penalty Spot": "players_penalty",
        "Far Post": "players_far_post",
        "Far Post Long": "players_far_post_long",
        "Box Front": "players_box_front",
    }
    # detect available player columns
    available_player_cols = {k:v for k,v in player_cols.items() if v in dd.columns}

    for label,zx,zy,zw,zh in zones:
        mask=(dd["x2"]>=zx)&(dd["x2"]<zx+zw)&(dd["y2"]>=zy)&(dd["y2"]<zy+zh)
        cnt=int(mask.sum())
        pct = (cnt/total*100) if total>0 else 0.0
        avg_players=None
        # if explicit column exists for this label, use mean
        if label in available_player_cols:
            col = available_player_cols[label]
            avg_players = float(dd.loc[mask, col].mean()) if cnt>0 else 0.0
        rows.append({"zone_label": label, "deliveries": cnt, "pct_of_total": pct, "avg_players_in_zone": avg_players})
    df_stats = pd.DataFrame(rows)
    # if avg_players_in_zone all None, compute an overall avg players in box
    if df_stats["avg_players_in_zone"].isna().all():
        # try players_box or players_in_box
        if "players_box" in dd.columns:
            overall_avg = float(dd["players_box"].mean())
        else:
            total_in_box = len(dd[(dd["x2"]>=BOX_X0)&(dd["y2"]>=BOX_Y0)&(dd["y2"]<=BOX_Y1)])
            overall_avg = round(total_in_box/max(total,1)*5,1)
        df_stats["avg_players_in_zone"] = overall_avg
    return df_stats

def chart_zone_player_stats(df, theme_name, flip_y=False, style_overrides=None, corner_side="right"):
    s=resolve_style(theme_name, style_overrides or {})
    vert=s.get("pitch_vertical",False)
    stats = compute_zone_player_stats(df, corner_side=corner_side)
    # create a compact figure showing numbers similar to second pic
    fig,ax=_base_fig(s,(6,6))
    ax.set_facecolor(s["panel"])
    ax.axis("off")
    # layout: two columns of zones + big badge for avg players in box
    left = stats.iloc[:5]
    right = stats.iloc[5:]
    # text positions
    start_y = 0.85
    dy = 0.14
    x_left = 0.12
    x_right = 0.55
    fs = max(s["tick_size"]+2,10)
    for i, row in left.reset_index().iterrows():
        ax.text(x_left, start_y - i*dy, f"{row['zone_label']}: {int(row['deliveries'])}", fontsize=fs, color=s["text"], transform=fig.transFigure)
        ax.text(x_left+0.28, start_y - i*dy, f"{row['pct_of_total']:.0f}%", fontsize=fs-2, color=s["muted"], transform=fig.transFigure)
    for i, row in right.reset_index().iterrows():
        ax.text(x_right, start_y - i*dy, f"{row['zone_label']}: {int(row['deliveries'])}", fontsize=fs, color=s["text"], transform=fig.transFigure)
        ax.text(x_right+0.28, start_y - i*dy, f"{row['pct_of_total']:.0f}%", fontsize=fs-2, color=s["muted"], transform=fig.transFigure)
    # big badge: average players in box (use mean of avg_players_in_zone)
    avg_players = stats["avg_players_in_zone"].mean()
    ax.add_patch(plt.Circle((0.85,0.25),0.08,transform=fig.transFigure,facecolor=s["danger"],edgecolor=s["pitch_lines"],linewidth=1))
    ax.text(0.85,0.25,f"{avg_players:.1f}",ha="center",va="center",fontsize=fs+2,color="white",transform=fig.transFigure)
    ax.text(0.85,0.16,"Avg. players\nin box",ha="center",va="top",fontsize=fs-2,color=s["muted"],transform=fig.transFigure)
    chart_title(ax, f"{'Right' if corner_side=='right' else 'Left'} Side Corners", s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# =========================================================
# REMAINING STANDARD CHARTS (unchanged except mapping fix in start map)
# =========================================================
def chart_delivery_start_map(df,tn,flip_y=False,ov=None):
    s=resolve_style(tn,ov); vert=s.get("pitch_vertical",False)
    pitch=make_pitch(s,vert); dff=_prep(df,flip_y)
    fig,ax=_base_fig(s,(6,8) if vert else (8,6))
    pitch.draw(ax=ax); style_pitch_axes(ax,s,vert)
    if s.get("show_thirds",False): _draw_thirds(ax,s,vert)
    dd=dff.dropna(subset=["x_start_plot","y_start_plot"]).copy()
    px=dd["x_start_plot"] if not vert else dd["y_start_plot"]
    py=dd["y_start_plot"] if not vert else dd["x_start_plot"]
    pitch.scatter(px,py,ax=ax,s=s["marker_size"],color=s["accent"],
                  edgecolors=s["pitch_lines"],linewidth=s["marker_edge_width"],alpha=s["alpha"])
    chart_title(ax,"Delivery Start Map",s)
    if s["tight_layout"]: fig.tight_layout()
    return fig

# (other chart functions like heatmap left as-is or truncated for brevity)
# You should keep the rest of your original chart functions here unchanged.
# Ensure CHART_BUILDERS mapping in your app imports the new functions names:
# e.g. CHART_BUILDERS = {
#   "Delivery Trajectories - Left Corner": chart_delivery_trajectories_left,
#   "Delivery Trajectories - Right Corner": chart_delivery_trajectories_right,
#   "Delivery End Scatter - Left Corner": chart_delivery_end_scatter_left,
#   "Delivery End Scatter - Right Corner": chart_delivery_end_scatter_right,
#   "Zone Player Stats": chart_zone_player_stats,
#   ...
# }
