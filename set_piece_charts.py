import io
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

from data_utils import bool01
from ui_theme import build_chart_style

# =========================================================
# OPTIONAL IMPORTS (SAFE FALLBACKS)
# =========================================================
try:
    from mplsoccer import Pitch as MplsoccerPitch
except Exception:
    MplsoccerPitch = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# =========================================================
# SIMPLE PITCH FALLBACK
# =========================================================
class SimplePitch:
    def __init__(
        self,
        pitch_length=100,
        pitch_width=64,
        pitch_color="#1F5F3B",
        line_color="#FFFFFF",
        linewidth=1.4,
        stripe=False,
        stripe_color=None,
        line_zorder=2,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.linewidth = linewidth
        self.stripe = stripe
        self.stripe_color = stripe_color
        self.line_zorder = line_zorder

    def draw(self, ax):
        ax.set_facecolor(self.pitch_color)

        if self.stripe and self.stripe_color:
            stripe_w = self.pitch_length / 10.0
            for i in range(10):
                if i % 2 == 0:
                    ax.add_patch(
                        Rectangle(
                            (i * stripe_w, 0),
                            stripe_w,
                            self.pitch_width,
                            facecolor=self.stripe_color,
                            edgecolor="none",
                            alpha=0.35,
                            zorder=0,
                        )
                    )

        ax.add_patch(
            Rectangle(
                (0, 0),
                self.pitch_length,
                self.pitch_width,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        ax.plot(
            [self.pitch_length / 2, self.pitch_length / 2],
            [0, self.pitch_width],
            color=self.line_color,
            lw=self.linewidth,
            zorder=self.line_zorder,
        )

        center_x = self.pitch_length / 2
        center_y = self.pitch_width / 2

        ax.add_patch(
            plt.Circle(
                (center_x, center_y),
                9.15,
                fill=False,
                color=self.line_color,
                lw=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        # Left boxes
        ax.add_patch(
            Rectangle(
                (0, (self.pitch_width - 40.32) / 2),
                16.5,
                40.32,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )
        ax.add_patch(
            Rectangle(
                (0, (self.pitch_width - 18.32) / 2),
                5.5,
                18.32,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        # Right boxes
        ax.add_patch(
            Rectangle(
                (self.pitch_length - 16.5, (self.pitch_width - 40.32) / 2),
                16.5,
                40.32,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )
        ax.add_patch(
            Rectangle(
                (self.pitch_length - 5.5, (self.pitch_width - 18.32) / 2),
                5.5,
                18.32,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        ax.scatter([11, self.pitch_length - 11], [center_y, center_y], c=self.line_color, s=8, zorder=self.line_zorder)

        ax.add_patch(
            Arc(
                (11, center_y),
                18.3,
                18.3,
                angle=0,
                theta1=310,
                theta2=50,
                color=self.line_color,
                lw=self.linewidth,
                zorder=self.line_zorder,
            )
        )
        ax.add_patch(
            Arc(
                (self.pitch_length - 11, center_y),
                18.3,
                18.3,
                angle=0,
                theta1=130,
                theta2=230,
                color=self.line_color,
                lw=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        goal_depth = 1.5
        goal_width = 7.32
        ax.add_patch(
            Rectangle(
                (-goal_depth, center_y - goal_width / 2),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )
        ax.add_patch(
            Rectangle(
                (self.pitch_length, center_y - goal_width / 2),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor=self.line_color,
                linewidth=self.linewidth,
                zorder=self.line_zorder,
            )
        )

        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        return ax

    def scatter(self, x, y, ax, **kwargs):
        return ax.scatter(x, y, **kwargs)

    def kdeplot(self, x, y, ax, fill=True, levels=40, alpha=0.72, cmap="Blues"):
        return ax.hist2d(
            x,
            y,
            bins=[22, 14],
            range=[[0, self.pitch_length], [0, self.pitch_width]],
            cmap=cmap,
            alpha=alpha,
        )


# =========================================================
# CHART CONFIG
# =========================================================
CHART_REQUIREMENTS: Dict[str, List[str]] = {
    "Delivery Start Map": ["x", "y"],
    "Delivery Heatmap": ["x2", "y2"],
    "Delivery End Scatter": ["x2", "y2"],
    "Delivery Trajectories": ["x", "y", "x2", "y2"],
    "Average Delivery Path": ["x", "y", "x2", "y2"],
    "Heat + Trajectories": ["x", "y", "x2", "y2"],
    "Trajectory Clusters": ["x", "y", "x2", "y2"],
    "Delivery Length Distribution": ["x", "y", "x2", "y2"],
    "Delivery Direction Map": ["x", "y", "x2", "y2"],
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


# =========================================================
# STYLE + RC
# =========================================================
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

    if MplsoccerPitch is not None:
        return MplsoccerPitch(
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

    return SimplePitch(
        pitch_length=100,
        pitch_width=64,
        pitch_color=style["pitch"],
        line_color=style["pitch_lines"],
        linewidth=style["line_width"],
        stripe=stripe,
        stripe_color=style.get("pitch_stripe"),
        line_zorder=2,
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

    ax.tick_params(colors=style["muted"], labelsize=style["tick_size"])
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

    if not style.get("show_ticks", False):
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


def _base_figure(style: dict, figsize=(8, 6)):
    apply_global_rcparams(style)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(style["bg"])
    ax.set_facecolor(style["panel"])
    return fig, ax


# =========================================================
# DATA HELPERS
# =========================================================
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


def _clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _auto_scale_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    لو y / y2 / y3 جاية 0-100 بدل 0-64 نعمل scaling تلقائي.
    وكمان نعمل clip عشان مفيش نقطة تطلع برا الملعب.
    """
    out = df.copy()
    out = _clean_numeric(out, ["x", "y", "x2", "y2", "x3", "y3"])

    # scale Y if looks like 0-100 space
    for yc in ["y", "y2", "y3"]:
        if yc in out.columns and out[yc].notna().any():
            max_y = out[yc].max()
            if pd.notna(max_y) and max_y > 64.5:
                out[yc] = out[yc] * 64.0 / 100.0

    # optional scale X if somehow > 100 and close to 120-like providers
    for xc in ["x", "x2", "x3"]:
        if xc in out.columns and out[xc].notna().any():
            max_x = out[xc].max()
            if pd.notna(max_x) and max_x > 100.5 and max_x <= 121:
                out[xc] = out[xc] * 100.0 / max_x

    # clip to pitch bounds
    for xc in ["x", "x2", "x3"]:
        if xc in out.columns:
            out[xc] = out[xc].clip(lower=0, upper=100)

    for yc in ["y", "y2", "y3"]:
        if yc in out.columns:
            out[yc] = out[yc].clip(lower=0, upper=64)

    return out


def _get_delivery_color_map(style: dict):
    return {
        "inswing": style["accent"],
        "outswing": style["warning"],
        "straight": style["accent_2"],
        "driven": style["success"],
        "short": style["danger"],
    }


def _corner_anchor(x: float, y: float) -> Tuple[float, float, str]:
    """
    Normalize start point to nearest logical corner if event starts near corners.
    Returns x, y, side_label
    """
    if pd.isna(x) or pd.isna(y):
        return x, y, "unknown"

    side = "right" if x >= 50 else "left"
    half = "top" if y <= 32 else "bottom"

    # force clear corner starts to exact corners
    if side == "right":
        x = 100
    else:
        x = 0

    y = 0 if half == "top" else 64
    return x, y, f"{side}_{half}"


def _curve_rad_for_delivery(dtype: str, corner_label: str) -> float:
    """
    rad positive / negative controls curve direction.
    inswing = لجوه
    outswing = لبرة
    """
    dtype = str(dtype).lower()

    # right_top / right_bottom / left_top / left_bottom
    if corner_label == "right_top":
        if dtype == "inswing":
            return -0.22
        if dtype == "outswing":
            return 0.22
    elif corner_label == "right_bottom":
        if dtype == "inswing":
            return 0.22
        if dtype == "outswing":
            return -0.22
    elif corner_label == "left_top":
        if dtype == "inswing":
            return 0.22
        if dtype == "outswing":
            return -0.22
    elif corner_label == "left_bottom":
        if dtype == "inswing":
            return -0.22
        if dtype == "outswing":
            return 0.22

    return 0.0


def _prepare_delivery_df(df: pd.DataFrame, flip_y: bool = False) -> pd.DataFrame:
    dff = apply_flip_y(df, flip_y)
    dff = _auto_scale_coordinates(dff)

    needed = [c for c in ["x", "y", "x2", "y2", "delivery_type"] if c in dff.columns]
    dff = dff.copy()

    if "x" in dff.columns and "y" in dff.columns:
        corner_data = dff.apply(
            lambda r: _corner_anchor(r.get("x"), r.get("y")),
            axis=1,
            result_type="expand",
        )
        dff["x_start_plot"] = corner_data[0]
        dff["y_start_plot"] = corner_data[1]
        dff["corner_label"] = corner_data[2]
    else:
        dff["x_start_plot"] = dff.get("x")
        dff["y_start_plot"] = dff.get("y")
        dff["corner_label"] = "unknown"

    if "x2" in dff.columns:
        dff["x2"] = dff["x2"].clip(0, 100)
    if "y2" in dff.columns:
        dff["y2"] = dff["y2"].clip(0, 64)

    return dff


def _draw_curved_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    style: dict,
    rad: float = 0.0,
    linewidth_mult: float = 1.0,
    alpha_mult: float = 1.0,
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=style["trajectory_headwidth"] * 4.0,
        linewidth=style["trajectory_width"] * linewidth_mult,
        color=color,
        alpha=style["trajectory_alpha"] * alpha_mult,
    )
    ax.add_patch(arrow)


def _safe_cluster_labels(dd: pd.DataFrame, n_clusters: int = 3) -> pd.Series:
    features = dd[["x_start_plot", "y_start_plot", "x2", "y2"]].dropna().copy()
    if len(features) < max(n_clusters, 3):
        return pd.Series([0] * len(dd), index=dd.index)

    if KMeans is not None:
        try:
            n = min(n_clusters, max(2, len(features)))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = km.fit_predict(features)
            out = pd.Series(index=features.index, data=labels)
            return out.reindex(dd.index).fillna(0).astype(int)
        except Exception:
            pass

    # fallback بسيط بدون sklearn
    bins_x = pd.qcut(features["x2"], q=min(3, len(features)), duplicates="drop", labels=False)
    bins_y = pd.qcut(features["y2"], q=min(3, len(features)), duplicates="drop", labels=False)
    labels = (bins_x.fillna(0).astype(int) * 10 + bins_y.fillna(0).astype(int))
    out = pd.Series(index=features.index, data=labels)
    return out.reindex(dd.index).fillna(0).astype(int)


# =========================================================
# CHARTS
# =========================================================
def chart_delivery_start_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x_start_plot", "y_start_plot"]).copy()

    pitch.scatter(
        dd["x_start_plot"],
        dd["y_start_plot"],
        ax=ax,
        s=style["marker_size"],
        color=style["accent"],
        edgecolors=style["pitch_lines"],
        linewidth=style["marker_edge_width"],
        alpha=style["alpha"],
    )

    set_chart_title(ax, "Delivery Start Map", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_delivery_heatmap(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

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
                cmap=style["heatmap_cmap"],
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
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8, 6))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x2", "y2"]).copy()

    if "delivery_type" in dd.columns:
        color_map = _get_delivery_color_map(style)
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


def chart_delivery_trajectories(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8.4, 6.4))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()
    color_map = _get_delivery_color_map(style)

    if "delivery_type" in dd.columns:
        for dtype, grp in dd.groupby("delivery_type"):
            dtype_l = str(dtype).lower()
            for _, r in grp.iterrows():
                rad = _curve_rad_for_delivery(dtype_l, str(r.get("corner_label", "unknown")))
                _draw_curved_arrow(
                    ax=ax,
                    x1=r["x_start_plot"],
                    y1=r["y_start_plot"],
                    x2=r["x2"],
                    y2=r["y2"],
                    color=color_map.get(dtype_l, style["accent"]),
                    style=style,
                    rad=rad,
                )

        if style.get("show_legend", True):
            handles = []
            labels = []
            for k, v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower() == k).any():
                    handles.append(Line2D([0], [0], color=v, lw=style["trajectory_width"] + 1))
                    labels.append(k.title())
            if handles:
                leg = ax.legend(handles, labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
                style_legend(leg, style)
    else:
        for _, r in dd.iterrows():
            _draw_curved_arrow(
                ax=ax,
                x1=r["x_start_plot"],
                y1=r["y_start_plot"],
                x2=r["x2"],
                y2=r["y2"],
                color=style["accent"],
                style=style,
                rad=0.0,
            )

    set_chart_title(ax, "Delivery Trajectories", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_average_delivery_path(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8.2, 6.2))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()

    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        color_map = _get_delivery_color_map(style)
        for dtype, grp in dd.groupby("delivery_type"):
            if len(grp) == 0:
                continue
            avg_x1 = grp["x_start_plot"].mean()
            avg_y1 = grp["y_start_plot"].mean()
            avg_x2 = grp["x2"].mean()
            avg_y2 = grp["y2"].mean()
            rad = _curve_rad_for_delivery(str(dtype).lower(), str(grp["corner_label"].mode().iloc[0]) if grp["corner_label"].notna().any() else "unknown")

            _draw_curved_arrow(
                ax=ax,
                x1=avg_x1,
                y1=avg_y1,
                x2=avg_x2,
                y2=avg_y2,
                color=color_map.get(str(dtype).lower(), style["accent"]),
                style=style,
                rad=rad,
                linewidth_mult=2.2,
                alpha_mult=1.15,
            )

            pitch.scatter([avg_x2], [avg_y2], ax=ax, s=style["marker_size"] * 1.2, color=color_map.get(str(dtype).lower(), style["accent"]), edgecolors=style["pitch_lines"], linewidth=1.2)

        if style.get("show_legend", True):
            handles = []
            labels = []
            for k, v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower() == k).any():
                    handles.append(Line2D([0], [0], color=v, lw=style["trajectory_width"] * 2))
                    labels.append(f"{k.title()} Avg")
            if handles:
                leg = ax.legend(handles, labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
                style_legend(leg, style)
    else:
        avg_x1 = dd["x_start_plot"].mean()
        avg_y1 = dd["y_start_plot"].mean()
        avg_x2 = dd["x2"].mean()
        avg_y2 = dd["y2"].mean()

        _draw_curved_arrow(
            ax=ax,
            x1=avg_x1,
            y1=avg_y1,
            x2=avg_x2,
            y2=avg_y2,
            color=style["accent"],
            style=style,
            rad=0.0,
            linewidth_mult=2.2,
            alpha_mult=1.1,
        )
        pitch.scatter([avg_x2], [avg_y2], ax=ax, s=style["marker_size"] * 1.25, color=style["accent"], edgecolors=style["pitch_lines"], linewidth=1.2)

    set_chart_title(ax, "Average Delivery Path", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_heat_plus_trajectories(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8.4, 6.4))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()

    # Heat first
    if len(dd):
        try:
            pitch.kdeplot(
                dd["x2"], dd["y2"],
                ax=ax,
                fill=True,
                levels=40,
                alpha=style["kde_alpha"] * 0.65,
                cmap=style["heatmap_cmap"],
            )
        except Exception:
            pass

    # Then trajectories فوق
    color_map = _get_delivery_color_map(style)
    if "delivery_type" in dd.columns and dd["delivery_type"].notna().any():
        for dtype, grp in dd.groupby("delivery_type"):
            dtype_l = str(dtype).lower()
            for _, r in grp.iterrows():
                rad = _curve_rad_for_delivery(dtype_l, str(r.get("corner_label", "unknown")))
                _draw_curved_arrow(
                    ax=ax,
                    x1=r["x_start_plot"],
                    y1=r["y_start_plot"],
                    x2=r["x2"],
                    y2=r["y2"],
                    color=color_map.get(dtype_l, style["accent"]),
                    style=style,
                    rad=rad,
                    linewidth_mult=0.9,
                    alpha_mult=0.9,
                )

        if style.get("show_legend", True):
            handles = []
            labels = []
            for k, v in color_map.items():
                if (dd["delivery_type"].astype(str).str.lower() == k).any():
                    handles.append(Line2D([0], [0], color=v, lw=style["trajectory_width"] + 0.6))
                    labels.append(k.title())
            if handles:
                leg = ax.legend(handles, labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
                style_legend(leg, style)
    else:
        for _, r in dd.iterrows():
            _draw_curved_arrow(
                ax=ax,
                x1=r["x_start_plot"],
                y1=r["y_start_plot"],
                x2=r["x2"],
                y2=r["y2"],
                color=style["accent"],
                style=style,
                rad=0.0,
                linewidth_mult=0.9,
                alpha_mult=0.9,
            )

    set_chart_title(ax, "Heat + Trajectories", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_trajectory_clusters(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

    fig, ax = _base_figure(style, figsize=(8.4, 6.4))
    pitch.draw(ax=ax)
    style_pitch_axes(ax, style)

    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()
    if len(dd) == 0:
        set_chart_title(ax, "Trajectory Clusters", style)
        if style["tight_layout"]:
            fig.tight_layout()
        return fig

    dd["cluster"] = _safe_cluster_labels(dd, n_clusters=3)
    cluster_palette = [style["accent"], style["warning"], style["success"], style["accent_2"], style["danger"]]

    handles = []
    labels = []

    for i, (cluster_id, grp) in enumerate(dd.groupby("cluster")):
        color = cluster_palette[i % len(cluster_palette)]

        # Draw light arrows لكل cluster
        for _, r in grp.iterrows():
            dtype_l = str(r.get("delivery_type", "")).lower()
            rad = _curve_rad_for_delivery(dtype_l, str(r.get("corner_label", "unknown")))
            _draw_curved_arrow(
                ax=ax,
                x1=r["x_start_plot"],
                y1=r["y_start_plot"],
                x2=r["x2"],
                y2=r["y2"],
                color=color,
                style=style,
                rad=rad,
                linewidth_mult=0.9,
                alpha_mult=0.7,
            )

        # Average path for cluster
        avg_x1 = grp["x_start_plot"].mean()
        avg_y1 = grp["y_start_plot"].mean()
        avg_x2 = grp["x2"].mean()
        avg_y2 = grp["y2"].mean()
        mode_corner = grp["corner_label"].mode().iloc[0] if grp["corner_label"].notna().any() else "unknown"
        mode_dtype = grp["delivery_type"].mode().iloc[0] if ("delivery_type" in grp.columns and grp["delivery_type"].notna().any()) else ""
        avg_rad = _curve_rad_for_delivery(str(mode_dtype).lower(), str(mode_corner))

        _draw_curved_arrow(
            ax=ax,
            x1=avg_x1,
            y1=avg_y1,
            x2=avg_x2,
            y2=avg_y2,
            color=color,
            style=style,
            rad=avg_rad,
            linewidth_mult=2.4,
            alpha_mult=1.15,
        )
        pitch.scatter([avg_x2], [avg_y2], ax=ax, s=style["marker_size"] * 1.25, color=color, edgecolors=style["pitch_lines"], linewidth=1.2)

        handles.append(Line2D([0], [0], color=color, lw=style["trajectory_width"] * 2))
        labels.append(f"Cluster {int(cluster_id) + 1} ({len(grp)})")

    if style.get("show_legend", True) and handles:
        leg = ax.legend(handles, labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
        style_legend(leg, style)

    set_chart_title(ax, "Trajectory Clusters", style)
    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_delivery_length_distribution(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6, 4.8))

    dff = _prepare_delivery_df(df, flip_y)
    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()

    if len(dd) == 0:
        lengths = pd.Series(dtype=float)
    else:
        lengths = ((dd["x2"] - dd["x_start_plot"]) ** 2 + (dd["y2"] - dd["y_start_plot"]) ** 2) ** 0.5

    ax.hist(lengths, bins=12, color=style["accent"], edgecolor=style["lines"], linewidth=0.8, alpha=0.92)
    themed_bar(ax, style)
    set_chart_title(ax, "Delivery Length Distribution", style)
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_delivery_direction_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    fig, ax = _base_figure(style, figsize=(7.6, 4.8))

    dff = _prepare_delivery_df(df, flip_y)
    dd = dff.dropna(subset=["x_start_plot", "y_start_plot", "x2", "y2"]).copy()

    if len(dd) == 0:
        summary = pd.Series(dtype=float)
    else:
        dx = dd["x2"] - dd["x_start_plot"]
        dy = dd["y2"] - dd["y_start_plot"]
        angles = dy.combine(dx, lambda yv, xv: math.degrees(math.atan2(yv, xv)))
        labels = pd.cut(
            angles,
            bins=[-181, -60, -10, 10, 60, 181],
            labels=["Down", "Down-In", "Straight", "Up-In", "Up"],
            include_lowest=True,
        )
        summary = labels.value_counts().reindex(["Down", "Down-In", "Straight", "Up-In", "Up"]).fillna(0)

    ax.bar(summary.index.astype(str), summary.values, color=style["accent_2"], edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Delivery Direction Map", style)
    ax.set_ylabel("Count")

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

    dff = _prepare_delivery_df(df, flip_y)
    counts = get_target_zone_series(dff).value_counts()

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

    dff = _prepare_delivery_df(df, flip_y)
    dd = dff.copy()
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

    ax.barh(counts.index[::-1], counts.values[::-1], color=style["warning"], edgecolor=style["lines"], linewidth=0.8)
    themed_bar(ax, style)
    set_chart_title(ax, "Routine Breakdown", style)
    ax.set_xlabel("Count")

    if style["tight_layout"]:
        fig.tight_layout()
    return fig


def chart_shot_map(df: pd.DataFrame, theme_name: str, flip_y: bool = False, style_overrides: Optional[dict] = None):
    style = resolve_style(theme_name, style_overrides)
    pitch = make_pitch(style)
    dff = _prepare_delivery_df(df, flip_y)

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
    dff = _prepare_delivery_df(df, flip_y)

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
    dff = _prepare_delivery_df(df, flip_y)

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
                cmap=style["heatmap_cmap"],
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

    ax.barh(seq_counts.index[::-1], seq_counts.values[::-1], color=style["accent"], edgecolor=style["lines"], linewidth=0.8)
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
    "Delivery Start Map": chart_delivery_start_map,
    "Delivery Heatmap": chart_delivery_heatmap,
    "Delivery End Scatter": chart_delivery_end_scatter,
    "Delivery Trajectories": chart_delivery_trajectories,
    "Average Delivery Path": chart_average_delivery_path,
    "Heat + Trajectories": chart_heat_plus_trajectories,
    "Trajectory Clusters": chart_trajectory_clusters,
    "Delivery Length Distribution": chart_delivery_length_distribution,
    "Delivery Direction Map": chart_delivery_direction_map,
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
