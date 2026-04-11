import streamlit as st
from typing import Dict, Optional


THEMES = {
    "The Athletic Dark": {
        "bg": "#0E1117",
        "panel": "#111827",
        "panel_2": "#0F172A",
        "pitch": "#1F5F3B",
        "pitch_stripe": None,
        "text": "#FFFFFF",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
        "accent": "#38BDF8",
        "accent_2": "#22C55E",
        "danger": "#FF4D4D",
        "warning": "#FFD400",
        "success": "#00FF6A",
    },
    "Opta Dark": {
        "bg": "#0E1117",
        "panel": "#141A22",
        "panel_2": "#101720",
        "pitch": "#1F5F3B",
        "pitch_stripe": None,
        "text": "#FFFFFF",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
        "accent": "#00C2FF",
        "accent_2": "#60A5FA",
        "danger": "#FF5A5F",
        "warning": "#FACC15",
        "success": "#22C55E",
    },
    "Sofa Light": {
        "bg": "#F7F9FC",
        "panel": "#FFFFFF",
        "panel_2": "#EEF2F7",
        "pitch": "#2F6B3A",
        "pitch_stripe": None,
        "text": "#111111",
        "muted": "#5A6572",
        "lines": "#DDE3EA",
        "goal": "#444444",
        "pitch_lines": "#FFFFFF",
        "accent": "#2563EB",
        "accent_2": "#06B6D4",
        "danger": "#DC2626",
        "warning": "#D97706",
        "success": "#16A34A",
    },
    "Black Stripe": {
        "bg": "#000000",
        "panel": "#000000",
        "panel_2": "#0A0A0A",
        "pitch": "#000000",
        "pitch_stripe": "#0A0A0A",
        "text": "#FFFFFF",
        "muted": "#B7B7B7",
        "lines": "#2A2A2A",
        "goal": "#FFFFFF",
        "pitch_lines": "#FFFFFF",
        "accent": "#38BDF8",
        "accent_2": "#A78BFA",
        "danger": "#FF4D4D",
        "warning": "#FFD400",
        "success": "#00FF6A",
    },
}

FONT_FAMILIES = [
    "DejaVu Sans",
    "Roboto Slab",
    "Roboto Slab Light",
    "Roboto Slab Bold",
    "Arial",
    "Helvetica",
    "Verdana",
    "Trebuchet MS",
    "Tahoma",
]

HEATMAP_STYLES = [
    "Blues",
    "Reds",
    "Greens",
    "Purples",
    "Oranges",
    "YlOrRd",
    "YlGnBu",
    "coolwarm",
    "magma",
    "viridis",
    "cividis",
]


def get_theme(theme_name: str) -> Dict:
    return THEMES.get(theme_name, THEMES["The Athletic Dark"]).copy()


def build_chart_style(theme_name: str, controls: Optional[Dict] = None) -> Dict:
    base = get_theme(theme_name)
    controls = controls or {}

    return {
        "theme_name": theme_name,
        "bg": controls.get("bg", base["bg"]),
        "panel": controls.get("panel", base["panel"]),
        "panel_2": controls.get("panel_2", base.get("panel_2", base["panel"])),
        "pitch": controls.get("pitch", base["pitch"]),
        "pitch_stripe": controls.get("pitch_stripe", base.get("pitch_stripe")),
        "text": controls.get("text", base["text"]),
        "muted": controls.get("muted", base["muted"]),
        "lines": controls.get("lines", base["lines"]),
        "goal": controls.get("goal", base["goal"]),
        "pitch_lines": controls.get("pitch_lines", base["pitch_lines"]),
        "accent": controls.get("accent", base["accent"]),
        "accent_2": controls.get("accent_2", base["accent_2"]),
        "danger": controls.get("danger", base["danger"]),
        "warning": controls.get("warning", base["warning"]),
        "success": controls.get("success", base["success"]),
        "font_family": controls.get("font_family", "DejaVu Sans"),
        "title_size": controls.get("title_size", 16),
        "label_size": controls.get("label_size", 11),
        "tick_size": controls.get("tick_size", 10),
        "legend_size": controls.get("legend_size", 10),
        "title_weight": controls.get("title_weight", "bold"),
        "line_width": controls.get("line_width", 1.4),
        "marker_size": controls.get("marker_size", 90),
        "marker_edge_width": controls.get("marker_edge_width", 1.2),
        "trajectory_width": controls.get("trajectory_width", 1.8),
        "trajectory_headwidth": controls.get("trajectory_headwidth", 4.5),
        "alpha": controls.get("alpha", 0.9),
        "trajectory_alpha": controls.get("trajectory_alpha", 0.75),
        "grid_alpha": controls.get("grid_alpha", 0.18),
        "kde_alpha": controls.get("kde_alpha", 0.72),
        "show_grid": controls.get("show_grid", True),
        "show_legend": controls.get("show_legend", True),
        "show_title": controls.get("show_title", True),
        "show_ticks": controls.get("show_ticks", False),
        "tight_layout": controls.get("tight_layout", True),
        "pitch_pad_x": controls.get("pitch_pad_x", 2),
        "pitch_pad_y": controls.get("pitch_pad_y", 2),
        "export_dpi": controls.get("export_dpi", 260),
        "heatmap_cmap": controls.get("heatmap_cmap", "Blues"),
        "pitch_vertical": controls.get("pitch_vertical", False),
        "show_thirds": controls.get("show_thirds", False),
        "scatter_dot_color": controls.get("scatter_dot_color", base["accent"]),
        "arrow_colors": controls.get("arrow_colors", {}),
        "bar_colors": controls.get("bar_colors", {}),
        "shirt_body_color": controls.get("shirt_body_color", base["accent"]),
        "shirt_sleeve_color": controls.get("shirt_sleeve_color", base["panel"]),
        "shirt_number_color": controls.get("shirt_number_color", base["bg"]),
    }


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;700&display=swap');

            :root {
                --bg: #0b1220;
                --card: #111827;
                --card-2: #0f172a;
                --border: #243041;
                --text: #f3f4f6;
                --muted: #9ca3af;
                --accent: #38bdf8;
                --accent-2: #22c55e;
                --input-bg: #f8fafc;
                --input-text: #0f172a;
                --sidebar-input-bg: #111827;
                --sidebar-input-text: #f8fafc;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56,189,248,0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(34,197,94,0.10), transparent 24%),
                    linear-gradient(180deg, #09111f 0%, #0b1220 48%, #0a1324 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.05rem;
                padding-bottom: 1.4rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
                max-width: 100%;
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(15,23,42,0.98), rgba(9,17,31,0.98)) !important;
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc !important;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] div[data-baseweb="input"] > div,
            [data-testid="stSidebar"] .stTextInput > div > div,
            [data-testid="stSidebar"] .stNumberInput > div > div,
            [data-testid="stSidebar"] .stColorPicker > div > div {
                background: var(--sidebar-input-bg) !important;
                color: var(--sidebar-input-text) !important;
                border: 1px solid #334155 !important;
                border-radius: 12px !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            .stTextInput > div > div,
            .stNumberInput > div > div,
            .stTextArea textarea {
                background: var(--input-bg) !important;
                color: var(--input-text) !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
            }

            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input,
            div[data-baseweb="input"] input,
            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                color: var(--input-text) !important;
                -webkit-text-fill-color: var(--input-text) !important;
            }

            div[data-baseweb="select"] svg,
            div[data-baseweb="input"] svg {
                fill: #334155 !important;
            }

            [data-baseweb="popover"] *,
            [role="listbox"] *,
            ul[role="listbox"] * {
                color: #0f172a !important;
            }

            [data-baseweb="popover"] {
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.22) !important;
            }

            [role="option"] {
                background: #ffffff !important;
                color: #0f172a !important;
            }

            [role="option"][aria-selected="true"] {
                background: #e0f2fe !important;
                color: #0f172a !important;
            }

            [role="option"]:hover {
                background: #eff6ff !important;
                color: #0f172a !important;
            }

            .stMultiSelect [data-baseweb="tag"] {
                background: rgba(56, 189, 248, 0.14) !important;
                border: 1px solid rgba(56, 189, 248, 0.24) !important;
                border-radius: 999px !important;
            }

            .stMultiSelect [data-baseweb="tag"] * {
                color: #0f172a !important;
            }

            h1, h2, h3, h4, h5, h6, p, span, div, label {
                color: var(--text);
            }

            .app-header {
                background:
                    linear-gradient(135deg, rgba(56,189,248,0.18), rgba(16,185,129,0.10)),
                    rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(255,255,255,0.10);
                padding: 24px 26px;
                border-radius: 24px;
                margin-bottom: 20px;
                box-shadow: 0 18px 40px rgba(0,0,0,0.24);
                position: relative;
                overflow: hidden;
            }

            .app-header::after {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.05) 45%, transparent 100%);
                pointer-events: none;
            }

            .app-title {
                font-size: 2rem;
                font-weight: 800;
                margin: 0;
                line-height: 1.08;
                letter-spacing: -0.03em;
            }

            .app-subtitle {
                color: var(--muted);
                margin-top: 8px;
                font-size: 0.98rem;
            }

            .panel-card {
                background:
                    linear-gradient(180deg, rgba(17,24,39,0.94), rgba(12,18,31,0.96));
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 20px;
                padding: 16px 16px 10px 16px;
                box-shadow: 0 14px 34px rgba(0,0,0,0.22);
                margin-bottom: 14px;
            }

            .panel-title {
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 10px;
                color: #ffffff;
            }

            .panel-note {
                color: var(--muted);
                font-size: 0.92rem;
                margin-top: -3px;
                margin-bottom: 10px;
            }

            .preview-shell {
                background:
                    linear-gradient(180deg, rgba(17,24,39,0.94), rgba(12,18,31,0.96));
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 22px;
                padding: 18px;
                min-height: 70vh;
                box-shadow: 0 16px 40px rgba(0,0,0,0.24);
            }

            .preview-placeholder {
                border: 1px dashed #334155;
                background: rgba(15,23,42,0.65);
                border-radius: 16px;
                padding: 28px 20px;
                text-align: center;
                color: #94a3b8;
                margin-top: 10px;
            }

            .stButton > button,
            .stDownloadButton > button {
                width: 100%;
                border-radius: 14px;
                border: 1px solid #2d3b50;
                color: white;
                font-weight: 700;
                padding: 0.6rem 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.22);
            }

            .stButton > button {
                background: linear-gradient(135deg, #0ea5e9, #2563eb);
            }

            .stDownloadButton > button {
                background: linear-gradient(135deg, #162235, #1e293b);
            }

            .small-kpi {
                background:
                    linear-gradient(180deg, rgba(15,23,42,0.90), rgba(10,15,27,0.92));
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 16px;
                padding: 13px 12px;
                text-align: center;
                box-shadow: 0 12px 24px rgba(0,0,0,0.16);
            }

            .small-kpi .label {
                color: #9ca3af;
                font-size: 0.86rem;
                margin-bottom: 6px;
            }

            .small-kpi .value {
                color: #ffffff;
                font-size: 1.14rem;
                font-weight: 800;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value):
    st.markdown(
        f'<div class="small-kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_placeholder(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="preview-placeholder">
            <div style="font-size:1.2rem;font-weight:800;margin-bottom:8px;">{title}</div>
            <div>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
