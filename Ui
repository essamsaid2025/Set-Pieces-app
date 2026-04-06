import streamlit as st


THEMES = {
    "The Athletic Dark": {
        "bg": "#0E1117",
        "panel": "#111827",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
    },
    "Opta Dark": {
        "bg": "#0E1117",
        "panel": "#141A22",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
    },
    "Sofa Light": {
        "bg": "white",
        "panel": "#F5F7FA",
        "pitch": "#2f6b3a",
        "text": "#111111",
        "muted": "#5A6572",
        "lines": "#DDE3EA",
        "goal": "#444444",
        "pitch_lines": "#FFFFFF",
    },
    "Black Stripe": {
        "bg": "#000000",
        "panel": "#000000",
        "pitch": "#000000",
        "pitch_stripe": "#0A0A0A",
        "text": "#FFFFFF",
        "muted": "#B7B7B7",
        "lines": "#2A2A2A",
        "goal": "#FFFFFF",
        "pitch_lines": "#FFFFFF",
    },
}


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --bg: #0b1220;
                --card: #111827;
                --card-2: #0f172a;
                --border: #243041;
                --text: #f3f4f6;
                --muted: #9ca3af;
                --accent: #38bdf8;
            }

            .stApp {
                background: linear-gradient(180deg, #09111f 0%, #0b1220 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.5rem;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
                max-width: 100%;
            }

            h1, h2, h3, h4, h5, h6, p, span, div, label {
                color: var(--text);
            }

            .app-header {
                background: linear-gradient(135deg, rgba(56,189,248,0.16), rgba(16,185,129,0.10));
                border: 1px solid rgba(255,255,255,0.08);
                padding: 20px 22px;
                border-radius: 20px;
                margin-bottom: 18px;
            }

            .app-title {
                font-size: 2rem;
                font-weight: 800;
                margin: 0;
                line-height: 1.1;
            }

            .app-subtitle {
                color: var(--muted);
                margin-top: 8px;
                font-size: 0.98rem;
            }

            .panel-card {
                background: rgba(17,24,39,0.92);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 16px 16px 10px 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.22);
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
                background: rgba(17,24,39,0.92);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 16px;
                min-height: 70vh;
                box-shadow: 0 10px 30px rgba(0,0,0,0.22);
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

            .section-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.12), rgba(255,255,255,0.03));
                margin: 14px 0 14px 0;
                border-radius: 999px;
            }

            div[data-testid="stFileUploader"] section {
                background: rgba(15,23,42,0.85);
                border: 1px dashed #334155;
                border-radius: 14px;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            .stTextInput > div > div,
            .stNumberInput > div > div {
                background: #0f172a;
            }

            .stButton > button {
                width: 100%;
                border-radius: 12px;
                border: 1px solid #2d3b50;
                background: linear-gradient(135deg, #0ea5e9, #2563eb);
                color: white;
                font-weight: 700;
                padding: 0.6rem 1rem;
            }

            .stDownloadButton > button {
                width: 100%;
                border-radius: 12px;
                border: 1px solid #2d3b50;
                background: #162235;
                color: white;
                font-weight: 700;
                padding: 0.6rem 1rem;
            }

            .small-kpi {
                background: rgba(15,23,42,0.85);
                border: 1px solid #243041;
                border-radius: 14px;
                padding: 12px;
                text-align: center;
            }

            .small-kpi .label {
                color: #9ca3af;
                font-size: 0.86rem;
                margin-bottom: 6px;
            }

            .small-kpi .value {
                color: #ffffff;
                font-size: 1.1rem;
                font-weight: 800;
            }

            .stExpander {
                border-radius: 14px !important;
                border: 1px solid #243041 !important;
                background: rgba(15,23,42,0.55) !important;
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
