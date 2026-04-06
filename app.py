import streamlit as st

from data_utils import load_data, normalize_set_piece_df, ensure_columns, apply_flip_y
from ui_theme import (
    inject_styles,
    render_header,
    render_kpi_card,
    render_placeholder,
    THEMES,
    FONT_FAMILIES,
    build_chart_style,
)
from set_piece_charts import (
    CHART_REQUIREMENTS,
    CHART_BUILDERS,
    save_report_pdf,
    fig_to_png_bytes,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Set Piece Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header(
    title="⚽ Set Piece Analysis App",
    subtitle="Upload CSV / Excel → Set Piece Reports → Professional Chart Styling Controls",
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎛️ Global Controls")

    theme_name = st.selectbox("Theme preset", list(THEMES.keys()), index=0)
    flip_y = st.checkbox("Flip Y axis", value=False)
    uploaded = st.file_uploader("Upload your set piece file", type=["csv", "xlsx", "xls"])

    st.markdown("---")
    st.markdown("## 🎨 Style Controls")

    base_theme = THEMES[theme_name]

    font_family = st.selectbox("Font family", FONT_FAMILIES, index=0)

    with st.expander("Colors", expanded=False):
        bg = st.color_picker("Figure background", base_theme["bg"])
        panel = st.color_picker("Panel / bar background", base_theme["panel"])
        pitch = st.color_picker("Pitch color", base_theme["pitch"])
        pitch_lines = st.color_picker("Pitch lines", base_theme["pitch_lines"])
        text = st.color_picker("Main text", base_theme["text"])
        muted = st.color_picker("Muted text", base_theme["muted"])
        lines = st.color_picker("Border / axis lines", base_theme["lines"])
        accent = st.color_picker("Primary accent", base_theme["accent"])
        accent_2 = st.color_picker("Secondary accent", base_theme["accent_2"])
        success = st.color_picker("Success", base_theme["success"])
        warning = st.color_picker("Warning", base_theme["warning"])
        danger = st.color_picker("Danger", base_theme["danger"])

    with st.expander("Typography", expanded=False):
        title_size = st.slider("Title size", 10, 28, 16)
        label_size = st.slider("Axis label size", 8, 22, 11)
        tick_size = st.slider("Tick size", 8, 20, 10)
        legend_size = st.slider("Legend size", 8, 20, 10)

    with st.expander("Markers & Lines", expanded=False):
        marker_size = st.slider("Marker size", 20, 220, 90)
        marker_edge_width = st.slider("Marker edge width", 0.0, 3.5, 1.2, 0.1)
        line_width = st.slider("Line width", 0.5, 4.0, 1.4, 0.1)
        alpha = st.slider("Marker alpha", 0.10, 1.00, 0.90, 0.05)
        kde_alpha = st.slider("Heatmap alpha", 0.10, 1.00, 0.72, 0.05)
        grid_alpha = st.slider("Grid alpha", 0.00, 0.50, 0.18, 0.02)

    with st.expander("Layout", expanded=False):
        show_title = st.checkbox("Show chart title", value=True)
        show_legend = st.checkbox("Show legends", value=True)
        show_grid = st.checkbox("Show grid on bar charts", value=True)
        show_ticks = st.checkbox("Show pitch ticks", value=True)
        tight_layout = st.checkbox("Use tight layout", value=True)
        export_dpi = st.slider("Export PNG DPI", 120, 400, 260)

    st.markdown("---")
    st.markdown("## 📊 Charts")
    all_charts = list(CHART_BUILDERS.keys())
    selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts)

    with st.expander("Chart requirements", expanded=False):
        for ch in selected_charts:
            st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS[ch]))

    generate_clicked = st.button("Generate Set Piece Analysis", use_container_width=True)

# =========================================================
# STYLE OBJECT
# =========================================================
style_overrides = {
    "bg": bg,
    "panel": panel,
    "pitch": pitch,
    "pitch_lines": pitch_lines,
    "text": text,
    "muted": muted,
    "lines": lines,
    "accent": accent,
    "accent_2": accent_2,
    "success": success,
    "warning": warning,
    "danger": danger,
    "font_family": font_family,
    "title_size": title_size,
    "label_size": label_size,
    "tick_size": tick_size,
    "legend_size": legend_size,
    "marker_size": marker_size,
    "marker_edge_width": marker_edge_width,
    "line_width": line_width,
    "alpha": alpha,
    "kde_alpha": kde_alpha,
    "grid_alpha": grid_alpha,
    "show_title": show_title,
    "show_legend": show_legend,
    "show_grid": show_grid,
    "show_ticks": show_ticks,
    "tight_layout": tight_layout,
    "export_dpi": export_dpi,
}
chart_style = build_chart_style(theme_name, style_overrides)

# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([1.0, 1.7], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Style Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-note">
            Theme: <b>{theme_name}</b><br>
            Font: <b>{font_family}</b><br>
            Marker size: <b>{marker_size}</b><br>
            Line width: <b>{line_width}</b><br>
            Alpha: <b>{alpha}</b><br>
            Export DPI: <b>{export_dpi}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Quick Colors</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.color_picker("Accent", value=accent, key="display_accent", disabled=True)
        st.color_picker("Success", value=success, key="display_success", disabled=True)
        st.color_picker("Danger", value=danger, key="display_danger", disabled=True)
    with c2:
        st.color_picker("Pitch", value=pitch, key="display_pitch", disabled=True)
        st.color_picker("Text", value=text, key="display_text", disabled=True)
        st.color_picker("Lines", value=lines, key="display_lines", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
    st.markdown("### 📊 Preview & Downloads")

    if uploaded is None:
        render_placeholder(
            "No file uploaded yet",
            "Upload a CSV / Excel file from the sidebar, then generate the analysis.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    df = load_data(uploaded)
    df = normalize_set_piece_df(df)
    df = apply_flip_y(df, flip_y=flip_y)

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Rows", len(df))
    with k2:
        render_kpi_card("Columns", len(df.columns))
    with k3:
        seq_n = int(df["sequence_id"].nunique()) if "sequence_id" in df.columns else 0
        render_kpi_card("Sequences", seq_n)

    with st.expander("Preview data (first 25 rows)", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(25), use_container_width=True)

    # =========================================================
    # FILTERS
    # =========================================================
    filter_cols = st.columns(4)

    with filter_cols[0]:
        if "set_piece_type" in df.columns:
            sp_values = [
                x for x in df["set_piece_type"].dropna().astype(str).unique().tolist()
                if x and x != "nan"
            ]
            if sp_values:
                selected_sp = st.multiselect("Set piece types", sp_values, default=sp_values)
                if selected_sp:
                    df = df[df["set_piece_type"].isin(selected_sp)].copy()

    with filter_cols[1]:
        if "team" in df.columns:
            teams = [
                x for x in df["team"].dropna().astype(str).unique().tolist()
                if x and x != "nan"
            ]
            if teams:
                selected_team = st.selectbox("Team", ["all"] + teams, index=0)
                if selected_team != "all":
                    df = df[df["team"] == selected_team].copy()

    with filter_cols[2]:
        if "side" in df.columns:
            sides = [
                x for x in df["side"].dropna().astype(str).unique().tolist()
                if x and x != "nan"
            ]
            if sides:
                selected_sides = st.multiselect("Side", sides, default=sides)
                if selected_sides:
                    df = df[df["side"].isin(selected_sides)].copy()

    with filter_cols[3]:
        if "delivery_type" in df.columns:
            dtypes = [
                x for x in df["delivery_type"].dropna().astype(str).unique().tolist()
                if x and x != "nan"
            ]
            if dtypes:
                selected_delivery = st.multiselect("Delivery type", dtypes, default=dtypes)
                if selected_delivery:
                    df = df[df["delivery_type"].isin(selected_delivery)].copy()

    if not generate_clicked:
        render_placeholder(
            "Ready to generate",
            "Adjust style, filters, and chart choices from the sidebar, then click Generate Set Piece Analysis.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    figures = []

    for chart_name in selected_charts:
        req = CHART_REQUIREMENTS[chart_name]
        missing = ensure_columns(df, req)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">{chart_name}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Requirements: ' + ", ".join(req) + '</div>',
            unsafe_allow_html=True,
        )

        if missing:
            st.warning("Missing columns: " + ", ".join(missing))
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        try:
            fig = CHART_BUILDERS[chart_name](
                df.copy(),
                theme_name=theme_name,
                flip_y=False,
                style_overrides=chart_style,
            )
            figures.append(fig)
            st.pyplot(fig, use_container_width=True)

            png_bytes = fig_to_png_bytes(fig, dpi=chart_style["export_dpi"])
            png_name = chart_name.lower().replace(" ", "_").replace("%", "pct") + ".png"

            st.download_button(
                f"⬇️ Download {chart_name} PNG",
                data=png_bytes,
                file_name=png_name,
                mime="image/png",
                key=f"png_{chart_name}",
            )
        except Exception as e:
            st.error(f"Could not render {chart_name}: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    if figures:
        pdf_bytes = save_report_pdf(figures)
        st.download_button(
            "⬇️ Download Set Piece Report PDF",
            data=pdf_bytes,
            file_name="set_piece_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No charts were rendered. Check the required columns shown under each chart.")

    st.markdown("</div>", unsafe_allow_html=True)
