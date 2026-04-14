import streamlit as st

from data_utils import load_data, normalize_set_piece_df, ensure_columns, apply_flip_y
from ui_theme import (
    inject_styles,
    render_header,
    render_kpi_card,
    render_placeholder,
    THEMES,
    FONT_FAMILIES,
    HEATMAP_STYLES,
    build_chart_style,
)
from set_piece_charts import (
    CHART_REQUIREMENTS,
    CHART_BUILDERS,
    save_report_pdf,
    fig_to_png_bytes,
)

st.set_page_config(
    page_title="Set Piece Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header(
    title="⚽ Set Piece Analysis App",
    subtitle="Upload CSV / Excel → Set Piece Reports → Styled Visual Analysis",
)

with st.sidebar:
    st.markdown("## 🎛️ Global Controls")

    theme_name = st.selectbox("Theme preset", list(THEMES.keys()), index=0)
    flip_y = st.checkbox("Flip Y axis", value=False)
    uploaded = st.file_uploader("Upload your set piece file", type=["csv", "xlsx", "xls"])

    st.markdown("---")
    st.markdown("## 🏟️ Pitch Controls")

    pitch_orientation = st.radio(
        "Pitch orientation",
        options=["Horizontal", "Vertical"],
        index=0,
        horizontal=True,
        help="Switch all pitch-based charts between landscape (horizontal) and portrait (vertical) layout.",
    )
    pitch_vertical = pitch_orientation == "Vertical"

    show_thirds = st.checkbox(
        "Show thirds lines",
        value=False,
        help="Overlay dashed lines dividing the pitch into Defensive / Middle / Attacking thirds.",
    )

    st.markdown("---")
    st.markdown("## 🎨 Style Controls")

    base_theme = THEMES[theme_name]
    font_family = st.selectbox("Font family", FONT_FAMILIES, index=0)
    heatmap_cmap = st.selectbox("Heatmap theme", HEATMAP_STYLES, index=0)

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

    with st.expander("Markers / Lines / Heat", expanded=False):
        marker_size = st.slider("Marker size", 20, 220, 90)
        marker_edge_width = st.slider("Marker edge width", 0.0, 3.5, 1.2, 0.1)
        line_width = st.slider("Line width", 0.5, 4.0, 1.4, 0.1)
        trajectory_width = st.slider("Trajectory width", 0.5, 5.0, 1.8, 0.1)
        trajectory_alpha = st.slider("Trajectory alpha", 0.10, 1.00, 0.75, 0.05)
        trajectory_headwidth = st.slider("Trajectory arrow head", 2.0, 10.0, 4.5, 0.5)
        alpha = st.slider("Marker alpha", 0.10, 1.00, 0.90, 0.05)
        kde_alpha = st.slider("Heatmap alpha", 0.10, 1.00, 0.72, 0.05)
        grid_alpha = st.slider("Grid alpha", 0.00, 0.50, 0.18, 0.02)

    with st.expander("Layout", expanded=False):
        show_title = st.checkbox("Show chart title", value=True)
        show_legend = st.checkbox("Show legends", value=True)
        show_grid = st.checkbox("Show grid on bar charts", value=True)
        show_ticks = st.checkbox("Show pitch ticks", value=False)
        tight_layout = st.checkbox("Use tight layout", value=True)
        export_dpi = st.slider("Export PNG DPI", 120, 400, 260)

    with st.expander("⚽ Scatter Dot Color", expanded=False):
        st.caption("Color for delivery end scatter points")
        scatter_dot_color = st.color_picker("Scatter dot color", base_theme["accent"])

    with st.expander("🗺️ First Contact Map — Action Colors", expanded=False):
        st.caption("Color per action type on the First Contact Location Map")
        fc_shot_color      = st.color_picker("Shot",       "#FFD400")
        fc_clearance_color = st.color_picker("Clearance",  "#FF4D4D")
        fc_header_color    = st.color_picker("Header",     "#38BDF8")
        fc_threat_color    = st.color_picker("Threat",     "#A78BFA")
        fc_cross_color     = st.color_picker("Cross",      "#22C55E")
        fc_foul_color      = st.color_picker("Foul",       "#F97316")
        fc_other_color     = st.color_picker("Other",      "#94A3B8")
        st.caption("Show / hide action types")
        fc_show_shot      = st.checkbox("Show Shot",       value=True)
        fc_show_clearance = st.checkbox("Show Clearance",  value=True)
        fc_show_header    = st.checkbox("Show Header",     value=True)
        fc_show_threat    = st.checkbox("Show Threat",     value=True)
        fc_show_cross     = st.checkbox("Show Cross",      value=True)
        fc_show_foul      = st.checkbox("Show Foul",       value=True)
        fc_show_other     = st.checkbox("Show Other",      value=True)

    with st.expander("🎯 Arrow Colors (by delivery type)", expanded=False):
        st.caption("Override arrow colors per delivery type")
        arrow_inswing  = st.color_picker("Inswing",  base_theme["accent"])
        arrow_outswing = st.color_picker("Outswing", base_theme["warning"])
        arrow_straight = st.color_picker("Straight", base_theme["accent_2"])
        arrow_driven   = st.color_picker("Driven",   base_theme["success"])
        arrow_short    = st.color_picker("Short",    base_theme["danger"])

    with st.expander("📊 Bar Chart Colors", expanded=False):
        st.caption("Override bar/histogram fill color")
        bar_default = st.color_picker("Default bar color", base_theme["accent"])
        bar_success  = st.color_picker("Success bar color", base_theme["success"])
        bar_danger   = st.color_picker("Danger bar color",  base_theme["danger"])

    with st.expander("👕 Shirt Colors (Taker Stats Table)", expanded=False):
        shirt_body   = st.color_picker("Shirt body",   base_theme["accent"])
        shirt_sleeve = st.color_picker("Shirt sleeves", base_theme["panel"])
        shirt_number = st.color_picker("Number color",  base_theme["bg"])

    st.markdown("---")
    st.markdown("## 📊 Charts")

    all_charts = list(CHART_BUILDERS.keys())

    # ── attacking defaults ───────────────────────────────────────────────────
    attacking_defaults = [
        "Delivery Trajectories - Left Corners",
        "Delivery Trajectories - Right Corners",
        "Attack Free Kick Trajectories",
        "Delivery End Scatter - Left Corner",
        "Delivery End Scatter - Right Corner",
        "Zone Delivery Count Map - Left Corner",
        "Zone Delivery Count Map - Right Corner",
        "Avg Players Per Zone - Left Corner",
        "Avg Players Per Zone - Right Corner",
        "First Contact Location Map",
        "Set Piece Landing Heatmap",
        "Taker Stats Table",
    ]

    # ── defensive defaults ───────────────────────────────────────────────────
    defensive_defaults = [
        "Defensive Shape Map",
        "Defender vs Attacker Zone Matchup",
        "Clearance Outcome Map",
        "Set Piece Conceded Heatmap",
        "Defensive Success Rate By Zone",
        "First Contact Win Rate Trend",
        "Second Ball Recovery Map",
    ]

    st.markdown("#### ⚔️ Attacking Charts")
    attacking_charts = st.multiselect(
        "Attacking charts",
        [c for c in all_charts if c not in defensive_defaults],
        default=[c for c in attacking_defaults if c in all_charts],
        label_visibility="collapsed",
    )

    st.markdown("#### 🛡️ Defensive Charts")
    st.caption(
        "These charts use: **result** (clearance detection), **first_contact_win**, "
        "**second_ball_win**, **defenders_near_post / defenders_far_post**, and **opponent** columns."
    )
    defensive_charts = st.multiselect(
        "Defensive charts",
        defensive_defaults,
        default=[c for c in defensive_defaults if c in all_charts],
        label_visibility="collapsed",
    )

    selected_charts = attacking_charts + defensive_charts

    with st.expander("Chart requirements", expanded=False):
        for ch in selected_charts:
            st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS.get(ch, [])))

    generate_clicked = st.button("Generate Set Piece Analysis", use_container_width=True)

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
    "trajectory_width": trajectory_width,
    "trajectory_alpha": trajectory_alpha,
    "trajectory_headwidth": trajectory_headwidth,
    "alpha": alpha,
    "kde_alpha": kde_alpha,
    "grid_alpha": grid_alpha,
    "show_title": show_title,
    "show_legend": show_legend,
    "show_grid": show_grid,
    "show_ticks": show_ticks,
    "tight_layout": tight_layout,
    "export_dpi": export_dpi,
    "heatmap_cmap": heatmap_cmap,
    # ── pitch layout ──────────────────────────────────────────────────────────
    "pitch_vertical": pitch_vertical,
    "show_thirds":    show_thirds,
    # ── scatter dot color ─────────────────────────────────────────────────────
    "scatter_dot_color": scatter_dot_color,
    # ── arrow colours ─────────────────────────────────────────────────────────
    "arrow_colors": {
        "inswing":  arrow_inswing,
        "outswing": arrow_outswing,
        "straight": arrow_straight,
        "driven":   arrow_driven,
        "short":    arrow_short,
    },
    # ── bar colours ───────────────────────────────────────────────────────────
    "bar_colors": {
        "default": bar_default,
        "success": bar_success,
        "danger":  bar_danger,
    },
    # ── shirt colours ─────────────────────────────────────────────────────────
    "shirt_body_color":   shirt_body,
    "shirt_sleeve_color": shirt_sleeve,
    "shirt_number_color": shirt_number,
}
chart_style = build_chart_style(theme_name, style_overrides)

left_col, right_col = st.columns([1.0, 1.75], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Style Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-note">
            Theme: <b>{theme_name}</b><br>
            Orientation: <b>{pitch_orientation}</b><br>
            Thirds: <b>{"On" if show_thirds else "Off"}</b><br>
            Heatmap: <b>{heatmap_cmap}</b><br>
            Font: <b>{font_family}</b><br>
            Marker size: <b>{marker_size}</b><br>
            Line width: <b>{line_width}</b><br>
            Export DPI: <b>{export_dpi}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Defensive column guide ────────────────────────────────────────────────
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🛡️ Defensive Column Guide</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-note">
        The defensive charts use these optional columns:<br><br>
        <b>first_contact_win</b> — yes/no or 1/0<br>
        <b>second_ball_win</b> — yes/no or 1/0<br>
        <b>result</b> — clearance / shot / header…<br>
        <b>outcome</b> — Successful / Unsuccessful<br>
        <b>defenders_near_post</b> — integer count<br>
        <b>defenders_far_post</b> — integer count<br>
        <b>opponent</b> — match label for trend chart<br><br>
        Charts degrade gracefully when columns are missing.
        </div>
        """,
        unsafe_allow_html=True,
    )
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

    # ── Filters ───────────────────────────────────────────────────────────────
    filter_cols = st.columns(4)

    with filter_cols[0]:
        if "set_piece_type" in df.columns:
            sp_values = [x for x in df["set_piece_type"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sp_values:
                selected_sp = st.multiselect("Set piece types", sp_values, default=sp_values)
                if selected_sp:
                    df = df[df["set_piece_type"].isin(selected_sp)].copy()

        # also support raw 'set_piece' column before normalisation
        elif "set_piece" in df.columns:
            sp_values = [x for x in df["set_piece"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sp_values:
                selected_sp = st.multiselect("Set piece types", sp_values, default=sp_values)
                if selected_sp:
                    df = df[df["set_piece"].isin(selected_sp)].copy()

    with filter_cols[1]:
        if "team" in df.columns:
            teams = [x for x in df["team"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if teams:
                selected_team = st.selectbox("Team", ["all"] + teams, index=0)
                if selected_team != "all":
                    df = df[df["team"] == selected_team].copy()

    with filter_cols[2]:
        if "side" in df.columns:
            sides = [x for x in df["side"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sides:
                selected_sides = st.multiselect("Side", sides, default=sides)
                if selected_sides:
                    df = df[df["side"].isin(selected_sides)].copy()

    with filter_cols[3]:
        if "delivery_type" in df.columns:
            dtypes = [x for x in df["delivery_type"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if dtypes:
                selected_delivery = st.multiselect("Delivery type", dtypes, default=dtypes)
                if selected_delivery:
                    df = df[df["delivery_type"].isin(selected_delivery)].copy()

    # ── Opponent filter (for trend chart) ────────────────────────────────────
    if "opponent" in df.columns:
        opps = [x for x in df["opponent"].dropna().astype(str).unique().tolist() if x and x != "nan"]
        if len(opps) > 1:
            selected_opps = st.multiselect("Opponent (for trend chart)", opps, default=opps)
            if selected_opps:
                df = df[df["opponent"].isin(selected_opps)].copy()

    if not generate_clicked:
        render_placeholder(
            "Ready to generate",
            "Adjust theme, orientation, thirds, heatmap style, and filters from the sidebar, then click Generate.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ── Render charts ─────────────────────────────────────────────────────────
    figures = []

    # Section headers
    att_set = set(attacking_charts)
    def_set = set(defensive_charts)
    shown_att_header = False
    shown_def_header = False

    for chart_name in selected_charts:
        # section header
        if chart_name in att_set and not shown_att_header:
            st.markdown("## ⚔️ Attacking Set Piece Charts")
            shown_att_header = True
        if chart_name in def_set and not shown_def_header:
            st.markdown("## 🛡️ Defensive Set Piece Charts")
            shown_def_header = True

        req = CHART_REQUIREMENTS.get(chart_name, [])
        missing = ensure_columns(df, req)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">{chart_name}</div>', unsafe_allow_html=True)
        if req:
            st.markdown('<div class="panel-note">Requirements: ' + ", ".join(req) + '</div>', unsafe_allow_html=True)

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
            "⬇️ Download Full Set Piece Report PDF",
            data=pdf_bytes,
            file_name="set_piece_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No charts were rendered. Check the required columns shown under each chart.")

    st.markdown("</div>", unsafe_allow_html=True)
