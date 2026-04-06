import streamlit as st

from data_utils import load_data, normalize_set_piece_df, ensure_columns, apply_flip_y
from ui_theme import inject_styles, render_header, render_kpi_card, render_placeholder, THEMES
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
    initial_sidebar_state="collapsed",
)

inject_styles()
render_header(
    title="⚽ Set Piece Analysis App",
    subtitle="Upload CSV / Excel → Corners / Free Kicks / Defensive Set Pieces / Reports",
)

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.05, 1.55], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Output & File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your set piece file", type=["csv", "xlsx", "xls"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎛️ Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-note">Same theme system and display language as your first app.</div>',
        unsafe_allow_html=True,
    )

    theme_name = st.selectbox("Choose theme", list(THEMES.keys()), index=0)
    flip_y = st.checkbox("Flip Y axis (use this if your Y=0 is at the bottom)", value=False)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Charts")
    all_charts = list(CHART_BUILDERS.keys())
    selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts)

    with st.expander("📌 Requirements under each chart are shown in preview", expanded=False):
        for ch in selected_charts:
            st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS[ch]))

    generate_clicked = st.button("Generate Set Piece Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
    st.markdown("### 📊 Preview & Downloads")

    if uploaded is None:
        render_placeholder(
            "No file uploaded yet",
            "Upload a CSV / Excel file from the left panel, then generate the analysis.",
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

    if "set_piece_type" in df.columns:
        sp_values = [
            x for x in df["set_piece_type"].dropna().astype(str).unique().tolist()
            if x and x != "nan"
        ]
        if sp_values:
            selected_sp = st.multiselect("Filter set piece types", sp_values, default=sp_values)
            if selected_sp:
                df = df[df["set_piece_type"].isin(selected_sp)].copy()

    if "team" in df.columns:
        teams = [
            x for x in df["team"].dropna().astype(str).unique().tolist()
            if x and x != "nan"
        ]
        if teams:
            selected_team = st.selectbox("Team filter", ["all"] + teams, index=0)
            if selected_team != "all":
                df = df[df["team"] == selected_team].copy()

    if "side" in df.columns:
        sides = [
            x for x in df["side"].dropna().astype(str).unique().tolist()
            if x and x != "nan"
        ]
        if sides:
            selected_sides = st.multiselect("Side filter", sides, default=sides)
            if selected_sides:
                df = df[df["side"].isin(selected_sides)].copy()

    if "delivery_type" in df.columns:
        dtypes = [
            x for x in df["delivery_type"].dropna().astype(str).unique().tolist()
            if x and x != "nan"
        ]
        if dtypes:
            selected_delivery = st.multiselect("Delivery type filter", dtypes, default=dtypes)
            if selected_delivery:
                df = df[df["delivery_type"].isin(selected_delivery)].copy()

    if not generate_clicked:
        render_placeholder(
            "Ready to generate",
            "Configure your settings and click Generate Set Piece Analysis.",
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
            fig = CHART_BUILDERS[chart_name](df.copy(), theme_name)
            figures.append(fig)
            st.pyplot(fig, use_container_width=True)

            png_bytes = fig_to_png_bytes(fig)
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
