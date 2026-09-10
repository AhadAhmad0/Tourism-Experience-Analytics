"""
Tourism Experience Analytics - Predict page
Rating prediction + Visit Mode classification for a chosen attraction and visitor profile.
"""

import streamlit as st
from utils import load_artifacts, get_attraction_options, get_geography_options, build_feature_row, predict_rating, predict_visit_mode
import pandas as pd

st.set_page_config(page_title="Predict — Tourism Analytics", page_icon="\u2b50", layout="wide")

st.title("\u2b50 Predict Rating & Visit Mode")

artifacts, missing = load_artifacts()
if missing:
    st.error("Model files missing. Go to the Home page for details on what's needed.")
    st.stop()

attraction_options = get_attraction_options(artifacts)
geo = get_geography_options(artifacts)

st.markdown("Select a visitor's location and an attraction to get instant predictions.")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Visitor Profile")
        user_continent = st.selectbox("Continent", sorted(geo['continents']))
        user_country = st.selectbox("Country", sorted(geo['countries']))
        visit_year = st.number_input("Visit Year", min_value=2013, max_value=2030, value=2024)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)), index=5)
    with c2:
        st.subheader("Attraction")
        attraction_names = attraction_options.sort_values('Attraction')['Attraction'].tolist()
        selected_attraction_name = st.selectbox("Attraction", attraction_names)
        selected_attraction = attraction_options[attraction_options['Attraction'] == selected_attraction_name].iloc[0]
        st.caption(f"Type: {selected_attraction['AttractionType']}")

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = build_feature_row(
        user_continent=user_continent,
        user_country=user_country,
        attraction_id=selected_attraction['AttractionId'],
        visit_year=visit_year,
        visit_month=visit_month,
        artifacts=artifacts,
    )

    rating_pred = predict_rating(row, artifacts)
    mode_pred, mode_proba = predict_visit_mode(row, artifacts)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Rating")
        st.metric("Rating (1-5)", f"{rating_pred:.1f} \u2b50")
        if rating_pred < 3:
            st.warning("Below-average predicted rating — this attraction/visitor combination may not be a strong match.")

    with col2:
        st.subheader("Predicted Visit Mode")
        st.metric("Most Likely Visit Mode", mode_pred)
        if mode_proba:
            proba_df = pd.DataFrame(list(mode_proba.items()), columns=['Mode', 'Probability']).sort_values('Probability', ascending=False)
            st.dataframe(proba_df, hide_index=True, use_container_width=True)

    st.caption(
        "These predictions use cold-start defaults for a new visitor (no prior rating history). "
        "For a returning user with visit history, predictions would incorporate their actual "
        "past behavior via the recommendation models on the next page."
    )
