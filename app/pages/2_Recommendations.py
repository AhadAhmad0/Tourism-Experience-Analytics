"""
Tourism Experience Analytics - Recommendations page
"""

import streamlit as st
from utils import load_artifacts, get_attraction_options, recommend_collaborative, recommend_content_based

st.set_page_config(page_title="Recommendations — Tourism Analytics", page_icon="\U0001F3AF", layout="wide")

st.title("\U0001F3AF Attraction Recommendations")

artifacts, missing = load_artifacts()
if missing:
    st.error("Model files missing. Go to the Home page for details on what's needed.")
    st.stop()

attraction_options = get_attraction_options(artifacts)

tab1, tab2 = st.tabs(["\U0001F465 Collaborative Filtering (Existing Users)", "\U0001F4CD Content-Based (Any User)"])

with tab1:
    st.markdown(
        "Recommends attractions based on what **similar visitors** rated highly. "
        "Only works for users who already have rating history in the dataset — "
        "this is the classic collaborative filtering cold-start limitation."
    )
    user_item_matrix = artifacts['user_item_matrix']
    sample_users = user_item_matrix.dropna(how='all').index.tolist()[:200]  # cap for a responsive dropdown

    selected_user = st.selectbox("Select an existing User ID", sample_users)

    if st.button("Get Collaborative Recommendations", type="primary"):
        recs = recommend_collaborative(selected_user, artifacts, n=8)
        if recs.empty:
            st.warning("No recommendations available for this user — they may have no rating history.")
        else:
            st.dataframe(
                recs.rename(columns={'Attraction': 'Attraction', 'AttractionType': 'Type', 'Score': 'Match Score'}),
                hide_index=True, use_container_width=True
            )

with tab2:
    st.markdown(
        "Recommends attractions **similar in type** to one you already like. "
        "Works for anyone — even a brand-new visitor with zero rating history — "
        "since it only needs a single attraction of interest, not a rating history."
    )
    attraction_names = attraction_options.sort_values('Attraction')['Attraction'].tolist()
    selected_attraction_name = st.selectbox("I like this attraction:", attraction_names)
    selected_attraction = attraction_options[attraction_options['Attraction'] == selected_attraction_name].iloc[0]

    if st.button("Get Similar Attractions", type="primary"):
        recs = recommend_content_based(selected_attraction['AttractionId'], artifacts, n=8)
        if recs.empty:
            st.warning("No similar attractions found.")
        else:
            st.dataframe(
                recs.rename(columns={'Attraction': 'Attraction', 'AttractionType': 'Type', 'Similarity': 'Similarity Score'}),
                hide_index=True, use_container_width=True
            )

st.divider()
st.caption(
    "A production system would blend both approaches: content-based suggestions for new "
    "users, shifting toward collaborative filtering as a user builds up rating history."
)
