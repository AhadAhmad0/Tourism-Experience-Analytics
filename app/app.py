"""
Tourism Experience Analytics - Home page
Run with: streamlit run app.py
"""

import streamlit as st
from utils import load_artifacts

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="\U0001F30D", layout="wide")

st.title("\U0001F30D Tourism Experience Analytics")
st.subheader("Rating Prediction, Visit Mode Classification & Attraction Recommendations")

st.markdown(
    "Predict how a visitor will rate an attraction, what kind of trip they're likely on, "
    "and discover attractions they'd probably enjoy -- all from a single tourism dataset "
    "of 52,930 visits across 1,698 attractions."
)

artifacts, missing = load_artifacts()

if missing:
    st.error(
        "**Model/reference files not found.** This app needs the following files in the "
        "same folder as `app.py`:\n\n" + "\n".join(f"- `{f}`" for f in missing) +
        "\n\nDownload these from your Kaggle notebook's `/kaggle/working` output."
    )
else:
    st.success("Models and reference data loaded and ready.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Transactions", "52,930")
with col2:
    st.metric("Users", "33,530")
with col3:
    st.metric("Attractions", "1,698")
with col4:
    st.metric("Visit Modes", "5")

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    **\u2b50 Rating Prediction**
    Predict the rating (1-5) a visitor is likely to give a specific attraction,
    based on their location and the attraction's track record.
    """)
with c2:
    st.markdown("""
    **\U0001F465 Visit Mode Classification**
    Predict whether a visit is most likely Business, Couples, Family,
    Friends, or Solo -- useful for tailoring the experience in advance.
    """)
with c3:
    st.markdown("""
    **\U0001F3AF Recommendations**
    Get attraction suggestions two ways: collaborative filtering (what
    similar visitors loved) and content-based filtering (similar attraction types).
    """)

st.divider()
st.caption("Use the sidebar to navigate between pages.")

with st.expander("About this model — read before trusting the results"):
    st.markdown("""
    - Rating and visit mode models are trained on historical transaction data joined
      across 9 relational tables (users, attractions, geography, and visit records).
    - A genuinely new user in this app has no rating history, so predictions use
      global averages as a cold-start default -- the same limitation any real
      recommendation platform faces for brand-new visitors.
    - Collaborative filtering recommendations only work for **existing users** in the
      training data (it needs prior ratings to find similar users). Content-based
      filtering works for anyone, since it only needs one attraction of interest.
    - This tool is a decision-support and exploration aid, not a guarantee of visitor
      satisfaction.
    """)
