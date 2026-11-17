# Tourism Experience Analytics

An end-to-end data science and machine learning project built to predict attraction ratings, classify visitor travel modes, and generate personalized attraction recommendations.

## Problem & Objectives
Tourism platforms need accurate personalization to match travelers with relevant experiences. This project addresses three core tasks using historical visit logs:
1. **Rating Prediction (Regression):** Estimate the rating (1–5) a user will give an attraction.
2. **Visit Mode Classification:** Predict travel context (*Business, Couples, Family, Friends, Solo*).
3. **Attraction Recommendation:** Suggest relevant destinations via item-based collaborative filtering and content-based filtering.

## Dataset & Data Engineering
Built by integrating 9 relational tables (`Updated_Item.xlsx`, transaction logs, and user/geographic lookups) covering **52,930 visits** across **33,530 users** and **1,698 attractions** (excluding the stale 30-row `Item.xlsx`). Key preprocessing steps included:
- **Schema Normalization:** Resolved mixed text-and-ID keys in `AttractionTypeId` (where 1,668 rows contained category strings instead of numeric IDs) to prevent a ~98% join failure.
- **Leakage Prevention:** Replaced raw user and item rating averages with leave-one-out target encoding.

## Modeling & Tech Stack
- **Models:** Linear Regression, Logistic Regression, Random Forest, and XGBoost (`scikit-learn`, `xgboost`).
- **Recommenders:** Cosine-similarity matrices for item-based collaborative filtering and content-based filtering.
- **Web App:** Interactive multi-page **Streamlit** application (`app/app.py`, `app/pages/`) serialized with `joblib`.

## Limitations & Setup
Only 30 of the 1,698 attractions contain historical visit logs; collaborative filtering and custom item encodings apply to those 30 items, while unseen attractions fall back to global defaults and content-based matching.

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
