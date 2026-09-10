"""
Tourism Experience Analytics - Shared utilities
Loads all trained models + reference data, and reconstructs the exact feature
set used in training for a brand-new (user, attraction, visit) combination.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACT_FILES = {
    'rating_model': 'tourism_rating_model.joblib',
    'visitmode_model': 'tourism_visitmode_model.joblib',
    'visitmode_label_encoder': 'tourism_visitmode_label_encoder.joblib',
    'scaler': 'tourism_feature_scaler.joblib',
    'feature_cols': 'tourism_feature_columns.joblib',
    'item_similarity': 'tourism_item_similarity.joblib',
    'content_similarity': 'tourism_content_similarity.joblib',
    'user_item_matrix': 'tourism_user_item_matrix.joblib',
    'item_lookup': 'tourism_item_lookup.joblib',
    'mode_lookup': 'tourism_mode_lookup.joblib',
    'city_lookup': 'tourism_city_lookup.joblib',
    'country_lookup': 'tourism_country_lookup.joblib',
    'region_lookup': 'tourism_region_lookup.joblib',
    'continent_lookup': 'tourism_continent_lookup.joblib',
    'user_avg_rating': 'tourism_user_avg_rating.joblib',
    'attraction_avg_rating': 'tourism_attraction_avg_rating.joblib',
    'global_avg_rating': 'tourism_global_avg_rating.joblib',
}

NUMERIC_FEATURES = ['VisitYear', 'VisitMonth', 'User_Visit_Count', 'Attraction_Visit_Count',
                     'User_Avg_Rating_LOO', 'Attraction_Avg_Rating_LOO']

CATEGORICAL_PREFIX_COLS = ['Continent', 'Region', 'Country', 'AttractionType']


@st.cache_resource(show_spinner="Loading models and reference data...")
def load_artifacts():
    missing = [f for f in ARTIFACT_FILES.values() if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return None, missing

    artifacts = {key: joblib.load(os.path.join(MODEL_DIR, fname)) for key, fname in ARTIFACT_FILES.items()}
    return artifacts, []


def get_attraction_options(artifacts):
    """Return a DataFrame of attractions for dropdown display, with resolved type."""
    item_df = artifacts['item_lookup']
    return item_df[['AttractionId', 'Attraction', 'AttractionType', 'AttractionCityId']].copy()


def get_geography_options(artifacts):
    """Return continent/region/country name lists for the input form."""
    continent_df = artifacts['continent_lookup']
    country_df = artifacts['country_lookup']
    return {
        'continents': continent_df[continent_df['Continent'] != '-']['Continent'].tolist(),
        'countries': country_df[country_df['Country'] != '-']['Country'].tolist(),
    }


def build_feature_row(user_continent, user_country, attraction_id, visit_year, visit_month, artifacts):
    """
    Reconstruct one model-ready row for a (new or existing) user visiting a given
    attraction, matching the exact feature engineering used in training -- but using
    plain (non-leave-one-out) averages, since there's no target to leak for a new row.
    """
    item_df = artifacts['item_lookup']
    attraction_row = item_df[item_df['AttractionId'] == attraction_id].iloc[0]
    attraction_type = attraction_row['AttractionType']

    attraction_avg_rating = artifacts['attraction_avg_rating'].get(attraction_id, artifacts['global_avg_rating'])
    attraction_visit_count = int((artifacts['user_item_matrix'][attraction_id].notna()).sum()) \
        if attraction_id in artifacts['user_item_matrix'].columns else 0

    # New user in the app has no rating history -- use the global average and a visit count of 0,
    # exactly like a genuine cold-start user would look at inference time.
    user_avg_rating = artifacts['global_avg_rating']
    user_visit_count = 0

    raw_row = {
        'VisitYear': visit_year,
        'VisitMonth': visit_month,
        'User_Visit_Count': user_visit_count,
        'Attraction_Visit_Count': attraction_visit_count,
        'User_Avg_Rating_LOO': user_avg_rating,
        'Attraction_Avg_Rating_LOO': attraction_avg_rating,
        'Continent': user_continent,
        'Country': user_country,
        'Region': 'Unknown',  # not collected directly in the simplified app form
        'AttractionType': attraction_type,
    }

    row_df = pd.DataFrame([raw_row])
    row_encoded = pd.get_dummies(row_df, columns=CATEGORICAL_PREFIX_COLS, drop_first=False)

    feature_cols = artifacts['feature_cols']
    row_aligned = row_encoded.reindex(columns=feature_cols, fill_value=0)

    scaler = artifacts['scaler']
    row_aligned[NUMERIC_FEATURES] = scaler.transform(row_aligned[NUMERIC_FEATURES])

    return row_aligned


def predict_rating(row_aligned, artifacts):
    model = artifacts['rating_model']
    pred = model.predict(row_aligned)[0]
    return float(np.clip(pred, 1.0, 5.0))


def predict_visit_mode(row_aligned, artifacts):
    model = artifacts['visitmode_model']
    label_encoder = artifacts['visitmode_label_encoder']
    mode_df = artifacts['mode_lookup']

    pred_enc = model.predict(row_aligned)[0]
    pred_label_id = label_encoder.inverse_transform([pred_enc])[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(row_aligned)[0]
        class_ids = label_encoder.inverse_transform(np.arange(len(proba)))
        id_to_name = dict(zip(mode_df['VisitModeId'], mode_df['VisitMode']))
        proba_dict = {id_to_name.get(cid, str(cid)): float(p) for cid, p in zip(class_ids, proba)}
    else:
        proba_dict = {}

    id_to_name = dict(zip(mode_df['VisitModeId'], mode_df['VisitMode']))
    pred_label_name = id_to_name.get(pred_label_id, str(pred_label_id))

    return pred_label_name, proba_dict


def recommend_collaborative(user_id, artifacts, n=5):
    user_item_matrix = artifacts['user_item_matrix']
    item_similarity_df = artifacts['item_similarity']
    item_df = artifacts['item_lookup']

    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return pd.DataFrame()

    scores = item_similarity_df[user_ratings.index].dot(user_ratings) / \
        item_similarity_df[user_ratings.index].sum(axis=1).replace(0, np.nan)
    scores = scores.drop(index=user_ratings.index, errors='ignore')
    top = scores.sort_values(ascending=False).head(n)

    result = item_df[item_df['AttractionId'].isin(top.index)][['AttractionId', 'Attraction', 'AttractionType']].copy()
    result['Score'] = result['AttractionId'].map(top)
    return result.sort_values('Score', ascending=False)


def recommend_content_based(attraction_id, artifacts, n=5):
    content_similarity_df = artifacts['content_similarity']
    item_df = artifacts['item_lookup']

    if attraction_id not in content_similarity_df.index:
        return pd.DataFrame()

    top = content_similarity_df[attraction_id].drop(index=attraction_id, errors='ignore').sort_values(ascending=False).head(n)
    result = item_df[item_df['AttractionId'].isin(top.index)][['AttractionId', 'Attraction', 'AttractionType']].copy()
    result['Similarity'] = result['AttractionId'].map(top)
    return result.sort_values('Similarity', ascending=False)
