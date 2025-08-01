import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.ai_strategies import (
    get_popular_products,
    get_location_based,
    get_rfm_recommendations,
    get_price_sensitive_products
)

# Load models
svd_model = joblib.load("models/svd_model.joblib")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")
product_ids = joblib.load("models/product_ids.joblib")

def recommend_products(user_id, df, top_n=10):
    # Collaborative Filtering
    cf_preds = []
    for pid in product_ids:
        try:
            pred = svd_model.predict(user_id, pid).est
            cf_preds.append((pid, pred))
        except:
            continue
    cf_preds = sorted(cf_preds, key=lambda x: x[1], reverse=True)[:top_n]
    cf_df = pd.DataFrame(cf_preds, columns=['product_id', 'score'])

    # Content-Based Filtering
    user_products = df[df['customer_id'] == user_id]['product_id'].unique()
    content_scores = {}

    for pid in user_products:
        if pid in product_ids:
            idx = product_ids.index(pid)
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            for i, score in enumerate(sim_scores):
                if product_ids[i] not in user_products:
                    content_scores[product_ids[i]] = content_scores.get(product_ids[i], 0) + score

    cbf_df = pd.DataFrame(content_scores.items(), columns=['product_id', 'score'])
    cbf_df = cbf_df.sort_values(by='score', ascending=False).head(top_n)

    # Merge CF and CBF
    user_recs = pd.merge(cf_df, cbf_df, on='product_id', how='outer', suffixes=('_cf', '_cbf')).fillna(0)
    user_recs['score'] = user_recs['score_cf'] + user_recs['score_cbf']

    # AI Strategies
    location_products = get_location_based(df, user_id, top_n=top_n)
    rfm_products = get_rfm_recommendations(df, user_id, top_n=top_n)
    popular_products = get_popular_products(df, top_n=top_n)
    price_sensitive_products = get_price_sensitive_products(df, user_id, top_n=top_n)

    # Add scores from AI strategies
    weight_loc, weight_rfm, weight_pop, weight_price = 0.5, 0.4, 0.3, 0.3
    user_recs['score'] += user_recs['product_id'].isin(location_products).astype(int) * weight_loc
    user_recs['score'] += user_recs['product_id'].isin(rfm_products).astype(int) * weight_rfm
    user_recs['score'] += user_recs['product_id'].isin(popular_products).astype(int) * weight_pop
    user_recs['score'] += user_recs['product_id'].isin(price_sensitive_products).astype(int) * weight_price

    # Final sort
    final_recs = user_recs.sort_values(by='score', ascending=False).head(top_n)['product_id'].tolist()
    return final_recs
