import pandas as pd
import numpy as np
import os
from joblib import dump
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def get_existing_customer_id(df, name):
    user = df[df['customer_name'].str.lower() == name.lower()]
    if not user.empty:
        return user.iloc[0]['customer_id']
    return None

def get_user_city_state(customer_id, df):
    try:
        user_row = df[df['customer_id'] == customer_id].iloc[0]
        return user_row['customer_city'], user_row['customer_state']
    except Exception as e:
        print(f"[Warning] Could not fetch city/state from dataframe: {e}")
        return None, None


# --- Helper: Get top-N predictions ---
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# --- Main hybrid recommendation function ---
def get_hybrid_recommendations(df, top_n_count=5):
    df = df[df['review_score'].between(1, 5)]
    user_counts = df['customer_id'].value_counts()
    product_counts = df['product_id'].value_counts()
    df = df[df['customer_id'].isin(user_counts[user_counts > 2].index)]
    df = df[df['product_id'].isin(product_counts[product_counts > 2].index)]

    # Collaborative Filtering
    rating_df = df[['customer_id', 'product_id', 'review_score']]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating_df, reader)

    if len(rating_df) >= 50:
        param_grid = {
            'n_epochs': [10, 20],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.04],
            'n_factors': [10, 50]
        }
        gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=5)
        gs.fit(data)
        best_model = gs.best_estimator['rmse']
    else:
        best_model = SVDpp()

    trainset = data.build_full_trainset()
    best_model.fit(trainset)

    # Save model
    dump(best_model, "models/svd_model.joblib")

    predictions = best_model.test(trainset.build_anti_testset())
    top_n = get_top_n(predictions, n=top_n_count)

    # Content-Based Filtering
    content_df = df[['product_id', 'category', 'product_description']].drop_duplicates().fillna('')
    content_df['combined'] = content_df['category'] + ' ' + content_df['product_description']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_df['combined'])

    dump(tfidf, "models/tfidf_vectorizer.joblib")
    dump(tfidf_matrix, "models/tfidf_matrix.joblib")
    dump(content_df['product_id'].tolist(), "models/product_ids.joblib")

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df['customer_last_purchased_product'] = df.groupby('customer_id')['product_id'].transform('last')

    content_index = {pid: idx for idx, pid in enumerate(content_df['product_id'])}
    pred_df = []

    for user_id, recs in top_n.items():
        last_product = df[df['customer_id'] == user_id]['customer_last_purchased_product'].iloc[0]
        last_idx = content_index.get(last_product)

        for product_id, score in recs:
            prod_idx = content_index.get(product_id)
            content_score = 0.0
            if prod_idx is not None and last_idx is not None:
                content_score = float(cosine_similarity(tfidf_matrix[prod_idx], tfidf_matrix[last_idx])[0][0])
            hybrid_score = 0.8 * score + 0.2 * content_score
            pred_df.append({
                'customer_id': user_id,
                'recommended_product_id': product_id,
                'pred_rating': score,
                'content_score': content_score,
                'hybrid_score': hybrid_score
            })

    return pd.DataFrame(pred_df).sort_values(['customer_id', 'hybrid_score'], ascending=[True, False])

# --- Run only if script is executed directly ---
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    file_path = "D:\\Product Recommendation System-Project 5\\data\\Ecommerce_Datasets.xlsx"
    df = pd.read_excel(file_path, sheet_name=0)

    print("Generating hybrid recommendations...")
    hybrid_recs = get_hybrid_recommendations(df, top_n_count=5)
    print("\nSample Hybrid Recommendations:")
    print(hybrid_recs.head(10))
