import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# Load dataset
df = pd.read_excel("D:/Product Recommendation System-Project 5/data/Ecommerce_Datasets.xlsx")

# --- COLLABORATIVE FILTERING PART ---

# Prepare CF data
cf_df = df[['customer_id', 'product_id', 'review_score']].dropna()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(cf_df[['customer_id', 'product_id', 'review_score']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
algo = SVD()
algo.fit(trainset)

# Build product lookup
all_products = df['product_id'].dropna().unique()
rated_by_user = cf_df.groupby('customer_id')['product_id'].apply(set).to_dict()

# --- CONTENT BASED PART ---

# Drop rows with missing product descriptions
cb_df = df.dropna(subset=['product_description'])

# Combine category and description
cb_df['text'] = cb_df['product_category_name'].fillna('') + ' ' + cb_df['product_description'].fillna('')

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(cb_df['text'])
cosine_sim = cosine_similarity(tfidf_matrix)
product_indices = pd.Series(cb_df.index, index=cb_df['product_id']).drop_duplicates()

# Get most recently viewed product for each user
user_recent = df.sort_values(by='review_creation_date').groupby('customer_id').last().reset_index()

# --- HYBRID RECOMMENDATION GENERATION ---

hybrid_recs = []
alpha = 0.5  # weight for CF
beta = 0.5   # weight for CB

for _, row in user_recent.iterrows():
    customer_id = row['customer_id']
    recent_product = row['product_id']

    if recent_product not in product_indices:
        continue

    idx = product_indices[recent_product]
    cb_scores = list(enumerate(cosine_sim[idx]))

    # Filter top N CB recommendations excluding the product itself
    # Safely convert similarity scores to floats before sorting
    cleaned_cb_scores = [(i, float(score[0]) if isinstance(score, (np.ndarray, list)) else float(score))
                         for i, score in cb_scores]

    # Sort and slice top results
    cb_scores = sorted(cleaned_cb_scores, key=lambda x: x[1], reverse=True)[1:25]

    for i, sim_score in cb_scores:
        recommended_product = cb_df.iloc[i]['product_id']
        # Avoid suggesting the same product or already rated ones
        if recommended_product == recent_product:
            continue

        # Predict rating from CF
        try:
            cf_score = algo.predict(customer_id, recommended_product).est
        except:
            cf_score = 3.0  # Neutral fallback

        # Hybrid Score
        hybrid_score = alpha * cf_score + beta * sim_score

        hybrid_recs.append({
            'customer_id': customer_id,
            'input_product_id': recent_product,
            'recommended_product_id': recommended_product,
            'similarity_score': round(sim_score, 4),
            'predicted_rating': round(cf_score, 2),
            'hybrid_score': round(hybrid_score, 4)
        })

# Get top 5 per user
hybrid_df = pd.DataFrame(hybrid_recs)
hybrid_df = hybrid_df.sort_values(['customer_id', 'hybrid_score'], ascending=[True, False])
top5_hybrid = hybrid_df.groupby('customer_id').head(5).reset_index(drop=True)

# Save to Excel
top5_hybrid.to_excel("hybrid_recommendations.xlsx", index=False)
print("✅ Hybrid recommendations saved to D:/hybrid_recommendations.xlsx")
