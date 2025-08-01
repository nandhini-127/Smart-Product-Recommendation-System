import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_excel("D:\Product Recommendation System-Project 5\data\Ecommerce_Datasets.xlsx")

# Drop missing descriptions
df = df.dropna(subset=['product_description'])

# Combine category and description
df['text'] = df['product_category_name'].fillna('') + ' ' + df['product_description'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Build mapping from product_id to index
product_indices = pd.Series(df.index, index=df['product_id']).drop_duplicates()

# Get most recently viewed product for each user
user_recent = df.sort_values(by='review_creation_date').groupby('customer_id').last().reset_index()

# Prepare result storage
recommendations = []

for _, row in user_recent.iterrows():
    customer_id = row['customer_id']
    prod_id = row['product_id']

    if prod_id not in product_indices:
        continue

    idx = product_indices[prod_id]
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Safely convert scores to floats
    sim_scores = [
        (i, float(score[0]) if isinstance(score, (np.ndarray, list)) else float(score))
        for i, score in sim_scores
    ]

    # Sort the scores and take top 5 recommendations (excluding the product itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    sim_scores = sim_scores[1:6]

    top_products = [(df.iloc[i]['product_id'], score) for i, score in sim_scores]

    for prod, score in top_products:
        recommendations.append({
            'customer_id': customer_id,
            'input_product_id': prod_id,
            'recommended_product_id': prod,
            'similarity_score': round(score, 4)
        })

# Save to Excel
recommendation_df = pd.DataFrame(recommendations)
recommendation_df.to_excel("content_based_recommendations.xlsx", index=False)

print("✅ Content-based recommendations saved to D:/content_based_recommendations.xlsx")
