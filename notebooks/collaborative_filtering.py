import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# Step 1: Load Dataset
df = pd.read_excel("D:\Product Recommendation System-Project 5\data\Ecommerce_Datasets.xlsx")  # Your dataset
print("Dataset loaded with shape:", df.shape)

# Step 2: Filter required columns
df = df[['customer_id', 'product_id', 'review_score']].dropna()

# Step 3: Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['customer_id', 'product_id', 'review_score']], reader)

# Step 4: Train/Test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Step 5: Train the SVD model
algo = SVD()
algo.fit(trainset)

# Step 6: Predict ratings for all user-item pairs not in training data
all_customers = df['customer_id'].unique()
all_products = df['product_id'].unique()

# Build a dictionary of items already rated by user
rated_by_user = df.groupby('customer_id')['product_id'].apply(set).to_dict()

# Predict top-N for all users
top_n = defaultdict(list)
n = 5  # number of recommendations per user

for uid in all_customers:
    rated_items = rated_by_user.get(uid, set())
    unseen_items = [iid for iid in all_products if iid not in rated_items]

    predictions = [algo.predict(uid, iid) for iid in unseen_items]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n[uid] = [(pred.iid, pred.est) for pred in predictions[:n]]

# Step 7: Convert results to DataFrame
results = []
for uid, user_recs in top_n.items():
    for iid, rating in user_recs:
        results.append({
            'customer_id': uid,
            'product_id': iid,
            'predicted_rating': round(rating, 2)
        })

recommendations_df = pd.DataFrame(results)
print(recommendations_df.head())

# Step 8: Save to Excel
recommendations_df.to_excel("D:/"
                            "collaborative_recommendations.xlsx", index=False)
print("✅ Collaborative filtering recommendations saved to Excel.")
