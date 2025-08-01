import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import StandardScaler
def get_popular_products(df, top_n=10):
    top_products = (
        df.groupby('product_id')
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    return top_products

def get_location_based(df, user_id, top_n=10):
    user_location = df[df['customer_id'] == user_id][['customer_city', 'customer_state']].dropna().iloc[0]
    location_products = df[
        (df['customer_city'] == user_location['customer_city']) |
        (df['customer_state'] == user_location['customer_state'])
    ]['product_id'].value_counts().head(top_n).index.tolist()
    return location_products




def get_rfm_recommendations(df, user_id, top_n=10):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Set current date for Recency calculation
    now = pd.to_datetime('2024-12-31')

    # Ensure timestamp column is datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # Ensure a 'price' column exists
    if 'price' not in df.columns:
        df['price'] = df['payment_value']  # adjust as needed

    # Step 1: Compute raw RFM values
    rfm = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (now - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    }).reset_index()

    # Step 2: Normalize RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Step 3: Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['RFM_Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Step 4: Select top customers (lowest recency, high frequency/monetary)
    # Sort by cluster based on frequency and monetary
    top_customers = rfm.sort_values(by=['Frequency', 'Monetary'], ascending=False)['customer_id'].head(50)

    # Step 5: Filter original orders
    top_customer_orders = df[df['customer_id'].isin(top_customers)]

    # Step 6: Recommend top-N most popular products among those customers
    recommended_products = top_customer_orders['product_id'].value_counts().head(top_n).index.tolist()

    return recommended_products

def get_price_sensitive_products(df, user_id, top_n=10):
    customer_df = df[df['customer_id'] == user_id]
    avg_spend = customer_df.groupby('order_id')['price'].sum().mean()
    product_prices = df.groupby('product_id')['price'].mean()
    affordable_products = product_prices[product_prices <= avg_spend].sort_values(ascending=False).head(top_n).index.tolist()
    return affordable_products
