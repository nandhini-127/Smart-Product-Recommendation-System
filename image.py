import requests
import json

# Fetch from Fake Store API
response = requests.get("https://dummyjson.com/products?limit=50&skip=100")
products = response.json()

# Save to local file
with open("productssss.json", "w") as f:
    json.dump(products, f, indent=2)

print("Saved", len(products), "products to products.json")
