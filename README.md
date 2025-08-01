# 🛒 Aivora — Where Intelligence Meets Convenience

Aivora is an intelligent, AI-driven **hybrid product recommendation system** developed during an internship at **AK Infopark Pvt. Ltd.**. The system combines **collaborative filtering (SVD++)**, **content-based filtering (TF-IDF)**, and **AI strategies** such as RFM analysis, popularity trends, price sensitivity clustering, and location-aware recommendations to provide personalized product suggestions.

---

## 🚀 Features

-  User Authentication (Login & Signup)
-  User Profile with City and State info
-  Product Listing with images and details
-  Personalized Hybrid Recommendations
- 🧠 AI Strategy-based suggestions:
  - ✅ **Trending Products** (popularity over time)
  - ✅ **Location-based Recommendations**
  - ✅ **RFM Analysis** (Recency-Frequency-Monetary)
  - ✅ **Price Sensitivity Clustering**
-  Customer Feedback Collection
-  Help & FAQ Section
-  Flask-based Web Application with dynamic pages

---

## 🧠 Recommendation Logic

### 🔸 Hybrid Recommendation System
- **Collaborative Filtering:** SVD++ algorithm using `surprise` library
- **Content-Based Filtering:** TF-IDF vectorization on product descriptions and categories
- **Combination Strategy:** Merges CF and CBF scores for personalized suggestions

### 🔸 AI Strategy Enhancements
- **RFM Segmentation:** Prioritizes active and valuable customers
- **Popularity Trends:** Recommends top-N products based on review count & recent orders
- **Price Sensitivity:** Clusters users by average spending using K-Means
- **Location-Aware Recs:** Shows what’s popular in user's city/state

---

## 🏗️ Project Structure\

<details> <summary>Click to expand</summary>
Smart-Product-Recommendation-System/
│
├── app.py                        # Flask application entry point
├── requirements.txt             # Project dependencies
│
├── static/
│   └── data/
│       └── products.json        # Product catalog with metadata and images
│
├── templates/                   # HTML pages (home, login, profile, recommendations, etc.)
│
├── src/
│   ├── recommend.py             # Main recommendation logic
│   ├── model_utils.py           # SVD, TF-IDF models and hybrid integration
│   └── ai_strategies.py         # AI-based RFM, popularity, clustering logic
│
├── feedback.db                  # SQLite DB for user feedback
├── main.db                      # User auth and profile database
├── orders.db                    # Order history database
│
└── README.md                    # Project documentation
</details>

---

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nandhini-127/Smart-Product-Recommendation-System.git
   cd Smart-Product-Recommendation-System
   
2. **Run the application:**

   ```bash
   python app.py
   ```
3. **Open in browser:**
   ```bash
   http://127.0.0.1:5000
   ```

## 🧪 Tech Stack
Backend: Flask (Python)

Machine Learning: Surprise (SVD++), Scikit-learn (TF-IDF, KMeans)

Frontend: HTML, CSS (responsive UI)

Database: SQLite (3 DBs: users, feedback, orders)

Data: Brazilian eCommerce Dataset (custom-transformed to Indian market)


## 📩 Feedback and Contributions
If you have any suggestions or feedback, feel free to open an issue or pull request.
You can also share feedback using the built-in feedback form in the app.

## 👩‍💻 Developed By
Nandhini S.S.
Intern at AK Infopark Pvt. Ltd.

## 📜 License
This project is for educational and internship purposes. Please contact the author for reuse permissions.
