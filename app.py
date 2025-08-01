from flask import Flask, render_template, request, redirect, url_for, session
import json
import hashlib
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from datetime import datetime
from src.recommend import recommend_products
from src.ai_strategies import (
    get_popular_products,
    get_location_based,
    get_rfm_recommendations,
    get_price_sensitive_products
)
from src.model_utils import get_user_city_state, get_existing_customer_id

# Load dataset
original_df = pd.read_excel("data\Ecommerce_Datasets.xlsx")

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main.db'  # Default DB
app.config['SQLALCHEMY_BINDS'] = {
    'orders': 'sqlite:///orders.db',
    'feedback': 'sqlite:///feedback.db'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==================== USER MODEL (main.db) ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(80))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(255))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))


# --- ORDERS: Stored in orders_db (no foreign key allowed to users table)
class Order(db.Model):
    __bind_key__ = 'orders'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.String(100), unique=True, nullable=False)
    product_id = db.Column(db.String(100))
    category = db.Column(db.String(255))
    price = db.Column(db.Float)
    order_purchase_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    customer_id = db.Column(db.String(100), nullable=False)  

# --- FEEDBACK: Stored in feedback_db
class Feedback(db.Model):
    __bind_key__ = 'feedback'

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=False)




def standardize_product(product):
    return {
        
        'product_id': product.get('product_id', str(product.get('id'))),  # dataset matching
        'title': product.get('title'),
        'image': product.get('image') or (product.get('images', [''])[0] if product.get('images') else ''),
        'category': product.get('category')['name'] if isinstance(product.get('category'), dict) else product.get('category'),
        'rating': product.get('rating', {'rate': 4.0, 'count': 0}),
        'price': product.get('price'),
        'description': product.get('description', '')
    }

# Load product data
with open("static/data/products.json", "r", encoding="utf-8") as f:
    raw_products = json.load(f)

    products = [standardize_product(p) for p in raw_products]

product_lookup = {str(product["product_id"]): product for product in products}

def hash_email(email):
    return hashlib.md5(email.encode()).hexdigest()

def get_product_by_id(product_id):
    return next((p for p in products if str(p.get('product_id')) == str(product_id)), None)


@app.route('/')
def landing():
    return render_template('landing.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name')

        
        customer_id = get_existing_customer_id(original_df, name)

        if customer_id:
            user_city, user_state = get_user_city_state(original_df, customer_id)
            session['user'] = name
            session['customer_id'] = customer_id
            session['user_city'] = user_city
            session['user_state'] = user_state
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Customer not found in dataset. Please check your name.")

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')

        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error="User already exists")

        customer_id = hash_email(email)
        new_user = User(name=name, email=email, password=password, customer_id=customer_id)
        db.session.add(new_user)
        db.session.commit()
        session['email'] = email
        session['user'] = name
        session['customer_id'] = customer_id
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    username = session['user']
    featured_products = products[:75]
    return render_template("home.html", username=username, featured_products=featured_products)

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = get_product_by_id(product_id)
    if not product:
        return "Product not found", 404
    return render_template("product_details.html", product=product)


@app.route('/recommendations')
def recommendations():
    if 'customer_id' not in session:
        return redirect(url_for('login'))

    customer_id = session['customer_id']
    hybrid_ids = recommend_products(user_id=customer_id, df=original_df, top_n=10)

    # AI Strategy-based sections
    trending_ids = get_popular_products(df=original_df, top_n=10)
    location_ids = get_location_based(df=original_df, user_id=customer_id, top_n=10)
    rfm_ids = get_rfm_recommendations(df=original_df, user_id=customer_id, top_n=10)
    price_ids = get_price_sensitive_products(df=original_df, user_id=customer_id, top_n=10)

    # Get full product info
    def get_product_details(product_ids):
        return [p for p in products if p['product_id'] in product_ids]

    return render_template('recommendations.html',
    hybrid_recs=get_product_details(hybrid_ids) if hybrid_ids else [],
    trending=get_product_details(trending_ids) if trending_ids else [],
    location=get_product_details(location_ids) if location_ids else [],
    rfm=get_product_details(rfm_ids) if rfm_ids else [],
    price_sensitive=get_product_details(price_ids) if price_ids else [],
)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'customer_id' not in session:
        return redirect(url_for('login'))

    customer_id = session['customer_id']
    user = User.query.filter_by(customer_id=customer_id).first()

    if not user:
        return "User not found",404

    if request.method == 'POST':
        user.name = request.form.get('name')
        user.email = request.form.get('email')
        user.address = request.form.get('address')
        user.customer_id = hash_email(user.email)
        user.city = request.form.get('city')
        user.state = request.form.get('state')

        db.session.commit()

        session['email'] = user.email
        session['user'] = user.name
        session['customer_id'] = user.customer_id
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    message = ""
    if request.method == 'POST':
        action = request.form.get('action')
        description = request.form.get('description') or ''
        rating = request.form.get('rating') or 0
        customer_id = session.get('customer_id', 'anonymous')

        if action == "rate":
            fb = Feedback(customer_id=customer_id, rating=int(rating), description="Rated: " + str(rating))
            message = "⭐ Rating submitted!"
        elif action == "issue":
            fb = Feedback(customer_id=customer_id, rating=0, description="Issue: " + description)
            message = "🚨 Issue reported!"
        elif action == "suggest":
            fb = Feedback(customer_id=customer_id, rating=0, description="Suggestion: " + description)
            message = "💡 Suggestion submitted!"
        else:
            fb = Feedback(customer_id=customer_id, rating=int(rating), description=description)
            message = "✅ Feedback submitted!"

        db.session.add(fb)
        db.session.commit()
        return render_template('feedback.html', message=message)

    return render_template('feedback.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

import secrets
from datetime import datetime


@app.route('/place_order/<product_id>', methods=['POST'])
def place_order(product_id):
    if 'customer_id' not in session:
        return redirect(url_for('login'))

    customer_id = session['customer_id']
    product = get_product_by_id(product_id)
    if not product:
        return "Product not found", 404

    order_id = secrets.token_hex(16)
    timestamp = datetime.now()
    price = product.get('price', 0)
    category = product.get('category', 'Uncategorized') 

    new_order = Order(
        order_id=order_id,
        customer_id=customer_id,
        product_id=product_id,
        category=category,
        order_purchase_timestamp=timestamp,
        price=price
    )
    db.session.add(new_order)
    db.session.commit()

    return render_template("order_success.html", product=product, order_id=order_id, timestamp=timestamp)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates tables in all configured databases
    app.run(debug=True, use_reloader=False)
