import os 
import sqlite3
import logging
import pickle
import uuid
import pandas as pd
import json
import plotly
import plotly.express as px
from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = 'secret_key'

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_action(user_id, action, result):
    logging.info(f"User: {user_id} | Action: {action} | Result: {result}")

with open('models.pkl', 'rb') as f:
    artifacts = pickle.load(f)

encoder = artifacts['encoder']
main_model = artifacts['main_model']
metrics_df = artifacts['metrics']
df = artifacts['df']
banks_list = artifacts['banks']
best_model_name = artifacts.get('best_model_name', 'LogisticRegression')
top_5_neg_words = artifacts.get('top_5_neg_words', [])

conn = sqlite3.connect('reviews.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS user_reviews
             (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, bank TEXT, rating INTEGER, review TEXT, sentiment TEXT)''')
conn.commit()

def mask_author(name):
    return str(name)[:2] + "***" if len(str(name)) > 2 else "*"

def mask_address(address):
    return str(address)[:2] + "***" + str(address)[-2:] if len(str(address)) >= 4 else "***"

def predict_sentiment(text):
    vec = encoder.encode([text])
    pred = main_model.predict(vec)[0]
    proba = main_model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 1)
    label = "Позитивный" if pred == 1 else "Негативный"
    return label, confidence

@app.before_request
def make_session():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())[:8]

@app.after_request
def log_request(response):
    user_id = session.get('user_id', 'anonymous')
    log_action(user_id, f"{request.method} {request.path}", f"status={response.status_code}")
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        bank = request.form['bank']
        rating = int(request.form['rating'])
        review = request.form['review']
        
        label, confidence = predict_sentiment(review)
        sentiment_label = f"{label} ({confidence}%)"
        
        c.execute("INSERT INTO user_reviews (user_id, bank, rating, review, sentiment) VALUES (?, ?, ?, ?, ?)",
                  (session['user_id'], bank, rating, review, sentiment_label))
        conn.commit()
        log_action(session['user_id'], "Leave_Review", f"Bank:{bank}, Pred:{sentiment_label}")
        prediction = sentiment_label

    user_reviews = pd.read_sql_query(f"SELECT * FROM user_reviews WHERE user_id='{session['user_id']}'", conn)
    
    if not user_reviews.empty and len(user_reviews[user_reviews['rating'] >= 4]) > 0:
        liked_banks = user_reviews[user_reviews['rating'] >= 4]['bank'].tolist()
        similar_users = df[df['bank'].isin(liked_banks)]
        recs = similar_users.groupby('bank')['rating'].mean().sort_values(ascending=False)
        recs = recs[~recs.index.isin(liked_banks)].head(3).index.tolist()
        
        if not recs:
            recs = df.groupby('bank')['rating'].mean().sort_values(ascending=False).head(3).index.tolist()
    else:
        recs = df.groupby('bank')['rating'].mean().sort_values(ascending=False).head(3).index.tolist()
        
    return render_template('index.html', banks=banks_list, prediction=prediction, recs=recs)

@app.route('/stats')
def stats():
    stats_df = df.copy()
    stats_df['author'] = stats_df['author'].apply(mask_author)
    stats_df['address'] = stats_df['address'].apply(mask_address)
    bank_stats = stats_df.groupby('bank').agg(
        Rating=('rating', 'mean'),
        Reviews=('rating', 'count'),
        Pos_Share=('sentiment', lambda x: x.mean() * 100)
    ).round(2).reset_index()
    return render_template('stats.html', tables=[bank_stats.to_html(classes='table', index=False)], top_5_neg_words=top_5_neg_words)

@app.route('/models')
def show_models():
    return render_template('models.html', tables=[metrics_df.to_html(classes='table', index=False)], best_model_name=best_model_name)

@app.route('/clusters')
def clusters():
    plot_df = df.copy()
    plot_df['short_review'] = plot_df['review'].astype(str).str.replace('\n', ' ').apply(lambda x: x[:80] + '...' if len(x) > 80 else x)

    fig = px.scatter(
        plot_df, x='pca_x', y='pca_y', 
        color=plot_df['cluster'].astype(str),
        hover_data={'pca_x': False, 'pca_y': False, 'bank': True, 'rating': True, 'short_review': True}, 
        title="Визуализация кластеров"
    )
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    cluster_stats = df.groupby('cluster').agg(
        Sentiment_Mean=('sentiment', 'mean'), 
        Count=('review', 'count'),
        Top_10_Words=('cluster_top_words', 'first')
    ).round(2).reset_index()
    
    return render_template('clusters.html', tables=[cluster_stats.to_html(classes='table', index=False)], graph_html=graph_html)

@app.route('/logs')
def logs():
    try:
        with open('app.log', 'r') as f:
            log_lines = f.readlines()[-50:]
    except FileNotFoundError:
        log_lines = []
    return render_template('logs.html', logs=log_lines)

if __name__ == '__main__':
    print("Запуск: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)