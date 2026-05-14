import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

if not os.path.exists('static'):
    os.makedirs('static')

df = pd.read_csv('bank_reviews3.csv').dropna(subset=['review', 'rating', 'bank'])
def get_sentiment(rating):
    if rating >= 4:
        return 1
    else:
        return 0

df['sentiment'] = df['rating'].apply(get_sentiment)
df['date'] = pd.to_datetime(df['date'], errors='coerce') 
banks = df['bank'].unique().tolist()

encoder = SentenceTransformer('all-MiniLM-L6-v2')
X = encoder.encode(df['review'].tolist())
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_to_train = {
    'LogisticRegression': (LogisticRegression(max_iter=500), 
                           {'C': [0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}),
    'RandomForest': (RandomForestClassifier(random_state=42), 
                     {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=42), 
                {'n_estimators': [50], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}),
    'MLPClassifier': (MLPClassifier(max_iter=300, random_state=42), 
                      {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}),
    'CatBoost': (CatBoostClassifier(verbose=0, random_state=42), 
                 {'iterations': [50, 100], 'depth': [4, 6], 'learning_rate': [0.03, 0.1]})
}

metrics_list = []
best_models = {}

plt.figure(figsize=(8, 6))

for name, (model, params) in models_to_train.items():
    model.fit(X_train, y_train)
    f1_base = f1_score(y_test, model.predict(X_test))
    
    grid = GridSearchCV(model, params, cv=3, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    y_pred = best_model.predict(X_test)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name}')

    metrics_list.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 3),
        'Precision': round(precision_score(y_test, y_pred), 3),
        'Recall': round(recall_score(y_test, y_pred), 3),
        'F1_before': round(f1_base, 3),
        'F1_after': round(f1_score(y_test, y_pred), 3)
    })

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Доля ложноположительных (FPR)')
plt.ylabel('Доля истинно положительных (TPR)')
plt.title('ROC-кривые')
plt.legend(loc='lower right')
plt.savefig('static/roc_curves.png')
plt.close()

metrics_df = pd.DataFrame(metrics_list)
best_model_name = metrics_df.loc[metrics_df['F1_after'].idxmax()]['Model']
main_model = best_models[best_model_name]

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
df['pca_x'] = X_2d[:, 0]
df['pca_y'] = X_2d[:, 1]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster_dbscan'] = dbscan.fit_predict(X)

vectorizer = CountVectorizer(max_features=500, stop_words=None)
X_words = vectorizer.fit_transform(df['review'].astype(str))
feature_names = vectorizer.get_feature_names_out()

top_words_map = {}
for cluster_id in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cluster_id
    cluster_word_counts = X_words[mask.values].sum(axis=0).A1
    top_indices = cluster_word_counts.argsort()[::-1][:10]
    top_words_map[cluster_id] = ', '.join(feature_names[top_indices])

df['cluster_top_words'] = df['cluster'].map(top_words_map)
neg_reviews = df[df['sentiment'] == 0]['review'].dropna().astype(str)
vectorizer_neg = CountVectorizer(max_features=5)
vectorizer_neg.fit(neg_reviews)
top_5_neg_words = list(vectorizer_neg.get_feature_names_out())

plt.figure(figsize=(7, 4))
df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Распределение оценок')
plt.xlabel('Оценка')
plt.ylabel('Количество отзывов')
plt.savefig('static/rating_dist.png')
plt.close()

plt.figure(figsize=(10, 4))
df_time = df.dropna(subset=['date'])
df_time.groupby(df_time['date'].dt.to_period('M')).size().plot(kind='line', marker='o')
plt.title('Динамика отзывов')
plt.xlabel('Дата (месяц)')
plt.ylabel('Количество отзывов')
plt.savefig('static/time_dynamics.png')
plt.close()

neg_text = ' '.join(df[df['sentiment'] == 0]['review'].astype(str).tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
plt.figure(figsize=(10, 4))
plt.imshow(wc)
plt.axis('off')
plt.savefig('static/wordcloud.png')
plt.close()

artifacts = {
    'encoder': encoder,
    'main_model': main_model,
    'metrics': metrics_df,
    'df': df,
    'banks': banks,
    'best_model_name': best_model_name, 
    'top_5_neg_words': top_5_neg_words  
}
with open('models.pkl', 'wb') as f:
    pickle.dump(artifacts, f)