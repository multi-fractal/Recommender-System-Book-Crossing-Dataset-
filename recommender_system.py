# !pip install scikit-surprise

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np

# 1. Load Data
def load_data(ratings_path, books_path, users_path):
    ratings = pd.read_csv(ratings_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    books = pd.read_csv(books_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    return ratings, books, users

# 2. Clean Columns
def clean_columns(ratings, books, users):
    ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
    books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year', 'Publisher',
    'Image-S', 'Image-M', 'Image-L']
    users.columns = ['User-ID', 'Location', 'Age']
    books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year', 'Publisher']]
    users = users[['User-ID', 'Age']]
    return ratings, books, users

# 3. Clean Data
def clean_data(ratings, books, users):
    ratings = ratings.dropna()
    users['Age'] = users['Age'].apply(lambda x: x if 5 < x < 100 else None)
    users = users.dropna(subset=['Age'])
    ratings = ratings[ratings['Book-Rating'] > 0]
    active_users = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(active_users[active_users >= 20].index)]
    popular_books = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(popular_books[popular_books >= 20].index)]
    return ratings, books, users

# 4. Train SVD Model
def train_svd_model(ratings):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model, trainset, testset

# 5. Evaluate Model (RMSE / MAE)
def evaluate_model(model, testset):
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return rmse, mae, predictions

# 6. Top-K Metrics (Precision@K, Recall@K, NDCG@K)
def precision_recall_at_k(predictions, k=10, threshold=7):
    from collections import defaultdict
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        num_relevant = sum((true_r >= threshold) for (_, true_r) in top_k)
        num_rel_total = sum((true_r >= threshold) for (_, true_r) in ratings)
        precisions.append(num_relevant / k if k else 0)
        recalls.append(num_relevant / num_rel_total if num_rel_total else 0)

    return np.mean(precisions), np.mean(recalls)

def ndcg_at_k(predictions, k=10, threshold=7):
    from collections import defaultdict
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = []
    
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        dcg = 0.0
        idcg = 0.0

        for i, (_, true_r) in enumerate(ratings[:k]):
            rel = 1 if true_r >= threshold else 0
            dcg += (2**rel - 1) / np.log2(i + 2)

        for i in range(min(k, sum(1 for (_, true_r) in ratings if true_r >= threshold))):
            idcg += 1 / np.log2(i + 2)

        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcgs)

# 7. Re-train model on all data
def retrain_on_full_data(ratings):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    full_trainset = data.build_full_trainset()

    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(full_trainset)

    return model

# 8. Recommend books for a specific user
def recommend_books(user_id, model, books, ratings, n=10, max_candidates=1000):
    rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])

    # Выбираем топ популярных книг, которых пользователь ещё не оценил
    book_popularity = ratings['ISBN'].value_counts()
    popular_books = [isbn for isbn in book_popularity.index if isbn not in rated_books][:max_candidates]

    predictions = []
    for isbn in popular_books:
        book_row = books.loc[books['ISBN'] == isbn]
        if book_row.empty:
            continue
        est = model.predict(user_id, isbn).est
        title = book_row['Book-Title'].values[0]
        author = book_row['Book-Author'].values[0]
        predictions.append({'ISBN': isbn, 'Title': title, 'Author': author, 'Predicted-Rating': est})

    predictions.sort(key=lambda x: x['Predicted-Rating'], reverse=True)
    return predictions[:n]

# Main Pipeline
def main():
    ratings, books, users = load_data('BX-Book-Ratings.csv', 'BX-Books.csv', 'BX-Users.csv')
    ratings, books, users = clean_columns(ratings, books, users)
    ratings, books, users = clean_data(ratings, books, users)
    model, trainset, testset = train_svd_model(ratings)
    rmse, mae, predictions = evaluate_model(model, testset)
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=7)
    ndcg = ndcg_at_k(predictions, k=10, threshold=7)

    print('Model Evaluation:')
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('Precision@10:', precision)
    print('Recall@10:', recall)
    print('NDCG@10:', ndcg)

# Re-train on all data for final deployment
    final_model = retrain_on_full_data(ratings)
    print('Final model retrained on all data.')

# Example: recommend books for a specific user
    user_id_example = ratings['User-ID'].iloc[0]
    recommendations = recommend_books(user_id_example, final_model, books, ratings, n=10)
    print(f'Recommendations for User {user_id_example}:')
    for rec in recommendations:
        print(rec)

# Uncomment to run
main()