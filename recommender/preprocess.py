# recommender/preprocess.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(movies_path: str, ratings_path: str):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def clean_movies(movies: pd.DataFrame):
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies = movies.dropna(subset=['genres'])
    return movies

def build_tfidf_matrix(movies: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    return tfidf, tfidf_matrix

def save_artifacts(movies, tfidf, tfidf_matrix, out_dir='data'):
    movies.to_csv(f'{out_dir}/movies_clean.csv', index=False)
    joblib.dump(tfidf, f'{out_dir}/tfidf.pkl')
    joblib.dump(tfidf_matrix, f'{out_dir}/tfidf_matrix.pkl')
    print("Artifacts saved.")

if __name__ == "__main__":
    movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
    movies = clean_movies(movies)
    tfidf, tfidf_matrix = build_tfidf_matrix(movies)
    save_artifacts(movies, tfidf, tfidf_matrix)