# recommender/preprocess.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def load_data(movies_path: str, ratings_path: str):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def clean_movies(movies: pd.DataFrame):
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies = movies.dropna(subset=['genres'])
    return movies

def build_similarity_matrix(movies: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def save_artifacts(movies: pd.DataFrame, cosine_sim, movies_out: str, sim_out: str):
    movies.to_csv(movies_out, index=False)
    joblib.dump(cosine_sim, sim_out)
    print("Artifacts saved.")

if __name__ == "__main__":
    movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
    movies = clean_movies(movies)
    cosine_sim = build_similarity_matrix(movies)
    save_artifacts(movies, cosine_sim, 'data/movies_clean.csv', 'data/cosine_sim.pkl')