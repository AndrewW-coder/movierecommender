# recommender/model.py
import joblib
import pandas as pd
from recommender.recommend import get_similar_movies

class MovieRecommender:
    def __init__(self, movies_path: str, tfidf_path: str, matrix_path: str):
        self.movies = pd.read_csv(movies_path)
        self.tfidf = joblib.load(tfidf_path)
        self.tfidf_matrix = joblib.load(matrix_path)
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    def recommend(self, title: str, n: int = 10):
        return get_similar_movies(
            title, self.movies, self.tfidf_matrix, self.indices, n
        )