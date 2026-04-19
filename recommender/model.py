# recommender/model.py
import joblib
import pandas as pd
from recommender.recommend import get_similar_movies

class MovieRecommender:
    def __init__(self, movies_path: str, sim_path: str):
        self.movies = pd.read_csv(movies_path)
        self.cosine_sim = joblib.load(sim_path)
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    def recommend(self, title: str, n: int = 10):
        return get_similar_movies(title, self.movies, self.cosine_sim, self.indices, n)