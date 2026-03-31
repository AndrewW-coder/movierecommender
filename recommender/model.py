import joblib
import pandas as pd

class MovieRecommender:
    def __init__(self, movies_path: str, sim_path: str):
        self.movies = pd.read_csv(movies_path)
        self.cosine_sim = joblib.load(sim_path)
        self.indices = pd.Series(
            self.movies.index,
            index = self.movies['title']
        ).drop_duplicates()

    def recommend(self, title: str, n: int = 10):
        if title not in self.indices:
            return None
        
        idx = self.indices[title]
        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]),
            key = lambda x: x[1],
            reverse = True
        )[1: n + 1]

        movies_indices = [i[0] for i in sim_scores]
        results = self.movies.iloc[movies_indices][['title', 'genres']].copy()
        results['similarity'] = [round(s[1], 4) for s in sim_scores]
        
        return results.to_dict(orient='records')