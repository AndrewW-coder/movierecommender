# recommender/recommend.py
import pandas as pd

def get_similar_movies(title: str, movies: pd.DataFrame, cosine_sim, indices, n: int = 10):
    if title not in indices:
        return None
    
    idx = indices[title]
    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    results = movies.iloc[movie_indices][['title', 'genres']].copy()
    results['similarity'] = [round(s[1], 4) for s in sim_scores]
    return results.to_dict(orient='records')