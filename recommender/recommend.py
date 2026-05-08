# recommender/recommend.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_movies(title, movies, tfidf_matrix, indices, n=10):
    if title not in indices:
        return None

    idx = indices[title]
    movie_vector = tfidf_matrix[idx]
    sim_scores_raw = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Sort and get top n (skip index 0 — the movie itself)
    top_indices = sim_scores_raw.argsort()[::-1][1:n+1]
    
    results = movies.iloc[top_indices][['title', 'genres']].copy()
    results['similarity'] = [round(float(sim_scores_raw[i]), 4) for i in top_indices]
    return results.to_dict(orient='records')