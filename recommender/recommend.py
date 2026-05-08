# recommender/recommend.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

def fix_title(title: str) -> str:
    # Matches ", The" or ", A" or ", An" right before the year bracket
    return re.sub(r'^(.*),\s(The|A|An)(\s\(\d{4}\))$', r'\2 \1\3', title)

def unfix_title(title: str) -> str:
    # Reverse of fix_title — converts back to storage format for index lookup
    return re.sub(r'^(The|A|An)\s(.*?)(\s\(\d{4}\))$', r'\2, \1\3', title)

def get_similar_movies(title, movies, tfidf_matrix, indices, n=10):
    # Try the title as given first, then try the storage format
    lookup = title if title in indices else unfix_title(title)
    
    if lookup not in indices:
        return None

    idx = indices[lookup]

    movie_vector = tfidf_matrix[idx]
    sim_scores_raw = cosine_similarity(movie_vector, tfidf_matrix).flatten()

    top_indices = sim_scores_raw.argsort()[::-1][1:n+1]

    results = movies.iloc[top_indices][['movieId', 'title', 'genres']].copy()
    results['similarity'] = [round(float(sim_scores_raw[i]), i) for i in top_indices]

    # Fix display titles without touching the underlying data
    results['title'] = results['title'].apply(fix_title)

    return results.to_dict(orient='records')