# recommender/preprocess.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(movies_path: str, ratings_path: str, tags_path: str):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    return movies, ratings, tags

def clean_movies(movies: pd.DataFrame, tags: pd.DataFrame):
    # Aggregate all tags per movie into one string
    tags_grouped = tags.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.astype(str).str.lower())
    ).reset_index()
    tags_grouped.columns = ['movieId', 'tags']

    # Merge tags into movies dataframe
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tags'] = movies['tags'].fillna('')

    # Clean genres — replace pipes with spaces
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

    # Combine genres + tags into one "soup" text field
    # Repeat genres twice so they carry more weight than tags
    movies['soup'] = movies['genres_clean'] + ' ' + movies['genres_clean'] + ' ' + movies['tags']

    return movies

def build_tfidf_matrix(movies: pd.DataFrame):
    tfidf = TfidfVectorizer(
        stop_words='english',
        min_df=2,          # ignore terms that appear in fewer than 2 movies
        max_df=0.8,        # ignore terms that appear in more than 80% of movies
        ngram_range=(1,2)  # capture both single words and two-word phrases
    )
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    print(f"Vocabulary size: {len(tfidf.vocabulary_)} terms")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix

def save_artifacts(movies, tfidf, tfidf_matrix, out_dir='data'):
    movies.to_csv(f'{out_dir}/movies_clean.csv', index=False)
    joblib.dump(tfidf, f'{out_dir}/tfidf.pkl')
    joblib.dump(tfidf_matrix, f'{out_dir}/tfidf_matrix.pkl')
    print("Artifacts saved.")

if __name__ == "__main__":
    movies, ratings, tags = load_data(
        'data/movies.csv',
        'data/ratings.csv',
        'data/tags.csv'
    )
    movies = clean_movies(movies, tags)
    tfidf, tfidf_matrix = build_tfidf_matrix(movies)
    save_artifacts(movies, tfidf, tfidf_matrix)
    print("Done! Check data/ for new artifacts.")