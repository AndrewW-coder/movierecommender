# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from recommender.model import MovieRecommender

app = FastAPI(title="Movie Recommender API")

# Allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender(
    movies_path="data/movies_clean.csv",
    tfidf_path="data/tfidf.pkl",
    matrix_path="data/tfidf_matrix.pkl"
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/recommend/{title}")
def recommend(title: str, n: int = 10):
    results = recommender.recommend(title, n)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    return {"query": title, "count": len(results), "recommendations": results}
# api/main.py
from recommender.recommend import fix_title

@app.get("/movies/search")
def search_movies(q: str):
    matches = recommender.movies[
        recommender.movies['title'].str.contains(q, case=False, na=False)
    ]['title'].head(10).tolist()
    
    # Fix display titles in autocomplete results too
    return {"results": [fix_title(t) for t in matches]}

@app.get("/weights/{title}")
def get_weights(title: str, top_n: int = 20):
    import re
    from recommender.recommend import unfix_title
    
    lookup = title if title in recommender.indices else unfix_title(title)
    if lookup not in recommender.indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    
    idx = recommender.indices[lookup]
    movie_vector = recommender.tfidf_matrix[idx]
    
    # Get feature names from the fitted vectorizer
    feature_names = recommender.tfidf.get_feature_names_out()
    
    # Get non-zero scores for this movie only
    cx = movie_vector.tocoo()
    scores = [(feature_names[j], round(float(v), 4)) for j, v in zip(cx.col, cx.data)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "title": title,
        "weights": [{"term": t, "weight": w} for t, w in scores[:top_n]]
    }