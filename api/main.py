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

@app.get("/movies/search")
def search_movies(q: str):
    matches = recommender.movies[
        recommender.movies['title'].str.contains(q, case=False, na=False)
    ]['title'].head(10).tolist()
    return {"results": matches}