from fastapi import FastAPI, HTTPException
from recommender.model import MovieRecommender

app = FastAPI(title="Movie Recommender API")

recommender = MovieRecommender(
    movies_path="data/movies_clean.csv",
    sim_path="data/cosine_sim.pkl"
)

@app.get("/")
def root():
    return {"message": "Movie Recommender API is running!"}

@app.get("/recommend/{title}")
def recommend(title: str, n: int = 10):
    results = recommender.recommend(title, n)
    
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{title}' not found in dataset"
        )
    
    return {
        "query": title,
        "count": len(results),
        "recommendations": results
    }

@app.get("/movies/search") 
def search_movies(q: str):
    matches = recommender.movies[
        recommender.movies['title'].str.contains(q, case=False, na=False)
    ]['title'].head(10).tolist()
    return {"results": matches}