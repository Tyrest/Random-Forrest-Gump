import time
from fastapi import FastAPI
import uvicorn
import requests

app = FastAPI()

def get_recommendations(userid: str):
    """
    Calls the local model service API (model_api.py) to get recommendations
    for the given userid. Returns a list of movie IDs.
    """
    url = f"http://127.0.0.1:8083/recommend/{userid}"
    response = requests.get(url)
    data = response.json()
    return data["recommendations"]

@app.get("/recommend/{userid}")
def recommend(userid: str):
    """
    Respond with a single-line comma-separated list of up to 20 movie IDs,
    from highest to lowest recommendation.
    """
    recommended_movie_ids = get_recommendations(userid)
    response_str = ",".join(map(str, recommended_movie_ids))
    return response_str

if __name__ == "__main__":
    # It proxies recommendation requests to the local PyTorch model on port 8083.
    uvicorn.run(app, host="0.0.0.0", port=8082)
