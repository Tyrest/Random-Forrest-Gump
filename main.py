import logging

import requests
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

app = FastAPI()


def get_recommendations(user_id: int):
    """
    Calls the local model service API (model_api.py) to get recommendations
    for the given user_id. Returns a list of movie IDs.
    """
    url = f"http://127.0.0.1:8083/recommend/{user_id}"
    response = requests.get(url)
    data = response.json()
    return data["recommendations"]


@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    """
    Respond with a single-line comma-separated list of up to 20 movie IDs,
    from highest to lowest recommendation.
    """
    recommended_movie_ids = get_recommendations(user_id)
    response_str = ",".join(map(str, recommended_movie_ids))
    logger.debug(f"Recommendations for user {user_id}: {response_str}")
    return response_str


if __name__ == "__main__":
    # It proxies recommendation requests to the local PyTorch model on port 8083.
    uvicorn.run(app, host="0.0.0.0", port=8082)
