import uvicorn
from fastapi import FastAPI

from collaborative_filtering import CollaborativeFiltering
from content_filtering import ContentFiltering

app = FastAPI()

model = CollaborativeFiltering.load("models/collaborative_filtering.pkl")


@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    """
    Retrieve recommendations from the loaded PyTorch model for the given userid.
    """
    recommendations = model.recommend(user_id)
    return {"recommendations": [movie_id for movie_id, pred in recommendations]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8083)
