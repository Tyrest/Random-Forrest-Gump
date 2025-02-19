import uvicorn
from fastapi import FastAPI

from collaborative_filtering import CollaborativeFiltering
from content_filtering import ContentFiltering

app = FastAPI()

model = CollaborativeFiltering.load("models/collaborative_filtering.pkl")


@app.get("/recommend/{userid}")
def recommend(userid: int):
    """
    Retrieve recommendations from the loaded PyTorch model for the given userid.
    """
    return {"recommendations": model.recommend(userid)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8083)
