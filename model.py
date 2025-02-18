from fastapi import FastAPI
import uvicorn

from content_filtering.inference import load_model_artifacts, recommend_movies_to_user

app = FastAPI()

artifacts_path = "content_filtering/model/recommender1.pkl"
artifacts = load_model_artifacts(artifacts_path)

def content_filtering_predict(user_id: int):
    """
    Return a list of recommended movie IDs for the given user ID using the content-based filtering model.
    """
    return recommend_movies_to_user(user_id, artifacts, top_n=20)

@app.get("/recommend/{userid}")
def recommend(userid: int):
    """
    Retrieve recommendations from the loaded PyTorch model for the given userid.
    """
    recommended_items = content_filtering_predict(userid)
    return {"recommendations": recommended_items}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8083)
