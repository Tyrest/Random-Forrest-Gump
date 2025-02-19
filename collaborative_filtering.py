import joblib
import pandas as pd
from surprise import KNNWithMeans, accuracy

from recommendation_model import RecommendationModel


class CollaborativeFiltering(RecommendationModel):
    def __init__(self, ratings_path, movies_path):
        super(CollaborativeFiltering, self).__init__()
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_json(self.movies_path)
        self.sim_options = {"name": "cosine", "user_based": True}

    def train(self, trainset):
        self.algo = KNNWithMeans(sim_options=self.sim_options)
        self.algo.fit(trainset)

    def evaluate(self, testset):
        if not hasattr(self, "algo"):
            raise ValueError("Model not trained yet. Call train() first.")
        predictions = self.algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae

    def recommend(self, user_id, k=20):
        all_movie_ids = set(self.movies_df["id"])
        watched_movies = set(
            self.ratings_df[self.ratings_df["userid"] == user_id]["movieid"]
        )
        unwatched_movies = all_movie_ids - watched_movies
        predictions = []
        for movie_id in unwatched_movies:
            pred = self.algo.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]

    def save(self, model_path):
        joblib.dump(
            {
                "algo": self.algo,
                "ratings_path": self.ratings_path,
                "movies_path": self.movies_path,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path):
        data = joblib.load(model_path)
        cf = CollaborativeFiltering(data["ratings_path"], data["movies_path"])
        cf.algo = data["algo"]
        return cf
