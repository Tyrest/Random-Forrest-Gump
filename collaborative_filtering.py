import joblib
from surprise import KNNWithMeans, accuracy

from recommendation_model import RecommendationModel


class CollaborativeFiltering(RecommendationModel):
    def __init__(self):
        super(CollaborativeFiltering, self).__init__()
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
        all_movie_ids = set(self.df["movie_id"].unique())
        rated_movies = set(self.df[self.df["user_id"] == user_id]["movie_id"])
        unrated_movies = all_movie_ids - rated_movies
        predictions = []
        for movie_id in unrated_movies:
            pred = self.algo.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k = predictions[:k]
        top_k_movie_ids = ",".join([str(movie_id) for movie_id, _ in top_k])
        top_k_with_ratings = ",".join(
            [f"{movie_id}: {rating:.2f}" for movie_id, rating in top_k]
        )
        return top_k_movie_ids, top_k_with_ratings

    def save(self, model_path):
        joblib.dump(self.algo, model_path)

    @classmethod
    def load(cls, model_path):
        algo = joblib.load(model_path)
        cf = CollaborativeFiltering()
        cf.algo = algo
        return cf


if __name__ == "__main__":
    # Example usage:
    cf = CollaborativeFiltering("data/rating_events.json")
    rmse, mae = cf.evaluate()
    user_id = "85304"
    movie_ids, movie_ratings = cf.recommend(user_id)
    print(f"Top 20 recommended movies for user {user_id}:")
    print(movie_ids)
    print(movie_ratings)
