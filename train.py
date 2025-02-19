import os

from collaborative_filtering import CollaborativeFiltering
from content_filtering import ContentFiltering
from utils import load_ratings

RATING_EVENTS_PATH = "data/rating_events.json"
MOVIES_PATH = "data/movies.json"
WATCHED_PATH = "data/watched.json"
COLLABORATIVE_FILTERING_PATH = "models/collaborative_filtering.pkl"
CONTENT_FILTERING_PATH = "models/content_filtering.pkl"

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    trainset, testset = load_ratings(RATING_EVENTS_PATH, test_size=0)

    collaborative_filtering_model = CollaborativeFiltering(MOVIES_PATH, WATCHED_PATH)
    collaborative_filtering_model.train(trainset)
    collaborative_filtering_model.save(COLLABORATIVE_FILTERING_PATH)

    content_filtering_model = ContentFiltering(MOVIES_PATH, WATCHED_PATH)
    content_filtering_model.train(trainset)
    content_filtering_model.save(CONTENT_FILTERING_PATH)

    print(
        "Sanity check to see if the model outputs two different recommendations for two different users in the trainset."
    )
    print(collaborative_filtering_model.recommend(6566))
    print(collaborative_filtering_model.recommend(32206))

    print()

    print(content_filtering_model.recommend(6566))
    print(content_filtering_model.recommend(32206))

    print()

    print("Then test loading the models back from disk.")
    cf = CollaborativeFiltering.load(COLLABORATIVE_FILTERING_PATH)
    print(cf.recommend(6566))
    print(cf.recommend(32206))

    print()

    cf = ContentFiltering.load(CONTENT_FILTERING_PATH)
    print(cf.recommend(6566))
    print(cf.recommend(32206))
