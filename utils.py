import json

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


def parse_rating_events(file_path):
    with open(file_path, "r") as f:
        events = json.load(f)
    rating_events = []
    for event in events:
        if event.get("type") != "rating":
            continue
        raw = event.get("raw", "")
        timestamp = raw.split(",")[0]  # Extract timestamp from raw
        user_id = event["user_details"]["user_id"]
        movie_id = event["movieid"]
        rating = int(event["rating"])
        rating_events.append(
            {
                "timestamp": timestamp,
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
            }
        )
    return pd.DataFrame(rating_events)


def load_ratings(rating_file_path, test_size=0.2):
    df = parse_rating_events(rating_file_path)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    if test_size == 0:
        return data.build_full_trainset(), None
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset
