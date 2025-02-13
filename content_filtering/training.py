import datetime
from collections import defaultdict
import pickle

from data_loading import load_events_from_json, split_events

def parse_date_to_year(date_str):
    if not date_str or date_str.lower() == 'null':
        return None
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").year

def build_movie_profile(movie_details):
    """
    Transform the 'movie_details' dict into a standard profile dict.
    """
    return {
        "adult": 1 if (str(movie_details.get("adult")).lower() == "true") else 0,
        "genres": set(g["id"] for g in movie_details.get("genres", [])),
        "original_language": movie_details.get("original_language", None),
        "popularity": float(movie_details.get("popularity", 0.0)),
        "production_companies": set(pc["id"] for pc in movie_details.get("production_companies", [])),
        "release_year": parse_date_to_year(movie_details.get("release_date", None)),
        "revenue": float(movie_details.get("revenue", 0.0)),
        "runtime": float(movie_details.get("runtime", 0.0)),
        "spoken_languages": set(lang["iso_639_1"] for lang in movie_details.get("spoken_languages", [])),
        "vote_average": float(movie_details.get("vote_average", 0.0))
    }

def train_model(json_file_path, model_artifact_path="recommender.pkl"):
    """
    Main training entry point:
      1) Reads raw events from JSON.
      2) Splits into movie_plays and ratings (Pandas) DataFrames.
      3) Builds user_watched, user_ratings, and movie_profiles.
      4) Computes optional global stats for numeric feature scaling.
      5) Saves artifacts (as a dict) to a pickle file.
    """

    # Load the events DataFrame
    df = load_events_from_json(json_file_path)

    # Split into two DataFrames
    movie_plays_df, ratings_df = split_events(df)

    # Build movie_profiles from the `movie_play` events
    movie_profiles = {}
    for idx, row in movie_plays_df.iterrows():
        m_id = row["movie_details"]["id"]
        if m_id not in movie_profiles:
            movie_profiles[m_id] = build_movie_profile(row["movie_details"])

    # Build user_watched
    user_watched = defaultdict(set)
    for idx, row in movie_plays_df.iterrows():
        user_id = row["userid"]
        m_id = row["movie_details"]["id"]
        user_watched[user_id].add(m_id)

    # Build user_ratings from `rating` events (keep only the most recent rating if multiple)
    user_ratings = defaultdict(dict)
    for idx, row in ratings_df.iterrows():
        user_id = row["userid"]
        movie_id = row["movieid"]
        rating_value = float(row["rating"])
        # TODO: need to actually parse this timestamp because right now the timestamp is the prefix of a longer string
        rating_timestamp = row["raw"] 
        # Check if we already have a rating for this user/movie
        if movie_id not in user_ratings[user_id]:
            user_ratings[user_id][movie_id] = (rating_value, rating_timestamp)
        else:
            _, existing_ts = user_ratings[user_id][movie_id]
            if rating_timestamp > existing_ts:
                user_ratings[user_id][movie_id] = (rating_value, rating_timestamp)

    # Compute global min/max for numeric features across all movies to use later for normalization
    numeric_keys = ["popularity", "revenue", "runtime", "vote_average", "release_year"]
    global_stats = {}
    for key in numeric_keys:
        vals = []
        for m_id, profile in movie_profiles.items():
            val = profile[key]
            if val is not None:
                vals.append(val)
        if len(vals) == 0:
            # fallback
            global_stats[f"{key}_min"] = 0
            global_stats[f"{key}_max"] = 1
        else:
            global_stats[f"{key}_min"] = min(vals)
            global_stats[f"{key}_max"] = max(vals)

    # For now, the "model" is being saved by pickling the following Python dictionary
    model_artifacts = {
        "movie_profiles": dict(movie_profiles),
        "user_watched": dict(user_watched),
        "user_ratings": {u: dict(rates) for u, rates in user_ratings.items()},
        "global_stats": global_stats
    }

    with open(model_artifact_path, "wb") as f:
        pickle.dump(model_artifacts, f)

    print(f"Model training complete. Artifacts saved to {model_artifact_path}")
