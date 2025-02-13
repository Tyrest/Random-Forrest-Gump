import pandas as pd

def load_events_from_json(json_file_path):
    """
    Reads a JSON file of events (both 'movie_play' and 'rating').
    Returns a Pandas DataFrame with each record as one row.
    """
    df = pd.read_json(json_file_path, orient="records")
    return df

def split_events(df):
    """
    Splits the events DataFrame into two separate DataFrames:
      - movie_plays_df
      - ratings_df
    """
    movie_plays_df = df[df["type"] == "movie_play"].copy()
    ratings_df = df[df["type"] == "rating"].copy()
    return movie_plays_df, ratings_df
