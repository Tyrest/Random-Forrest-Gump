# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12wvaa2Vw2kQgSnUNcTHtBBCqITgk0iFf
"""

# Commented out IPython magic to ensure Python compatibility.
# import re
# import numpy as np
# import pandas as pd
# from math import sqrt
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# %pip install scikit-surprise
import re
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

def parse_log_file(file_path):

    rating_events = []

    rating_pattern = re.compile(r'GET /rate/([^=]+)=(\d+)')

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Each line is expected to have 3 comma-separated parts
            parts = line.split(',')
            if len(parts) != 3:
                continue
            timestamp, user_id, request = parts
            user_id = user_id.strip()

            # Check for rating event using regex
            rating_match = rating_pattern.search(request)
            if rating_match:
                movie_id = rating_match.group(1)
                rating = int(rating_match.group(2))
                rating_events.append({
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating
                })
            # Optionally, you can also handle watch events if you want to derive implicit feedback.
            # For now, we only use explicit rating events.

    return pd.DataFrame(rating_events)

# 1. Parse the Log File
log_file_path = '/content/drive/MyDrive/test_3.txt'
df = parse_log_file(log_file_path)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

#2. Split the Data
trainset, testset = train_test_split(data, test_size=0.2)

sim_options = {
    'name': 'cosine',
    'user_based': True
}

algo = KNNWithMeans(sim_options=sim_options)
algo.fit(trainset)

def get_top_n(predictions, n=20):

    # Create a dictionary of predictions for each user
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

predictions = algo.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Compute MAE
mae = accuracy.mae(predictions)
print(f"MAE: {mae}")

def get_top_20_recommendations_for_user(user_id):
    all_movie_ids = set(df['movie_id'].unique())

    # Get the list of movies that the user has already rated
    rated_movies = set(df[df['user_id'] == user_id]['movie_id'])

    # Get the list of unrated movies for this user
    unrated_movies = all_movie_ids - rated_movies

    # Generate predictions for all unrated movies
    predictions = []
    for movie_id in unrated_movies:
        prediction = algo.predict(user_id, movie_id)
        predictions.append((movie_id, prediction.est))

    # Sort the predictions
    predictions.sort(key=lambda x: x[1], reverse = True)

    top_20 = predictions[:20]

    # Get the top 20 movie IDs
    top_20_movie_ids = [str(movie_id) for movie_id, _ in top_20]
    top_20_movie_ids_output = ",".join(top_20_movie_ids)

    top_20_with_ratings = [f"{movie_id}: {rating:.2f}" for movie_id, rating in top_20]
    top_20_with_ratings_output = ",".join(top_20_with_ratings)

    return top_20_movie_ids_output, top_20_with_ratings_output

user_id = '85304'  # The user ID for whom we want the recommendations
top_20_movie_ids, top_20_with_ratings = get_top_20_recommendations_for_user(user_id)

print(f"Top 20 recommended movies for user {user_id}:")
print(top_20_movie_ids)
print(top_20_with_ratings)

df['user_id'][0]