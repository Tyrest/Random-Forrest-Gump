import time
import random

from collaborative_filtering import CollaborativeFiltering
from content_filtering import ContentFiltering
from utils import load_ratings

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.json"

trainset, testset = load_ratings(RATINGS_PATH)

collaborative_filtering_model = CollaborativeFiltering(RATINGS_PATH, MOVIES_PATH)
t0 = time.time()
collaborative_filtering_model.train(trainset)
print("Collaborative filtering training time: {:.2f}s".format(time.time() - t0))
rmse, mae = collaborative_filtering_model.evaluate(testset)

print("-" * 50)
print("Collaborative Filtering")
print("-" * 50)
print("RMSE: {:.4f}".format(rmse))
print("MAE: {:.4f}".format(mae))
print("-" * 50)

content_filtering_model = ContentFiltering(RATINGS_PATH, MOVIES_PATH)
t0 = time.time()
content_filtering_model.train(trainset)
print("Content filtering training time: {:.2f}s".format(time.time() - t0))
rmse, mae = content_filtering_model.evaluate(testset)

print("-" * 50)
print("Content Filtering")
print("-" * 50)
print("RMSE: {:.4f}".format(rmse))
print("MAE: {:.4f}".format(mae))
print("-" * 50)


n_runs = 10
n_predictions = 10

unique_user_ids = set([row[0] for row in testset])
test_user_ids = random.sample(unique_user_ids, n_predictions)

collab_run_times = []
content_run_times = []

for _ in range(n_runs):
    start_time = time.time()
    for user_id in test_user_ids:
        collaborative_filtering_model.recommend(user_id)
    run_time = time.time() - start_time
    collab_run_times.append(run_time)


    start_time = time.time()
    for user_id in test_user_ids:
        content_filtering_model.recommend(user_id)
    run_time = time.time() - start_time
    content_run_times.append(run_time)

avg_collab_time = sum(collab_run_times) / len(collab_run_times)
avg_content_time = sum(content_run_times) / len(content_run_times)

print("Average time for {} predictions (over {} runs) - Collaborative Filtering: {:.4f}s"
      .format(n_predictions, n_runs, avg_collab_time))

print("Average time for {} predictions (over {} runs) - Content Filtering: {:.4f}s"
      .format(n_predictions, n_runs, avg_content_time))
