import argparse
import pandas as pd

from inference import load_model_artifacts, recommend_movies_to_user

def evaluate_model(model_artifact_path, test_movie_play_file, test_rating_file):
    """
    For each user in the test rating events:
      - Get top 20 recommendations.
      - For each test rating event, if the rating >= 4 then the movie should be recommended;
        if rating <= 3 then it should not be recommended.
      - Count TP, FP, TN, FN and compute overall accuracy.
    """
    artifacts = load_model_artifacts(model_artifact_path)
    
    test_rating_df = pd.read_json(test_rating_file, orient="records")
    test_rating_df["user_id"] = test_rating_df["user_details"].apply(lambda d: d.get("user_id"))
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    grouped = test_rating_df.groupby("user_id")
    
    for user_id, group in grouped:
        recommendations = recommend_movies_to_user(user_id, artifacts, top_n=20)
        recommended_ids = set(movie_id for movie_id, score in recommendations)

        for idx, row in group.iterrows():
            movie_id = row["movieid"]
            rating = float(row["rating"])
            if rating >= 4:
                if movie_id in recommended_ids:
                    TP += 1
                else:
                    FN += 1
            elif rating <= 3:
                if movie_id in recommended_ids:
                    FP += 1
                else:
                    TN += 1

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0

    print("Evaluation Metrics:")
    print("-------------------")
    print("True Positives:  ", TP)
    print("False Positives: ", FP)
    print("True Negatives:  ", TN)
    print("False Negatives: ", FN)
    print("Overall Accuracy: {:.3f}".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the content filtering model on test data."
    )
    parser.add_argument("--model_artifact", type=str, required=True,
                        help="Path to model artifact (pickle file)")
    parser.add_argument("--test_movie_play_file", type=str, required=True,
                        help="Path to test movie_play events JSON file")
    parser.add_argument("--test_rating_file", type=str, required=True,
                        help="Path to test rating events JSON file")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_artifact, args.test_movie_play_file, args.test_rating_file)
