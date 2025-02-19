import argparse
import pandas as pd
from content_filtering import ContentFiltering


def evaluate_model(model_artifact, test_movie_play_file, test_rating_file):
    model = ContentFiltering.load(model_artifact)
    test_rating_df = pd.read_json(test_rating_file, orient="records")
    test_rating_df["user_id"] = test_rating_df["user_details"].apply(
        lambda d: d.get("user_id")
    )
    TP = FP = TN = FN = 0
    for user_id, group in test_rating_df.groupby("user_id"):
        recommendations = model.recommend(user_id, k=20)
        rec_ids = set(r for (r, _) in recommendations)
        for idx, row in group.iterrows():
            movie_id = row["movieid"]
            rating = float(row["rating"])
            if rating >= 4:
                if movie_id in rec_ids:
                    TP += 1
                else:
                    FN += 1
            elif rating <= 3:
                if movie_id in rec_ids:
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
        description="Evaluate the content filtering model."
    )
    parser.add_argument(
        "--model_artifact",
        type=str,
        required=True,
        help="Path to model artifact (pickle file)",
    )
    parser.add_argument(
        "--test_movie_play_file",
        type=str,
        required=True,
        help="Path to test movie_play events JSON file",
    )
    parser.add_argument(
        "--test_rating_file",
        type=str,
        required=True,
        help="Path to test rating events JSON file",
    )
    args = parser.parse_args()
    evaluate_model(
        args.model_artifact, args.test_movie_play_file, args.test_rating_file
    )
