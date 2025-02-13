import sys
import argparse

from training import train_model
from inference import load_model_artifacts, recommend_movies_to_user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--json_file", type=str, help="Path to events JSON (for training).")
    parser.add_argument("--model_artifact", type=str, default="recommender.pkl")
    parser.add_argument("--user_id", type=str, help="User ID for inference mode.")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.json_file:
            print("Must provide --json_file for training.")
            sys.exit(1)
        # Train & save
        train_model(json_file_path=args.json_file, model_artifact_path=args.model_artifact)

    elif args.mode == "inference":
        if not args.user_id:
            print("Must provide --user_id for inference.")
            sys.exit(1)
        # Load artifacts
        artifacts = load_model_artifacts(args.model_artifact)
        # Recommend
        results = recommend_movies_to_user(args.user_id, artifacts, top_n=20)
        print(f"Recommendations for user {args.user_id}:")
        for movie_id, score in results:
            print(f"  {movie_id}: {score:.3f}")

if __name__ == "__main__":
    main()
