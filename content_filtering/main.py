import sys
import argparse
from content_filtering import ContentFiltering


def main():
    parser = argparse.ArgumentParser(
        description="Content filtering tool: train, inference, or inspect."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "inspect"],
        required=True,
        help="Mode to run.",
    )
    parser.add_argument(
        "--movie_play_file",
        type=str,
        help="Path to movie_play events JSON file (for training).",
    )
    parser.add_argument(
        "--rating_file",
        type=str,
        help="Path to rating events JSON file (for training).",
    )
    parser.add_argument(
        "--model_artifact",
        type=str,
        default="recommender.pkl",
        help="Path to model artifact.",
    )
    parser.add_argument("--user_id", type=str, help="User ID for inference.")
    parser.add_argument("--movie_id", type=str, help="Movie ID for inspect mode.")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.movie_play_file or not args.rating_file:
            print("Must provide --movie_play_file and --rating_file for training.")
            sys.exit(1)
        model = ContentFiltering.train(args.movie_play_file, args.rating_file)
        model.save(args.model_artifact)
    elif args.mode == "inference":
        if not args.user_id:
            print("Must provide --user_id for inference.")
            sys.exit(1)
        model = ContentFiltering.load(args.model_artifact)
        recommendations = model.recommend(args.user_id, k=20)
        print(f"Recommendations for user {args.user_id}:")
        for movie_id, score in recommendations:
            print(f"  {movie_id}: {score:.3f}")
    elif args.mode == "inspect":
        if not args.movie_id:
            print("Must provide --movie_id for inspect mode.")
            sys.exit(1)
        model = ContentFiltering.load(args.model_artifact)
        model.inspect_movie(args.movie_id)


if __name__ == "__main__":
    main()
