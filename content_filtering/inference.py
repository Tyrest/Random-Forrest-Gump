import argparse
from content_filtering import ContentFiltering


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for a user or inspect a movie."
    )
    parser.add_argument(
        "--mode",
        choices=["recommend", "inspect"],
        required=True,
        help="Action to perform.",
    )
    parser.add_argument(
        "--model_artifact",
        type=str,
        default="recommender.pkl",
        help="Path to model artifact.",
    )
    parser.add_argument("--user_id", type=str, help="User ID for recommendation.")
    parser.add_argument("--movie_id", type=str, help="Movie ID for inspection.")
    args = parser.parse_args()

    model = ContentFiltering.load(args.model_artifact)
    if args.mode == "recommend":
        if not args.user_id:
            print("Must provide --user_id for recommendation.")
            return
        recs = model.recommend(args.user_id, k=20)
        print(f"Recommendations for user {args.user_id}:")
        for movie_id, score in recs:
            print(f"  {movie_id}: {score:.3f}")
    elif args.mode == "inspect":
        if not args.movie_id:
            print("Must provide --movie_id for inspection.")
            return
        model.inspect_movie(args.movie_id)


if __name__ == "__main__":
    main()
