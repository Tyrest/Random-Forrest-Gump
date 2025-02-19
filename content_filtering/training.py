import argparse
from content_filtering import ContentFiltering


def main():
    parser = argparse.ArgumentParser(description="Train content filtering model.")
    parser.add_argument(
        "--movie_play_file",
        type=str,
        required=True,
        help="Path to movie_play events JSON file.",
    )
    parser.add_argument(
        "--rating_file",
        type=str,
        required=True,
        help="Path to rating events JSON file.",
    )
    parser.add_argument(
        "--model_artifact",
        type=str,
        default="recommender.pkl",
        help="Path to save the model artifact.",
    )
    args = parser.parse_args()
    model = ContentFiltering.train(args.movie_play_file, args.rating_file)
    model.save(args.model_artifact)


if __name__ == "__main__":
    main()
