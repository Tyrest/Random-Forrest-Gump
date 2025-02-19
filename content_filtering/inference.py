import argparse
from content_filtering import ContentFiltering


def scale_numeric(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def compute_similarity(movieA_id, movieB_id, movie_profiles, global_stats):
    A = movie_profiles[movieA_id]
    B = movie_profiles[movieB_id]

    def numeric_sim(key):
        a_val = A[key] if A[key] else 0
        b_val = B[key] if B[key] else 0
        a_scaled = scale_numeric(a_val, global_stats[f"{key}_min"], global_stats[f"{key}_max"])
        b_scaled = scale_numeric(b_val, global_stats[f"{key}_min"], global_stats[f"{key}_max"])
        return 1.0 - abs(a_scaled - b_scaled)

    pop_sim = numeric_sim("popularity")
    rev_sim = numeric_sim("revenue")
    run_sim = numeric_sim("runtime")
    vote_sim = numeric_sim("vote_average")
    year_sim = numeric_sim("release_year")

    genre_sim = jaccard(A["genres"], B["genres"])
    lang_sim  = jaccard(A["spoken_languages"], B["spoken_languages"])
    pc_sim    = jaccard(A["production_companies"], B["production_companies"])

    adult_sim = 1.0 if A["adult"] == B["adult"] else 0.0
    orig_lang_sim = 1.0 if A["original_language"] == B["original_language"] else 0.0

    w_pop = 1.0
    w_rev= 1.0
    w_run= 0.5
    w_vote = 1.0
    w_year = 0.5
    w_genre = 2.0
    w_lang = 2.0
    w_pc = 2.0
    w_lang = 1.0
    w_pc = 1.0
    w_adult = 0.5
    w_o_lang = 0.5
    w_o_lang = 0.5

    sim = (
        w_pop * pop_sim +
        w_rev * rev_sim +
        w_run * run_sim +
        w_vote * vote_sim +
        w_year * year_sim +
        w_genre * genre_sim +
        w_lang  * lang_sim +
        w_pc    * pc_sim +
        w_adult * adult_sim +
        w_o_lang * orig_lang_sim
    ) / (
        w_pop + w_rev + w_run + w_vote + w_year +
        w_genre + w_lang + w_pc + w_adult + w_o_lang
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
