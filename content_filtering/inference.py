import pickle

# https://en.wikipedia.org/wiki/Jaccard_index
def jaccard(setA, setB):
    if not setA and not setB:
        return 1.0  # if both are empty, treat as identical in that dimension
    return float(len(setA & setB)) / len(setA | setB)

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
    w_genre, w_lang, w_pc = 2.0
    w_lang = 1.0
    w_pc = 1.0
    w_adult, w_o_lang = 0.5
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
    return sim

def get_liked_movies(user_id, user_ratings):
    likes = set()
    if user_id not in user_ratings:
        return likes
    for m_id, (rating_value, _) in user_ratings[user_id].items():
        if rating_value >= 4:
            likes.add(m_id)
    return likes

def get_poorly_rated_movies(user_id, user_ratings):
    poor = set()
    if user_id not in user_ratings:
        return poor
    for m_id, (rating_value, _) in user_ratings[user_id].items():
        if rating_value <= 3:
            poor.add(m_id)
    return poor

def load_model_artifacts(model_artifact_path):
    with open(model_artifact_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

def recommend_movies_to_user(user_id, artifacts, top_n=20):
    """
    Returns a list of (movie_id, score).
    """
    movie_profiles = artifacts["movie_profiles"]
    user_watched   = artifacts["user_watched"]
    user_ratings   = artifacts["user_ratings"]
    global_stats   = artifacts["global_stats"]

    all_movie_ids = set(movie_profiles.keys())

    # Edge case: user not in user_watched dict
    if user_id not in user_watched:
        # e.g. a brand new user, fallback to top popularity
        return fallback_top_popular(movie_profiles, top_n)

    liked = get_liked_movies(user_id, user_ratings)
    poor = get_poorly_rated_movies(user_id, user_ratings)
    seen = user_watched[user_id]

    seen_not_poor = seen - poor

    if len(liked) > 0:
        reference_set = liked
    else:
        reference_set = seen_not_poor

    candidates = all_movie_ids - seen

    # If no reference set, fallback
    if len(reference_set) == 0:
        return fallback_top_popular(movie_profiles, top_n, candidates)

    scored_candidates = []
    for c_id in candidates:
        sim_scores = []
        for ref_id in reference_set:
            sim = compute_similarity(c_id, ref_id, movie_profiles, global_stats)
            sim_scores.append(sim)
        mean_sim = sum(sim_scores) / len(sim_scores)
        scored_candidates.append((c_id, mean_sim))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_n]

def fallback_top_popular(movie_profiles, top_n=20, candidates=None):
    """
    If the user has no watch history or everything is poorly rated,
    return the top_n movies by popularity (among all or among candidates).
    """
    if candidates is None:
        candidates = movie_profiles.keys()
    pop_scored = [(m_id, movie_profiles[m_id]["popularity"]) for m_id in candidates]
    pop_scored.sort(key=lambda x: x[1], reverse=True)
    return pop_scored[:top_n]

def inspect_movie(movie_id, artifacts):
    """
    For sanity checking:
      - Prints the specified movie's profile (all fields).
      - Then prints all movies in the model (with their profiles)
        ordered by similarity to the specified movie (including the movie itself).
    """
    movie_profiles = artifacts["movie_profiles"]
    global_stats = artifacts["global_stats"]

    if movie_id not in movie_profiles:
        print(f"Movie id '{movie_id}' not found in the model.")
        return

    # Print specified movie's profile.
    target_profile = movie_profiles[movie_id]
    print("=== Specified Movie Profile ===")
    print(f"Movie ID: {movie_id}")
    for key, value in target_profile.items():
        print(f"  {key}: {value}")
    print("=" * 40)

    # Compute similarity of every movie to the specified movie.
    similarities = []
    for mid, profile in movie_profiles.items():
        sim = compute_similarity(movie_id, mid, movie_profiles, global_stats)
        similarities.append((mid, sim, profile))

    # Sort by similarity descending (so the specified movie should be first with sim=1.0).
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\n=== All Movies Sorted by Similarity to Specified Movie ===")
    for mid, sim, profile in similarities:
        print(f"Movie ID: {mid} (Similarity: {sim:.3f})")
        for key, value in profile.items():
            print(f"  {key}: {value}")
        print("-" * 40)
