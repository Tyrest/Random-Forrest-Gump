import datetime
import math
import pickle
from collections import defaultdict

import pandas as pd

from recommendation_model import RecommendationModel

NUMERIC_KEYS = [
    "popularity",
    "revenue",
    "runtime",
    "vote_average",
    "release_year",
]


class ContentFiltering(RecommendationModel):
    def __init__(self, movies_path, watched_path):
        self.movies_path = movies_path
        self.watched_path = watched_path
        self.pop_scored = None

    def train(self, trainset):
        movies_df = pd.read_json(self.movies_path)
        movie_profiles = {}

        for _, row in movies_df.iterrows():
            m_id = row["id"]
            if m_id not in movie_profiles:
                movie_profiles[m_id] = ContentFiltering._build_movie_profile(row)

        watched_df = pd.read_json(self.watched_path)
        user_watched = defaultdict(set)

        for _, row in watched_df.iterrows():
            user_id = row["userid"]
            m_id = row["movie_id"]
            user_watched[user_id].add(m_id)

        user_ratings = defaultdict(dict)

        for user_id, movie_id, rating in trainset.build_testset():
            user_ratings[user_id][movie_id] = rating

        global_stats = {}

        for key in NUMERIC_KEYS:
            vals = [
                profile[key]
                for profile in movie_profiles.values()
                if profile[key] is not None
            ]
            if not vals:
                global_stats[f"{key}_min"], global_stats[f"{key}_max"] = 0, 1
            else:
                global_stats[f"{key}_min"], global_stats[f"{key}_max"] = min(vals), max(
                    vals
                )

        self.movie_profiles = movie_profiles
        self.user_watched = user_watched
        self.user_ratings = user_ratings
        self.global_stats = global_stats

    def evaluate(self, testset):
        predictions = {}

        rmses = []
        maes = []

        for user_id, movie_id, rating in testset:
            if user_id not in predictions:
                pred = self._get_predictions(user_id)
                predictions[user_id] = {m_id: score for m_id, score in pred}
            rmses.append((predictions[user_id][movie_id] - rating) ** 2)
            maes.append(abs(predictions[user_id][movie_id] - rating))

        rmse = (sum(rmses) / len(rmses)) ** 0.5
        mae = sum(maes) / len(maes)
        return rmse, mae

    def recommend(self, user_id, k=20):
        predictions = self._get_predictions(user_id)
        return predictions[:k]

    def inspect_movie(self, movie_id):
        if movie_id not in self.movie_profiles:
            print(f"Movie id '{movie_id}' not found in the model.")
            return
        target = self.movie_profiles[movie_id]
        print("=== Specified Movie Profile ===")
        print(f"Movie ID: {movie_id}")
        for key, value in target.items():
            print(f"  {key}: {value}")
        print("=" * 40)
        sims = []
        for mid in self.movie_profiles:
            sim = self._compute_similarity(movie_id, mid)
            sims.append((mid, sim, self.movie_profiles[mid]))
        sims.sort(key=lambda x: x[1], reverse=True)
        print("\n=== All Movies Sorted by Similarity ===")
        for mid, sim, profile in sims:
            print(f"Movie ID: {mid} (Similarity: {sim:.3f})")
            for key, value in profile.items():
                print(f"  {key}: {value}")
            print("-" * 40)

    def save(self, model_path="content_filtering.pkl"):
        artifacts = {
            "movie_profiles": self.movie_profiles,
            "user_watched": self.user_watched,
            "user_ratings": self.user_ratings,
            "global_stats": self.global_stats,
        }
        with open(model_path, "wb") as f:
            pickle.dump(artifacts, f)

    @classmethod
    def load(cls, model_path):
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
        model = cls("", "")
        model.movie_profiles = artifacts["movie_profiles"]
        model.user_watched = artifacts["user_watched"]
        model.user_ratings = artifacts["user_ratings"]
        model.global_stats = artifacts["global_stats"]
        return model

    def _compute_similarity(self, movieA_id, movieB_id):
        A = self.movie_profiles[movieA_id]
        B = self.movie_profiles[movieB_id]

        def numeric_sim(key):
            a_val = A[key] if A[key] else 0
            b_val = B[key] if B[key] else 0
            a_scaled = self._scale_numeric(
                a_val, self.global_stats[f"{key}_min"], self.global_stats[f"{key}_max"]
            )
            b_scaled = self._scale_numeric(
                b_val, self.global_stats[f"{key}_min"], self.global_stats[f"{key}_max"]
            )
            return 1.0 - abs(a_scaled - b_scaled)

        pop_sim = numeric_sim("popularity")
        rev_sim = numeric_sim("revenue")
        run_sim = numeric_sim("runtime")
        vote_sim = numeric_sim("vote_average")
        year_sim = numeric_sim("release_year")
        genre_sim = self._jaccard(A["genres"], B["genres"])
        lang_sim = self._jaccard(A["spoken_languages"], B["spoken_languages"])
        pc_sim = self._jaccard(A["production_companies"], B["production_companies"])
        adult_sim = 1.0 if A["adult"] == B["adult"] else 0.0
        orig_lang_sim = 1.0 if A["original_language"] == B["original_language"] else 0.0
        w = {
            "pop": 1.0,
            "rev": 1.0,
            "run": 0.5,
            "vote": 1.0,
            "year": 0.5,
            "genre": 1.0,
            "lang": 1.0,
            "pc": 1.0,
            "adult": 0.5,
            "o_lang": 0.5,
        }
        sim = (
            w["pop"] * pop_sim
            + w["rev"] * rev_sim
            + w["run"] * run_sim
            + w["vote"] * vote_sim
            + w["year"] * year_sim
            + w["genre"] * genre_sim
            + w["lang"] * lang_sim
            + w["pc"] * pc_sim
            + w["adult"] * adult_sim
            + w["o_lang"] * orig_lang_sim
        )
        total_weight = sum(w.values())
        return sim / total_weight

    def _get_liked_movies(self, user_id):
        likes = set()
        if user_id not in self.user_ratings:
            return likes
        for m_id, rating_value in self.user_ratings[user_id].items():
            if rating_value >= 4:
                likes.add(m_id)
        return likes

    def _get_poorly_rated_movies(self, user_id):
        poor = set()
        if user_id not in self.user_ratings:
            return poor
        for m_id, rating_value in self.user_ratings[user_id].items():
            if rating_value <= 3:
                poor.add(m_id)
        return poor

    def _normalize_scores(self, scored):
        scores = [s for _, s in scored]
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            max_score += 1

        def _normalize(score):
            """Normalizes score to 1-5 scale."""
            return 1 + (score - min_score) * 4 / (max_score - min_score)

        return [(m_id, _normalize(s)) for m_id, s in scored]

    def _fallback_top_popular(self, candidates):
        if self.pop_scored:
            return self.pop_scored
        self.pop_scored = [
            (m_id, math.log(self.movie_profiles[m_id]["popularity"]))
            for m_id in candidates
        ]
        self._normalize_scores(self.pop_scored)
        self.pop_scored.sort(key=lambda x: x[1], reverse=True)
        return self.pop_scored

    def _get_predictions(self, user_id):
        all_movie_ids = set(self.movie_profiles.keys())
        if user_id not in self.user_watched:
            return self._fallback_top_popular(all_movie_ids)
        liked = self._get_liked_movies(user_id)
        poor = self._get_poorly_rated_movies(user_id)
        seen = self.user_watched[user_id]
        seen_not_poor = seen - poor
        reference_set = liked if liked else seen_not_poor
        candidates = all_movie_ids - seen
        if not reference_set:
            return self._fallback_top_popular(candidates)
        scored = []
        for candidate in candidates:
            sims = [self._compute_similarity(candidate, ref) for ref in reference_set]
            scored.append((candidate, sum(sims) / len(sims)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return self._normalize_scores(scored)

    @staticmethod
    def _parse_date_to_year(date_str):
        if not date_str or str(date_str).lower() == "null":
            return None
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").year

    @staticmethod
    def _build_movie_profile(movie_details):
        return {
            "adult": 1 if str(movie_details.get("adult")).lower() == "true" else 0,
            "genres": set(g["id"] for g in movie_details.get("genres", [])),
            "original_language": movie_details.get("original_language"),
            "popularity": float(movie_details.get("popularity", 0.0)),
            "production_companies": set(
                pc["id"] for pc in movie_details.get("production_companies", [])
            ),
            "release_year": ContentFiltering._parse_date_to_year(
                movie_details.get("release_date")
            ),
            "revenue": float(movie_details.get("revenue", 0.0)),
            "runtime": float(movie_details.get("runtime", 0.0)),
            "spoken_languages": set(
                lang["iso_639_1"] for lang in movie_details.get("spoken_languages", [])
            ),
            "vote_average": float(movie_details.get("vote_average", 0.0)),
        }

    @staticmethod
    def _jaccard(setA, setB):
        if not setA and not setB:
            return 1.0
        return float(len(setA & setB)) / len(setA | setB)

    @staticmethod
    def _scale_numeric(value, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)
