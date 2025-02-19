#!/usr/bin/env python3
import json, sys, pandas as pd


def deduplicate_with_pandas(input_file, output_file):
    # Load the JSON array from the file.
    with open(input_file, "r") as f:
        data = json.load(f)

    # Flatten the JSON structure so that nested fields become columns.
    df = pd.json_normalize(data)

    # Ensure the required columns exist.
    if "user_details.user_id" not in df.columns or "movieid" not in df.columns:
        print("Required fields 'user_details.user_id' and/or 'movieid' not found.")
        sys.exit(1)

    # Drop duplicates based on the combination of user_details.user_id and movieid.
    dedup_df = df.drop_duplicates(subset=["user_details.user_id", "movieid"])

    # Convert the deduplicated DataFrame back to a list of dicts.
    dedup_data = dedup_df.to_dict(orient="records")

    # Write the deduplicated data as a JSON array.
    with open(output_file, "w") as f:
        json.dump(dedup_data, f, indent=2)

    print(f"Deduplicated from {len(data)} to {len(dedup_data)} records.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dedup_ratings_pandas.py input.json output.json")
        sys.exit(1)
    deduplicate_with_pandas(sys.argv[1], sys.argv[2])
