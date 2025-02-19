import json
from datetime import datetime

import polars as pl
import requests
from confluent_kafka import OFFSET_BEGINNING, Consumer, KafkaError, TopicPartition
from rich.live import Live
from rich.progress import track
from rich.prompt import Confirm
from rich.text import Text

consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "tyler_data_collector",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)
topic = "movielog22"  # for team 22


def on_assign(consumer, partitions):
    # For each assigned partition, reset offset to the beginning
    reset_partitions = [
        TopicPartition(tp.topic, tp.partition, OFFSET_BEGINNING) for tp in partitions
    ]
    consumer.assign(reset_partitions)


consumer.subscribe([topic], on_assign=on_assign)


def get_log_type(line):
    if "recommendation request" in line:
        return "recommendation"
    elif "GET /data/m/" in line:
        return "movie_play"
    elif "GET /rate/" in line:
        return "rating"
    else:
        return "unknown"


def get_movie_details(movieid):
    url = f"http://128.2.204.215:8080/movie/{movieid}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching movie details for {movieid}: {response.status_code}")
        return None


def get_user_details(userid):
    url = f"http://128.2.204.215:8080/user/{userid}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching user details for {userid}: {response.status_code}")
        return None


def process_log_line(log_line):
    type = get_log_type(log_line)

    if type == "unknown":
        raise ValueError(f"Unknown log type: {log_line}")

    parsed = {}

    if type == "rating":
        try:
            parts = log_line.split(",")
            timestamp = parts[0].strip()
            userid = parts[1].strip()
            request_part = parts[2].strip()
            rate_data = request_part.replace("GET /rate/", "")
            movieid, rating = rate_data.split("=")
            parsed["timestamp"] = timestamp
            parsed["userid"] = userid
            parsed["movieid"] = movieid
            parsed["rating"] = rating
        except Exception as e:
            print("Error parsing rating log:", e)

    return type, parsed


def store_data(data, filename, sort=None):
    df = pl.DataFrame(data)
    if sort:
        df = df.sort(sort)
    df.write_csv(filename)
    return df


def store_data_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    return data


if __name__ == "__main__":
    if not Confirm.ask(
        "This script will overwrite ratings.csv, users.csv, and movies.json. Continue?"
    ):
        print("Exiting...")

    print("Starting Kafka consumer for topic:", topic)
    rating_events = []

    try:
        with Live(
            Text("Current Rows: 0\nLatest Timestamp:"), refresh_per_second=4
        ) as live:
            while True:
                msg = consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print("Error: {}".format(msg.error()))
                        break

                log_line = msg.value().decode("utf-8")
                type, processed = process_log_line(log_line)

                if type == "rating":
                    # print("Processed rating:", processed)
                    rating_events.append(processed)

                    live.update(
                        Text(
                            f"Current Rows: {len(rating_events)}\nLatest Timestamp: {datetime.fromisoformat(rating_events[-1]['timestamp'])}"
                        )
                    )
    except KeyboardInterrupt:
        print("Interrupted. Saving rating events...")
    finally:
        consumer.close()

        rating_df = store_data(rating_events, "ratings.csv", sort="timestamp")
        print(f"{len(rating_events)} rating events saved to ratings.csv")

        userids = rating_df["userid"].unique()
        users = []
        print(f"Fetching details for {len(userids)} users...")
        for userid in track(
            userids, description="Fetching user details...", total=len(userids)
        ):
            user = get_user_details(userid)
            users.append(user)
        store_data(users, filename="users.csv")
        print(f"{len(users)} users saved to users.csv")

        movieids = rating_df["movieid"].unique()
        movies = []
        print(f"Fetching details for {len(movieids)} movies...")
        for movieid in track(
            movieids, description="Fetching movie details...", total=len(movieids)
        ):
            movie = get_movie_details(movieid)
            movies.append(movie)
        store_data_json(movies, filename="movies.json")
        print(f"{len(movies)} movies saved to movies.json")

        print("Exiting...")
