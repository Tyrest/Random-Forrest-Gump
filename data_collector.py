#!/usr/bin/env python3 PID: 49138
import json, requests
from confluent_kafka import Consumer, KafkaError

consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "movie_data_collector",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)
topic = "movielog22"  # for team 22
consumer.subscribe([topic])


def parse_log_line(line):
    if "recommendation request" in line:
        return {"type": "recommendation", "raw": line}
    elif "GET /data/m/" in line:
        return {"type": "movie_play", "raw": line}
    elif "GET /rate/" in line:
        return {"type": "rating", "raw": line}
    else:
        return {"type": "unknown", "raw": line}


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
    parsed = parse_log_line(log_line)

    if parsed["type"] == "movie_play":
        try:
            parts = log_line.split(",")
            userid = parts[1].strip()
            request_part = parts[2].strip()
            path = request_part.split(" ")[1]
            path_parts = path.split("/")
            movieid = path_parts[3]
            movie_details = get_movie_details(movieid)
            parsed["movie_details"] = movie_details
            parsed["userid"] = userid
        except Exception as e:
            print("Error parsing movie play log:", e)

    elif parsed["type"] == "rating":
        try:
            parts = log_line.split(",")
            userid = parts[1].strip()
            request_part = parts[2].strip()
            rate_data = request_part.replace("GET /rate/", "")
            movieid, rating = rate_data.split("=")
            user_details = get_user_details(userid)
            parsed["user_details"] = user_details
            parsed["movieid"] = movieid
            parsed["rating"] = rating
        except Exception as e:
            print("Error parsing rating log:", e)
    return parsed


def store_data(data, filename="processed_logs.json"):
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    print("Starting Kafka consumer for topic:", topic)
    try:
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
            processed = process_log_line(log_line)
            print("Processed log:", processed)
            store_data(processed)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
