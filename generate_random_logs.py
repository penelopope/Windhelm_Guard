#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

EVENT_TYPES = ["MESSAGE_SENT", "MESSAGE_EVALUATION", "STYLE_ANALYSIS"]
CHANNELS = ["general", "random", "announcements", "bot-commands"]

def random_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def random_message():
    sentences = [
        "Hello world!",
        "What a beautiful day",
        "Anyone up for a game?",
        "Check out this cool link",
        "Did you see the news?",
        "I love coding in Python",
        "Random thought of the day",
        "Who wants pizza?",
        "Just finished a marathon",
        "Feeling lucky today"
    ]
    return random.choice(sentences)

def random_evaluation():
    return {
        "channel_info": random.choice(CHANNELS),
        "message_content": random_message(),
        "reasoning": "Just a random evaluation",
        "should_reply": random.choice([True, False]),
        "reply_content": random_message()
    }

def random_style_analysis():
    return {
        "channel_name": random.choice(CHANNELS),
        "profile": {"tone": "friendly", "style": "casual"}
    }

def generate_logs_for_date(date_str, count=10):
    file_path = os.path.join(LOG_DIR, f"log_{date_str}.jsonl")
    with open(file_path, "a", encoding="utf-8") as f:
        for _ in range(count):
            event_type = random.choice(EVENT_TYPES)
            if event_type == "MESSAGE_SENT":
                details = {
                    "channel_info": random.choice(CHANNELS),
                    "content": random_message()
                }
            elif event_type == "MESSAGE_EVALUATION":
                details = random_evaluation()
            else:
                details = random_style_analysis()
            entry = {
                "timestamp": random_timestamp(),
                "event_type": event_type,
                "details": details
            }
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    # generate logs for past 5 days
    for i in range(5):
        dt = datetime.now() - timedelta(days=i)
        date_str = dt.strftime('%Y-%m-%d')
        generate_logs_for_date(date_str, count=random.randint(5, 15))
    print("Random logs generated for the past 5 days.")
