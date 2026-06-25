import os
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class DashboardLogger:
    @staticmethod
    def log_event(event_type, details):
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            file_path = os.path.join(LOG_DIR, f"log_{date_str}.jsonl")
            entry = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "event_type": event_type,
                "details": details
            }
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Logging error: {e}")

    @staticmethod
    def get_available_dates():
        dates = []
        if os.path.exists(LOG_DIR):
            for file in os.listdir(LOG_DIR):
                if file.startswith("log_") and file.endswith(".jsonl"):
                    date_str = file.replace("log_", "").replace(".jsonl", "")
                    dates.append(date_str)
        return sorted(dates, reverse=True)

    @staticmethod
    def get_logs_for_date(date_str):
        file_path = os.path.join(LOG_DIR, f"log_{date_str}.jsonl")
        logs = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            logs.append(json.loads(line))
                        except:
                            pass
        return logs
