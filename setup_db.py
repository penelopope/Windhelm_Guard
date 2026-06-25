#!/usr/bin/env python3
"""
Initialise the project's SQLite database and required JSON configuration files.

Running this script will:
- Create `members.db` with the necessary tables if it doesn't already exist.
- Generate `channel_styles.json`, `channel_conflict_state.json`,
  `friends.json`, and `random_comment_state.json` with sensible defaults.
"""

import os
import sqlite3
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "members.db")


def init_db() -> None:
    """Create the SQLite database and required tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            toxicity_score REAL DEFAULT 0.0,
            last_toxic_time REAL,
            last_warned_threshold INTEGER DEFAULT 0
        )
        """
    )

    # Offenses table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS offenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            toxicity_level INTEGER,
            reason TEXT,
            message TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    conn.commit()
    conn.close()
    print(f"[+] SQLite database initialised at: {DB_PATH}")


def ensure_json_files() -> None:
    """Create missing JSON config files with default content."""
    defaults = {
        "channel_styles.json": {},
        "channel_conflict_state.json": {},
        "friends.json": [],
        "random_comment_state.json": {"last_comment_time": 0.0},
    }
    for filename, default in defaults.items():
        path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(default, f, indent=4)
                print(f"[+] Created missing {filename}")
            except Exception as e:
                print(f"[!] Error creating {filename}: {e}")


if __name__ == "__main__":
    init_db()
    ensure_json_files()
