#!/usr/bin/env python3
import os
import sys
import time
import json
import sqlite3
import asyncio
import random
import datetime
from groq import Groq
from aiohttp import web
from logger import DashboardLogger
# Graceful shutdown handling (Render sends SIGTERM on restart)
import signal

LOCK_PATH = os.path.join(SCRIPT_DIR, "bot.lock")
if os.path.exists(LOCK_PATH):
    print("[🚫] Another bot instance is already running – exiting.")
    sys.exit(0)
# Create lock file for this process
open(LOCK_PATH, "w").close()

def _handle_shutdown(signum, frame):
    print("[⚡] Received shutdown signal, cleaning up…")
    try:
        os.remove(LOCK_PATH)
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)  # also handle Ctrl‑C locally
try:
    import discord
except ImportError:
    print("Error: 'discord.py' package is not installed. Please install it using:", file=sys.stderr)
    print("pip3 install discord.py --break-system-packages", file=sys.stderr)
    sys.exit(1)

# Base Workspace Directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env():
    env_file = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            key, val = parts[0].strip(), parts[1].strip()
                            if val.startswith(('"', "'")) and val.endswith(val[0]):
                                val = val[1:-1]
                            os.environ[key] = val
            print("Loaded environment variables from .env")
        except Exception as e:
            print(f"Warning: Failed to load .env: {e}", file=sys.stderr)

load_env()

class LLMKey:
    def __init__(self, provider, key, model=None):
        self.provider = provider
        self.key = key
        self.model = model

class LLMLoadBalancer:
    def __init__(self):
        self.keys = []
        self.current_index = 0
        
        # Load Groq keys
        groq_str = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
        for k in [x.strip() for x in groq_str.split(",") if x.strip()]:
            self.keys.append(LLMKey("groq", k, "llama-3.1-8b-instant"))
            
        # Load Gemini keys
        gemini_str = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or ""
        for k in [x.strip() for x in gemini_str.split(",") if x.strip()]:
            self.keys.append(LLMKey("gemini", k, "gemini-1.5-flash"))
            
        # Load Deepseek keys
        deepseek_str = os.environ.get("DEEPSEEK_API_KEYS") or os.environ.get("DEEPSEEK_API_KEY") or ""
        for k in [x.strip() for x in deepseek_str.split(",") if x.strip()]:
            self.keys.append(LLMKey("deepseek", k, "deepseek-chat"))
            
        # Load OpenAI keys
        openai_str = os.environ.get("OPENAI_API_KEYS") or os.environ.get("OPENAI_API_KEY") or ""
        for k in [x.strip() for x in openai_str.split(",") if x.strip()]:
            self.keys.append(LLMKey("openai", k, "gpt-4o-mini"))
            
        print(f"LLM Load Balancer initialized with {len(self.keys)} total keys.")

    def get_next_key(self):
        if not self.keys:
            return None
        key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key

    async def generate_chat_completion(self, system_prompt, user_prompt, response_json=False):
        if not self.keys:
            raise ValueError("No API keys configured for LLM Load Balancer.")
            
        import aiohttp
        
        attempts = len(self.keys)
        for attempt in range(attempts):
            llm_key = self.get_next_key()
            if not llm_key:
                continue
                
            provider = llm_key.provider
            key = llm_key.key
            model = llm_key.model
            
            print(f"[LLM Load Balancer] Attempting call with {provider} (model: {model})...")
            
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    if provider == "groq":
                        url = "https://api.groq.com/openai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": 0.3
                        }
                        if response_json:
                            payload["response_format"] = {"type": "json_object"}
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return result["choices"][0]["message"]["content"].strip()
                            else:
                                err_text = await resp.text()
                                print(f"[LLM Load Balancer] Groq API returned status {resp.status}: {err_text}", file=sys.stderr)
                                
                    elif provider == "openai":
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": 0.3
                        }
                        if response_json:
                            payload["response_format"] = {"type": "json_object"}
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return result["choices"][0]["message"]["content"].strip()
                            else:
                                err_text = await resp.text()
                                print(f"[LLM Load Balancer] OpenAI API returned status {resp.status}: {err_text}", file=sys.stderr)
                                
                    elif provider == "deepseek":
                        url = "https://api.deepseek.com/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": 0.3
                        }
                        if response_json:
                            payload["response_format"] = {"type": "json_object"}
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return result["choices"][0]["message"]["content"].strip()
                            else:
                                err_text = await resp.text()
                                print(f"[LLM Load Balancer] Deepseek API returned status {resp.status}: {err_text}", file=sys.stderr)
                                
                    elif provider == "gemini":
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                        headers = {
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "systemInstruction": {
                                "parts": [{"text": system_prompt}]
                            },
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": user_prompt}]
                                }
                            ],
                            "generationConfig": {
                                "temperature": 0.3
                            }
                        }
                        if response_json:
                            payload["generationConfig"]["responseMimeType"] = "application/json"
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                            else:
                                err_text = await resp.text()
                                print(f"[LLM Load Balancer] Gemini API returned status {resp.status}: {err_text}", file=sys.stderr)
                                
            except Exception as e:
                print(f"[LLM Load Balancer] Error calling {provider}: {e}", file=sys.stderr)
                
            print(f"[LLM Load Balancer] Provider {provider} failed. Trying next key/provider...", file=sys.stderr)
            
        raise RuntimeError("All LLM providers and keys in load balancer failed or rate-limited.")

    async def generate_vision_completion(self, url, prompt="Describe this image concisely and transcribe any prominent text.", response_json=False):
        groq_keys = [k for k in self.keys if k.provider == "groq"]
        if not groq_keys:
            print("[LLM Load Balancer] No Groq key configured for vision model.", file=sys.stderr)
            return None
            
        import aiohttp
        
        for lkey in groq_keys:
            key = lkey.key
            try:
                timeout = aiohttp.ClientTimeout(total=25)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    api_url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": url}}
                                ]
                            }
                        ],
                        "max_tokens": 300
                    }
                    if response_json:
                        payload["response_format"] = {"type": "json_object"}
                    async with session.post(api_url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result["choices"][0]["message"]["content"].strip()
                        else:
                            err_text = await resp.text()
                            print(f"[LLM Load Balancer] Groq Vision API returned status {resp.status}: {err_text}", file=sys.stderr)
            except Exception as e:
                print(f"[LLM Load Balancer] Error calling Groq Vision: {e}", file=sys.stderr)
        return None

llm_balancer = LLMLoadBalancer()
if not llm_balancer.keys:
    print("Error: No API keys configured in .env for Groq, Gemini, Deepseek, or OpenAI.", file=sys.stderr)
    sys.exit(1)

ALLOWED_CHANNELS = os.environ.get("ALLOWED_CHANNELS", "")
if ALLOWED_CHANNELS:
    ALLOWED_CHANNELS = [int(x.strip()) for x in ALLOWED_CHANNELS.split(",") if x.strip()]
else:
    ALLOWED_CHANNELS = []

NSFW_ALLOWED_CHANNELS = os.environ.get("NSFW_ALLOWED_CHANNELS", "")
if NSFW_ALLOWED_CHANNELS:
    NSFW_ALLOWED_CHANNELS = [int(x.strip()) for x in NSFW_ALLOWED_CHANNELS.split(",") if x.strip()]
else:
    NSFW_ALLOWED_CHANNELS = []

ALLOWED_GUILDS = os.environ.get("ALLOWED_GUILDS", "")
if ALLOWED_GUILDS:
    ALLOWED_GUILDS = [int(x.strip()) for x in ALLOWED_GUILDS.split(",") if x.strip()]
else:
    ALLOWED_GUILDS = []

OWNER_ID = os.environ.get("OWNER_ID", "").strip()

def get_persona():
    persona_path = os.path.join(SCRIPT_DIR, "persona.txt")
    if os.path.exists(persona_path):
        try:
            with open(persona_path, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            print(f"Error reading persona.txt: {e}", file=sys.stderr)
    return (
        "You are a City Guard from Windhelm in The Elder Scrolls V: Skyrim. "
        "You are ever-vigilant, rugged, and speak in a classic Nord guard mannerism."
    )

def get_lore():
    lore_path = os.path.join(SCRIPT_DIR, "lore.txt")
    if os.path.exists(lore_path):
        try:
            with open(lore_path, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            print(f"Error reading lore.txt: {e}", file=sys.stderr)
    return "No lore provided."

def get_convo_data():
    convo_path = os.path.join(SCRIPT_DIR, "convo_data.txt")
    if os.path.exists(convo_path):
        try:
            with open(convo_path, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            print(f"Error reading convo_data.txt: {e}", file=sys.stderr)
    return "No text examples provided."

def get_rules():
    rules_path = os.path.join(SCRIPT_DIR, "rules.txt")
    if os.path.exists(rules_path):
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            print(f"Error reading rules.txt: {e}", file=sys.stderr)
    return "No explicit community rules defined."

async def start_dashboard():
    async def dashboard_handler(request):
        dashboard_path = os.path.join(SCRIPT_DIR, "dashboard.html")
        return web.FileResponse(dashboard_path)

    async def api_dates(request):
        return web.json_response(DashboardLogger.get_available_dates())

    async def api_logs(request):
        date = request.query.get("date", "")
        return web.json_response(DashboardLogger.get_logs_for_date(date))

    app = web.Application()
    app.router.add_get('/', dashboard_handler)
    app.router.add_get('/api/dates', api_dates)
    app.router.add_get('/api/logs', api_logs)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("Dashboard Web Server running on http://localhost:8080")

class WindhelmGuardDiscordClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cooldowns = {}
        self.last_global_reply = 0
        self.ready_to_chat = False
        self.dashboard_started = False
        self.invites = {}
        self.init_db()
        # Ensure essential JSON files exist with default structures
        self.ensure_json_files()


    def init_db(self):
        db_path = os.path.join(SCRIPT_DIR, "members.db")
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            # Create users table
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    toxicity_score REAL DEFAULT 0.0,
                    last_toxic_time REAL,
                    last_warned_threshold INTEGER DEFAULT 0
                )
            """)
            # Create offenses table
            c.execute("""
                CREATE TABLE IF NOT EXISTS offenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp TEXT,
                    toxicity_level INTEGER,
                    reason TEXT,
                    message TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            conn.commit()
            # Ensure auxiliary JSON files exist
            self._ensure_file("channel_styles.json", default={})
            self._ensure_file("channel_conflict_state.json", default={})
            self._ensure_file("friends.json", default=[])
            self._ensure_file("random_comment_state.json", default={"last_comment_time": 0.0})
            
            # Check if we should migrate from critical_members.json
            c.execute("SELECT COUNT(*) FROM users")
            count = c.fetchone()[0]
            if count == 0:
                json_path = os.path.join(SCRIPT_DIR, "critical_members.json")
                if os.path.exists(json_path):
                    print("Migrating critical_members.json to SQLite database...")
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        for uid, udata in data.items():
                            name = udata.get("name", "Unknown")
                            score = float(udata.get("toxicity_score", 0.0))
                            last_toxic = udata.get("last_toxic_time")
                            if last_toxic is not None:
                                last_toxic = float(last_toxic)
                            warned = int(udata.get("last_warned_threshold", 0))
                            
                            c.execute("INSERT INTO users (user_id, name, toxicity_score, last_toxic_time, last_warned_threshold) VALUES (?, ?, ?, ?, ?)",
                                      (uid, name, score, last_toxic, warned))
                            
                            offenses = udata.get("offenses", [])
                            if isinstance(offenses, int):
                                offenses = []
                            for o in offenses:
                                ts = o.get("timestamp", "")
                                tox = int(o.get("toxicity_level", 0))
                                reason = o.get("reason", "")
                                msg = o.get("message", "")
                                c.execute("INSERT INTO offenses (user_id, timestamp, toxicity_level, reason, message) VALUES (?, ?, ?, ?, ?)",
                                          (uid, ts, tox, reason, msg))
                        conn.commit()
                        print("Migration completed successfully.")
                    except Exception as me:
                        print(f"Error during SQLite migration: {me}", file=sys.stderr)
            conn.close()
        except Exception as e:
            print(f"Error initializing SQLite database: {e}", file=sys.stderr)

    def load_channel_styles(self):
        styles_file = os.path.join(SCRIPT_DIR, "channel_styles.json")
        if os.path.exists(styles_file):
            try:
                with open(styles_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading channel_styles.json: {e}", file=sys.stderr)
        return {}

    def save_channel_styles(self, styles):
        styles_file = os.path.join(SCRIPT_DIR, "channel_styles.json")
        try:
            with open(styles_file, "w") as f:
                json.dump(styles, f, indent=4)
        except Exception as e:
            print(f"Error writing channel_styles.json: {e}", file=sys.stderr)

    def load_critical_members(self):
        db_path = os.path.join(SCRIPT_DIR, "members.db")
        data = {}
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            c.execute("SELECT user_id, name, toxicity_score, last_toxic_time, last_warned_threshold FROM users")
            users = c.fetchall()
            for u in users:
                uid, name, score, last_toxic, warned = u
                data[uid] = {
                    "name": name,
                    "toxicity_score": float(score) if score is not None else 0.0,
                    "last_toxic_time": float(last_toxic) if last_toxic is not None else None,
                    "last_warned_threshold": int(warned) if warned is not None else 0,
                    "offenses": [],
                    "total_offenses": 0
                }
                
            c.execute("SELECT user_id, timestamp, toxicity_level, reason, message FROM offenses ORDER BY id ASC")
            offenses = c.fetchall()
            for o in offenses:
                uid, ts, tox, reason, msg = o
                if uid in data:
                    data[uid]["offenses"].append({
                        "timestamp": ts,
                        "toxicity_level": int(tox),
                        "reason": reason,
                        "message": msg
                    })
                    
            for uid in data:
                data[uid]["total_offenses"] = len(data[uid]["offenses"])
                
            conn.close()
        except Exception as e:
            print(f"Error loading members from SQLite: {e}", file=sys.stderr)
        return data

    def save_critical_members(self, data):
        db_path = os.path.join(SCRIPT_DIR, "members.db")
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            for uid, udata in data.items():
                name = udata.get("name", "Unknown")
                score = float(udata.get("toxicity_score", 0.0))
                last_toxic = udata.get("last_toxic_time")
                if last_toxic is not None:
                    last_toxic = float(last_toxic)
                warned = int(udata.get("last_warned_threshold", 0))
                
                c.execute("""
                    INSERT INTO users (user_id, name, toxicity_score, last_toxic_time, last_warned_threshold)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        name=excluded.name,
                        toxicity_score=excluded.toxicity_score,
                        last_toxic_time=excluded.last_toxic_time,
                        last_warned_threshold=excluded.last_warned_threshold
                """, (uid, name, score, last_toxic, warned))
                
                c.execute("DELETE FROM offenses WHERE user_id = ?", (uid,))
                
                offenses = udata.get("offenses", [])
                for o in offenses:
                    ts = o.get("timestamp", "")
                    tox = int(o.get("toxicity_level", 0))
                    reason = o.get("reason", "")
                    msg = o.get("message", "")
                    c.execute("INSERT INTO offenses (user_id, timestamp, toxicity_level, reason, message) VALUES (?, ?, ?, ?, ?)",
                              (uid, ts, tox, reason, msg))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving members to SQLite: {e}", file=sys.stderr)

    def load_random_comment_state(self):
        state_file = os.path.join(SCRIPT_DIR, "random_comment_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading random_comment_state.json: {e}", file=sys.stderr)
        return {"last_comment_time": 0.0}

    def save_random_comment_state(self, state):
        state_file = os.path.join(SCRIPT_DIR, "random_comment_state.json")
        try:
            with open(state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error writing random_comment_state.json: {e}", file=sys.stderr)

    def _ensure_file(self, filename, default):
        """Create a JSON file with a default value if it does not exist."""
        path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(path):
            try:
                with open(path, "w") as f:
                    json.dump(default, f, indent=4)
                print(f"Created missing {filename} with default content.")
            except Exception as e:
                print(f"Error creating {filename}: {e}", file=sys.stderr)

    def ensure_json_files(self):
        """Public method to ensure all required JSON configuration files exist."""
        self._ensure_file("channel_styles.json", default={})
        self._ensure_file("channel_conflict_state.json", default={})
        self._ensure_file("friends.json", default=[])
        self._ensure_file("random_comment_state.json", default={"last_comment_time": 0.0})

    async def generate_fun_comment(self, channel, history_str, style_guidelines, lingo):
        prompt = f"""You are a City Guard from Skyrim, but you are also a casual participant in this Discord server.
The current conversation is casual and friendly.
You want to jump in with a rare, funny, and lighthearted comment.
Blend Skyrim guard lore with real-world/Indian culture or modern slang (e.g. "In Skyrim we have mead that we throw on each other in holi, bruh"). Keep it casual, matching the channel style if possible.

Channel style guidelines:
{style_guidelines}
{lingo}

Recent chat history:
{history_str}

Respond with a single line containing only your fun comment. Do not use quotes or prefixes.
"""
        try:
            content = await llm_balancer.generate_chat_completion(
                "You are a casual Skyrim Guard participant. Output only the message text.",
                prompt,
                response_json=False
            )
            return content.strip().strip('"')
        except Exception as e:
            print(f"Error generating fun comment: {e}", file=sys.stderr)
            return None

    async def resolve_member(self, guild, target_str):
        if not guild or not target_str:
            return None
        target_str = target_str.strip().lstrip("@")
        cleaned_search = target_str.replace("<@", "").replace(">", "").replace("!", "").strip()
        if cleaned_search.isdigit():
            try:
                return await guild.fetch_member(int(cleaned_search))
            except Exception:
                pass
        
        try:
            members = await guild.query_members(query=target_str, limit=5)
            if members:
                return members[0]
        except Exception:
            pass
            
        for member in guild.members:
            if (target_str.lower() in member.name.lower() or 
                (member.nick and target_str.lower() in member.nick.lower()) or
                target_str.lower() in member.display_name.lower()):
                return member
        return None

    async def execute_command(self, message, result):
        cmd = result.get("command_to_execute", "none")
        if not cmd or cmd == "none":
            return
        
        target = result.get("command_target")
        args = result.get("command_args") or {}
        
        author = message.author
        OWNER_ID = os.environ.get("OWNER_ID", "")
        is_owner = OWNER_ID and str(author.id) == OWNER_ID
        
        is_authorized = is_owner
        if not is_authorized and hasattr(author, "roles"):
            for r in author.roles:
                if r.id in [1399784920968073316, 1501626692203184238, 1519700860010102895]:
                    is_authorized = True
                    break
                    
        if not is_authorized:
            await message.reply("Wait... I know you. You do not have the authority to command the Jarl's guards. Move along, citizen.")
            return

        if cmd == "timeout":
            await self.cmd_timeout(message, target, args)
        elif cmd == "untimeout":
            await self.cmd_untimeout(message, target, args)
        elif cmd == "kick":
            await self.cmd_kick(message, target, args)
        elif cmd == "ban":
            await self.cmd_ban(message, target, args)
        elif cmd == "unban":
            await self.cmd_unban(message, target, args)
        elif cmd == "clear":
            await self.cmd_clear(message, target, args)
        elif cmd == "lock":
            await self.cmd_lock(message, target, args)
        elif cmd == "unlock":
            await self.cmd_unlock(message, target, args)
        elif cmd == "slowmode":
            await self.cmd_slowmode(message, target, args)
        elif cmd == "warn":
            await self.cmd_warn(message, target, args)
        elif cmd == "reset_toxicity":
            await self.cmd_reset_toxicity(message, target, args)

    async def cmd_timeout(self, message, target, args):
        guild = message.guild
        if not guild:
            await message.reply("This command can only be run in a server.")
            return
        if not target:
            await message.reply("Who is it you wish to timeout? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        if not member:
            await message.reply(f"I could not find the citizen '{target}' in the registry.")
            return
        
        duration_mins = args.get("duration_mins")
        try:
            duration_mins = int(duration_mins)
        except (ValueError, TypeError):
            duration_mins = 10
            
        reason = args.get("reason") or "Lollygagging"
        try:
            duration = datetime.timedelta(minutes=duration_mins)
            await member.timeout(duration, reason=reason)
            await message.channel.send(
                f"By order of the Jarl, {member.mention} has been cast into the dungeon (timed out) for {duration_mins} minutes. Reason: {reason}."
            )
        except Exception as e:
            await message.channel.send(f"I tried to lock up {member.display_name}, but the lock is rusted (error: {e}).")

    async def cmd_untimeout(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to pardon? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        if not member:
            await message.reply(f"I could not find the citizen '{target}' in the registry.")
            return
        reason = args.get("reason") or "Pardoned by the Jarl"
        try:
            await member.timeout(None, reason=reason)
            await message.channel.send(f"The Jarl has pardoned {member.mention}. Their timeout is removed.")
        except Exception as e:
            await message.channel.send(f"Failed to remove timeout from {member.display_name}: {e}")

    async def cmd_kick(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to kick? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        if not member:
            await message.reply(f"I could not find the citizen '{target}' in the registry.")
            return
        reason = args.get("reason") or "Exiled from Windhelm"
        try:
            await member.kick(reason=reason)
            await message.channel.send(f"By order of the Jarl, {member.display_name} has been kicked from the server. Reason: {reason}.")
        except Exception as e:
            await message.channel.send(f"Failed to kick {member.display_name}: {e}")

    async def cmd_ban(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to ban? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        user_to_ban = member
        if not user_to_ban:
            cleaned_search = target.replace("<@", "").replace(">", "").replace("!", "").strip()
            if cleaned_search.isdigit():
                try:
                    user_to_ban = await self.fetch_user(int(cleaned_search))
                except Exception:
                    pass
        if not user_to_ban:
            await message.reply(f"I could not resolve '{target}' to a user.")
            return
        reason = args.get("reason") or "High treason against the Jarl"
        try:
            await guild.ban(user_to_ban, reason=reason, delete_message_days=0)
            await message.channel.send(f"🔨 {user_to_ban.name} has been permanently banned from the server. Reason: {reason}.")
        except Exception as e:
            await message.channel.send(f"Failed to ban {user_to_ban.name}: {e}")

    async def cmd_unban(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to unban? State their name clearly.")
            return
        
        banned_user = None
        cleaned_search = target.replace("<@", "").replace(">", "").replace("!", "").strip()
        
        try:
            async for ban_entry in guild.bans():
                user = ban_entry.user
                if cleaned_search.isdigit() and user.id == int(cleaned_search):
                    banned_user = user
                    break
                if target.lower() in user.name.lower():
                    banned_user = user
                    break
        except Exception:
            pass
            
        if not banned_user:
            if cleaned_search.isdigit():
                try:
                    banned_user = await self.fetch_user(int(cleaned_search))
                except Exception:
                    pass
                    
        if not banned_user:
            await message.reply(f"I could not find a banned citizen matching '{target}'.")
            return
            
        reason = args.get("reason") or "Unbanned by request"
        try:
            await guild.unban(banned_user, reason=reason)
            await message.channel.send(f"🔓 {banned_user.name} has been unbanned. Reason: {reason}.")
        except Exception as e:
            await message.channel.send(f"Failed to unban {banned_user.name}: {e}")

    async def cmd_clear(self, message, target, args):
        channel = message.channel
        if target and target != "current":
            cleaned = target.replace("<#", "").replace(">", "").strip()
            if cleaned.isdigit():
                chan = self.get_channel(int(cleaned))
                if chan:
                    channel = chan
                    
        limit = args.get("clear_limit")
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 10
            
        try:
            deleted = await channel.purge(limit=limit + 1 if channel == message.channel else limit)
            deleted_count = len(deleted) - (1 if channel == message.channel else 0)
            await message.channel.send(f"🗑️ Swept {deleted_count} messages from {channel.mention}.", delete_after=5)
        except Exception as e:
            await message.reply(f"Failed to clear messages: {e}")

    async def cmd_lock(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        channel = message.channel
        if target and target != "current":
            cleaned = target.replace("<#", "").replace(">", "").strip()
            if cleaned.isdigit():
                chan = self.get_channel(int(cleaned))
                if chan:
                    channel = chan
                    
        reason = args.get("reason") or "Under guard lockdown"
        try:
            await channel.set_permissions(guild.default_role, send_messages=False, reason=reason)
            await channel.send(f"🔒 **This channel is now locked by order of the Jarl.** Reason: {reason}")
        except Exception as e:
            await message.reply(f"Failed to lock channel: {e}")

    async def cmd_unlock(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        channel = message.channel
        if target and target != "current":
            cleaned = target.replace("<#", "").replace(">", "").strip()
            if cleaned.isdigit():
                chan = self.get_channel(int(cleaned))
                if chan:
                    channel = chan
                    
        reason = args.get("reason") or "Lockdown lifted"
        try:
            await channel.set_permissions(guild.default_role, send_messages=None, reason=reason)
            await channel.send(f"🔓 **Lockdown lifted. Normal conversation may resume.**")
        except Exception as e:
            await message.reply(f"Failed to unlock channel: {e}")

    async def cmd_slowmode(self, message, target, args):
        channel = message.channel
        if target and target != "current":
            cleaned = target.replace("<#", "").replace(">", "").strip()
            if cleaned.isdigit():
                chan = self.get_channel(int(cleaned))
                if chan:
                    channel = chan
                    
        seconds = args.get("slowmode_seconds")
        try:
            seconds = int(seconds)
        except (ValueError, TypeError):
            seconds = 0
            
        try:
            await channel.edit(slowmode_delay=seconds)
            if seconds > 0:
                await channel.send(f"⏳ Slowmode has been set to {seconds} seconds.")
            else:
                await channel.send(f"⏳ Slowmode has been disabled.")
        except Exception as e:
            await message.reply(f"Failed to set slowmode: {e}")

    async def cmd_warn(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to warn? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        if not member:
            await message.reply(f"I could not find the citizen '{target}' in the registry.")
            return
            
        reason = args.get("reason") or "Disorderly conduct"
        
        members = self.load_critical_members()
        uid = str(member.id)
        fd = members.get(uid, {
            "name": member.display_name,
            "toxicity_score": 0.0,
            "last_toxic_time": None,
            "last_warned_threshold": 0,
            "offenses": [],
            "total_offenses": 0
        })
        if "offenses" not in fd or not isinstance(fd["offenses"], list):
            fd["offenses"] = []
        if "toxicity_score" not in fd:
            fd["toxicity_score"] = 0.0
            
        fd["toxicity_score"] = float(fd["toxicity_score"]) + 10.0
        fd["last_toxic_time"] = time.time()
        
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        fd["offenses"].append({
            "timestamp": timestamp_str,
            "toxicity_level": 50,
            "reason": f"[Jarl Warn] {reason}",
            "message": message.clean_content
        })
        fd["total_offenses"] = len(fd["offenses"])
        
        members[uid] = fd
        self.save_critical_members(members)
        
        await message.channel.send(f"⚠️ {member.mention} has been issued a warning. Reason: {reason}.")

    async def cmd_reset_toxicity(self, message, target, args):
        guild = message.guild
        if not guild:
            return
        if not target:
            await message.reply("Who is it you wish to pardon/reset? State their name clearly.")
            return
        member = await self.resolve_member(guild, target)
        if not member:
            await message.reply(f"I could not find the citizen '{target}' in the registry.")
            return
            
        members = self.load_critical_members()
        uid = str(member.id)
        if uid in members:
            fd = members[uid]
            fd["toxicity_score"] = 0.0
            fd["last_toxic_time"] = None
            fd["last_warned_threshold"] = 0
            fd["offenses"] = []
            fd["total_offenses"] = 0
            members[uid] = fd
            self.save_critical_members(members)
            await message.channel.send(f"🕊️ The registry for {member.mention} has been cleared. Their toxicity score is reset to 0.0 and past offenses have been pardoned.")
        else:
            await message.channel.send(f"🕊️ {member.mention} has no record of offenses in the Jarl's registry. They are already clean.")

    async def handle_reminder(self, message, result):
        delay = result.get("reminder_delay_seconds")
        remind_content = result.get("reminder_text") or "something"
        
        try:
            delay_secs = int(delay)
        except (ValueError, TypeError):
            return # invalid delay

        if delay_secs <= 0:
            return

        author_mention = message.author.mention
        channel = message.channel

        async def run_reminder(secs, content, target_channel, mention_str):
            await asyncio.sleep(secs)
            try:
                await target_channel.send(f"⏰ **Reminder for {mention_str}**: {content}")
            except Exception as e:
                print(f"Failed to send reminder: {e}")

        asyncio.create_task(run_reminder(delay_secs, remind_content, channel, author_mention))

    def load_conflict_state(self):
        state_file = os.path.join(SCRIPT_DIR, "channel_conflict_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading channel_conflict_state.json: {e}", file=sys.stderr)
        return {}

    def save_conflict_state(self, state):
        state_file = os.path.join(SCRIPT_DIR, "channel_conflict_state.json")
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"Error writing channel_conflict_state.json: {e}", file=sys.stderr)

    async def handle_profile_request(self, message, result):
        target_str = result.get("profile_target_user")
        if not target_str:
            await message.reply("Who is it you wish to look up? State their name clearly.")
            return
        target_str = target_str.strip().lstrip("@")

        # Check authorization of requestor
        author = message.author
        OWNER_ID = os.environ.get("OWNER_ID", "")
        is_owner = OWNER_ID and str(author.id) == OWNER_ID
        is_authorized = is_owner
        if not is_authorized and hasattr(author, "roles"):
            for r in author.roles:
                if r.id in [1399784920968073316, 1501626692203184238, 1519700860010102895]:
                    is_authorized = True
                    break

        if not is_authorized:
            await message.reply("Wait... I know you. You do not have permission to view the Jarl's registry.")
            return

        # Locate target user ID
        target_uid = None
        target_name = target_str

        # Try to resolve in guild first
        guild = message.guild
        if guild:
            target_member = None
            for m in message.mentions:
                if m.id != self.user.id:
                    target_member = m
                    break
            
            if not target_member:
                cleaned_search = target_str.replace("<@", "").replace(">", "").replace("!", "").strip()
                if cleaned_search.isdigit():
                    try:
                        target_member = await guild.fetch_member(int(cleaned_search))
                    except Exception:
                        pass
                if not target_member:
                    try:
                        members = await guild.query_members(query=target_str, limit=5)
                        if members:
                            target_member = members[0]
                    except Exception:
                        pass
                if not target_member:
                    for member in guild.members:
                        if (target_str.lower() in member.name.lower() or 
                            (member.nick and target_str.lower() in member.nick.lower()) or
                            target_str.lower() in member.display_name.lower()):
                            target_member = member
                            break
            if target_member:
                target_uid = str(target_member.id)
                target_name = target_member.display_name

        # Load records
        members_data = self.load_critical_members()
        profile_data = None
        
        # If resolved UID, look up in JSON
        if target_uid and target_uid in members_data:
            profile_data = members_data[target_uid]
        else:
            # Fallback search by name inside JSON keys
            for uid, user_data in members_data.items():
                if target_str.lower() in user_data.get("name", "").lower():
                    profile_data = user_data
                    target_uid = uid
                    target_name = user_data["name"]
                    break

        # Format profile Embed
        embed = discord.Embed(
            title=f"📜 Registry Profile: {target_name}",
            color=discord.Color.dark_purple(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        if target_uid:
            embed.add_field(name="User ID", value=target_uid, inline=True)

        if not profile_data or not profile_data.get("offenses"):
            embed.description = "This citizen has no offenses recorded. Their record is clean, like a fresh snowfall in Windhelm."
            embed.add_field(name="Cumulative Toxicity Score", value="0", inline=True)
            embed.add_field(name="Offenses Recorded", value="0", inline=True)
        else:
            embed.add_field(name="Cumulative Toxicity Score", value=str(profile_data.get("toxicity_score", 0)), inline=True)
            embed.add_field(name="Offenses Recorded", value=str(profile_data.get("total_offenses", 0)), inline=True)
            
            # List past offenses (limit to last 5 to keep embed clean)
            offense_list = ""
            for i, off in enumerate(profile_data["offenses"][-5:]):
                offense_list += (
                    f"**{i+1}. [{off['timestamp']}]** Toxicity: {off['toxicity_level']}/100\n"
                    f"└ Reason: *{off['reason']}*\n"
                    f"└ Msg: \"{off['message'][:80]}\"\n\n"
                )
            embed.add_field(name="Recent Recorded Offenses (Last 5)", value=offense_list, inline=False)

        # Decide where to send
        MOD_CHANNEL_ID = os.environ.get("MOD_CHANNEL_ID", "")
        is_mod_channel_or_dm = (MOD_CHANNEL_ID and str(message.channel.id) == MOD_CHANNEL_ID) or isinstance(message.channel, discord.DMChannel)

        if is_mod_channel_or_dm:
            await message.reply(embed=embed)
        else:
            await message.reply("I have checked the registry for that citizen and delivered the records securely to the Jarl's steward.")
            if MOD_CHANNEL_ID:
                mod_channel = self.get_channel(int(MOD_CHANNEL_ID))
                if mod_channel:
                    await mod_channel.send(embed=embed)

    async def log_to_mod_channel(self, embed):
        MOD_CHANNEL_ID = os.environ.get("MOD_CHANNEL_ID", "")
        if not MOD_CHANNEL_ID:
            return
        channel = self.get_channel(int(MOD_CHANNEL_ID))
        if channel:
            try:
                await channel.send(embed=embed)
            except Exception as e:
                print(f"Failed to send log embed to MOD_CHANNEL_ID: {e}", file=sys.stderr)

    async def log_to_server_log_channel(self, embed):
        LOG_CHANNEL_ID = os.environ.get("LOG_CHANNEL_ID", "")
        if not LOG_CHANNEL_ID:
            LOG_CHANNEL_ID = os.environ.get("MOD_CHANNEL_ID", "")
        if not LOG_CHANNEL_ID:
            return
        channel = self.get_channel(int(LOG_CHANNEL_ID))
        if channel:
            try:
                await channel.send(embed=embed)
            except Exception as e:
                print(f"Failed to send log embed to LOG_CHANNEL_ID: {e}", file=sys.stderr)

    async def send_threshold_mod_summary(self, member_profile, user_id, threshold, trigger_msg_url=None):
        MOD_CHANNEL_ID = os.environ.get("MOD_CHANNEL_ID", "")
        if not MOD_CHANNEL_ID:
            return
        channel = self.get_channel(int(MOD_CHANNEL_ID))
        if not channel:
            return
        
        offenses = member_profile.get("offenses", [])
        offense_summary = ""
        if offenses:
            for i, off in enumerate(offenses[-5:]):
                offense_summary += f"- **[{off.get('timestamp')}]** (Tox: {off.get('toxicity_level')}/100): {off.get('reason')}\n  *Msg: \"{off.get('message')}\"*\n"
        else:
            offense_summary = "No detailed offenses logged."

        embed = discord.Embed(
            title=f"🚨 Toxicity Threshold Crossed ({threshold}/100) 🚨",
            description=f"Citizen **{member_profile.get('name')}** (ID: `{user_id}`) has crossed the {threshold} toxicity threshold.",
            color=discord.Color.red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="Current Toxicity Score", value=f"{member_profile.get('toxicity_score'):.1f}", inline=True)
        embed.add_field(name="Total Offenses", value=str(member_profile.get("total_offenses", len(offenses))), inline=True)
        embed.add_field(name="Recent Offenses (Last 5)", value=offense_summary[:1024], inline=False)
        if trigger_msg_url:
            embed.add_field(name="Trigger Message Link", value=f"[Jump to Message]({trigger_msg_url})", inline=False)
        
        try:
            await channel.send(embed=embed)
        except Exception as e:
            print(f"Error sending threshold mod summary: {e}", file=sys.stderr)

    async def on_message_delete(self, message):
        if message.author.bot:
            return
        if ALLOWED_GUILDS and message.guild and message.guild.id not in ALLOWED_GUILDS:
            return
        embed = discord.Embed(
            title="🗑️ Message Deleted",
            color=discord.Color.red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="Author", value=f"{message.author.mention} ({message.author.id})", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention if hasattr(message.channel, "mention") else str(message.channel), inline=True)
        embed.add_field(name="Content", value=message.content[:1024] or "[No text content]", inline=False)
        await self.log_to_server_log_channel(embed)

    # async def on_message_edit(self, before, after):
    #     if before.author.bot:
    #         return
    #     if before.content == after.content:
    #         return
    #     if ALLOWED_GUILDS and before.guild and before.guild.id not in ALLOWED_GUILDS:
    #         return
    #     embed = discord.Embed(
    #         title="✏️ Message Edited",
    #         color=discord.Color.orange(),
    #         timestamp=datetime.datetime.now(datetime.timezone.utc)
    #     )
    #     embed.add_field(name="Author", value=f"{before.author.mention} ({before.author.id})", inline=True)
    #     embed.add_field(name="Channel", value=before.channel.mention if hasattr(before.channel, "mention") else str(before.channel), inline=True)
    #     embed.add_field(name="Before", value=before.content[:1024] or "[No text content]", inline=False)
    #     embed.add_field(name="After", value=after.content[:1024] or "[No text content]", inline=False)
    #     embed.add_field(name="Message Link", value=f"[Jump to Message]({after.jump_url})", inline=False)
    #     await self.log_to_server_log_channel(embed)

    async def on_voice_state_update(self, member, before, after):
        if ALLOWED_GUILDS and member.guild and member.guild.id not in ALLOWED_GUILDS:
            return
        if member.bot:
            return
        if before.channel == after.channel:
            return
        
        embed = discord.Embed(
            color=discord.Color.blue(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="User", value=f"{member.mention} ({member.id})", inline=True)
        
        if before.channel is None and after.channel is not None:
            embed.title = "🔊 Voice Joined"
            embed.add_field(name="Channel", value=after.channel.name, inline=True)
        elif before.channel is not None and after.channel is None:
            embed.title = "🔇 Voice Left"
            embed.add_field(name="Channel", value=before.channel.name, inline=True)
        elif before.channel is not None and after.channel is not None:
            embed.title = "🔁 Voice Moved"
            embed.add_field(name="Old Channel", value=before.channel.name, inline=True)
            embed.add_field(name="New Channel", value=after.channel.name, inline=True)
        
        await self.log_to_server_log_channel(embed)

    async def on_user_update(self, before, after):
        if after.bot:
            return
        embed = discord.Embed(
            title="👤 User Profile Updated",
            color=discord.Color.teal(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="User", value=f"{after.mention} ({after.id})", inline=False)
        
        changed = False
        if before.name != after.name:
            embed.add_field(name="Old Username", value=before.name, inline=True)
            embed.add_field(name="New Username", value=after.name, inline=True)
            changed = True
        if before.avatar != after.avatar:
            embed.add_field(name="Avatar Changed", value="User updated their profile picture.", inline=False)
            changed = True
            
        if changed:
            await self.log_to_server_log_channel(embed)

    async def on_member_update(self, before, after):
        if ALLOWED_GUILDS and before.guild and before.guild.id not in ALLOWED_GUILDS:
            return
        if after.bot:
            return
        embed = discord.Embed(
            title="👤 Member Updated",
            color=discord.Color.teal(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="Member", value=f"{after.mention} ({after.id})", inline=False)
        
        changed = False
        if before.nick != after.nick:
            embed.add_field(name="Old Nickname", value=str(before.nick), inline=True)
            embed.add_field(name="New Nickname", value=str(after.nick), inline=True)
            changed = True
            
        if before.roles != after.roles:
            old_role_names = [r.name for r in before.roles]
            new_role_names = [r.name for r in after.roles]
            added_roles = list(set(new_role_names) - set(old_role_names))
            removed_roles = list(set(old_role_names) - set(new_role_names))
            
            if added_roles:
                embed.add_field(name="Roles Added", value=", ".join(added_roles), inline=False)
                changed = True
            if removed_roles:
                embed.add_field(name="Roles Removed", value=", ".join(removed_roles), inline=False)
                changed = True

        if getattr(before, "communication_disabled_until", None) != getattr(after, "communication_disabled_until", None):
            if after.communication_disabled_until:
                duration_secs = (after.communication_disabled_until - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                duration_mins = max(1, int(duration_secs // 60))
                embed.add_field(name="Action", value="🔇 Timed Out (Muted)", inline=False)
                embed.add_field(name="Duration", value=f"{duration_mins} minutes", inline=True)
                
                moderator = "Unknown"
                reason = "No reason provided"
                try:
                    async for entry in before.guild.audit_logs(limit=5, action=discord.AuditLogAction.member_update):
                        if entry.target.id == after.id and getattr(entry.after, "communication_disabled_until", None) is not None:
                            moderator = f"{entry.user.mention} ({entry.user.name} / ID: {entry.user.id})"
                            if entry.reason:
                                reason = entry.reason
                            break
                except Exception:
                    pass
                embed.add_field(name="Moderator", value=moderator, inline=True)
                embed.add_field(name="Reason", value=reason, inline=True)
            else:
                embed.add_field(name="Action", value="🔊 Timeout Removed", inline=False)
                moderator = "Unknown"
                try:
                    async for entry in before.guild.audit_logs(limit=5, action=discord.AuditLogAction.member_update):
                        if entry.target.id == after.id and getattr(entry.before, "communication_disabled_until", None) is not None and getattr(entry.after, "communication_disabled_until", None) is None:
                            moderator = f"{entry.user.mention} ({entry.user.name} / ID: {entry.user.id})"
                            break
                except Exception:
                    pass
                embed.add_field(name="Moderator", value=moderator, inline=True)
            changed = True

        if changed:
            await self.log_to_server_log_channel(embed)

    async def track_invite_used(self, member):
        guild = member.guild
        if guild.id not in self.invites:
            return None, None

        old_invs = self.invites[guild.id]
        try:
            new_invs = await guild.invites()
        except Exception as e:
            print(f"Failed to fetch invites on join for {guild.name}: {e}")
            return None, None

        self.invites[guild.id] = new_invs

        for old_inv in old_invs:
            for new_inv in new_invs:
                if old_inv.code == new_inv.code:
                    if new_inv.uses > old_inv.uses:
                        return new_inv.inviter, new_inv

        for new_inv in new_invs:
            if new_inv.code not in [x.code for x in old_invs]:
                if new_inv.uses > 0:
                    return new_inv.inviter, new_inv

        return None, None

    async def on_member_join(self, member):
        if ALLOWED_GUILDS and member.guild.id not in ALLOWED_GUILDS:
            return
        if member.bot:
            return
        
        inviter, invite = await self.track_invite_used(member)
        
        embed = discord.Embed(
            title="📥 Citizen Entered the Hold (Member Joined)",
            color=discord.Color.green(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.set_thumbnail(url=member.display_avatar.url if member.avatar else None)
        embed.add_field(name="User", value=f"{member.mention} ({member.name})", inline=True)
        embed.add_field(name="User ID", value=member.id, inline=True)
        
        created_days = (datetime.datetime.now(datetime.timezone.utc) - member.created_at).days
        embed.add_field(name="Account Age", value=f"{created_days} days ago (Created: {member.created_at.strftime('%Y-%m-%d')})", inline=False)
        
        if inviter:
            embed.add_field(name="Invited By", value=f"{inviter.mention} ({inviter.name})", inline=True)
            embed.add_field(name="Invite Code", value=f"`{invite.code}` (Uses: {invite.uses})", inline=True)
        else:
            embed.add_field(name="Invited By", value="Unknown (Vanished into the wind / Vanity URL)", inline=False)
            
        await self.log_to_server_log_channel(embed)

    async def on_member_remove(self, member):
        if ALLOWED_GUILDS and member.guild.id not in ALLOWED_GUILDS:
            return
        if member.bot:
            return
        
        embed = discord.Embed(
            title="📤 Citizen Left the Hold (Member Left)",
            color=discord.Color.light_grey(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.set_thumbnail(url=member.display_avatar.url if member.avatar else None)
        embed.add_field(name="User", value=f"{member.mention} ({member.name})", inline=True)
        embed.add_field(name="User ID", value=member.id, inline=True)
        
        if member.joined_at:
            stay_days = (datetime.datetime.now(datetime.timezone.utc) - member.joined_at).days
            embed.add_field(name="Stay Duration", value=f"{stay_days} days (Joined: {member.joined_at.strftime('%Y-%m-%d')})", inline=False)
            
        await self.log_to_server_log_channel(embed)

    async def on_member_ban(self, guild, user):
        if ALLOWED_GUILDS and guild.id not in ALLOWED_GUILDS:
            return
        if user.bot:
            return
        embed = discord.Embed(
            title="🔨 Member Banned",
            color=discord.Color.red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="User", value=f"{user.mention} ({user.name}#{user.discriminator if hasattr(user, 'discriminator') else ''} / ID: {user.id})", inline=False)
        
        banner = "Unknown"
        reason = "No reason provided"
        try:
            async for entry in guild.audit_logs(limit=5, action=discord.AuditLogAction.ban):
                if entry.target.id == user.id:
                    banner = f"{entry.user.mention} ({entry.user.name} / ID: {entry.user.id})"
                    if entry.reason:
                        reason = entry.reason
                    break
        except Exception:
            pass
        embed.add_field(name="Banned By", value=banner, inline=True)
        embed.add_field(name="Reason", value=reason, inline=True)
        
        await self.log_to_server_log_channel(embed)

    async def on_member_unban(self, guild, user):
        if ALLOWED_GUILDS and guild.id not in ALLOWED_GUILDS:
            return
        if user.bot:
            return
        embed = discord.Embed(
            title="🔓 Member Unbanned",
            color=discord.Color.green(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="User", value=f"{user.mention} ({user.name}#{user.discriminator if hasattr(user, 'discriminator') else ''} / ID: {user.id})", inline=False)
        
        unbanner = "Unknown"
        reason = "No reason provided"
        try:
            async for entry in guild.audit_logs(limit=5, action=discord.AuditLogAction.unban):
                if entry.target.id == user.id:
                    unbanner = f"{entry.user.mention} ({entry.user.name} / ID: {entry.user.id})"
                    if entry.reason:
                        reason = entry.reason
                    break
        except Exception:
            pass
        embed.add_field(name="Unbanned By", value=unbanner, inline=True)
        embed.add_field(name="Reason", value=reason, inline=True)
        
        await self.log_to_server_log_channel(embed)

    async def on_invite_create(self, invite):
        if ALLOWED_GUILDS and invite.guild.id not in ALLOWED_GUILDS:
            return
        if invite.inviter and invite.inviter.bot:
            return
            
        guild_id = invite.guild.id
        if guild_id in self.invites:
            try:
                self.invites[guild_id] = await invite.guild.invites()
            except Exception as e:
                print(f"Failed to refresh invites cache on invite create: {e}")

        embed = discord.Embed(
            title="✉️ Invite Link Created",
            color=discord.Color.dark_teal(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="Creator", value=f"{invite.inviter.mention if invite.inviter else 'System'} ({invite.inviter.name if invite.inviter else ''})", inline=True)
        embed.add_field(name="Invite Link", value=invite.url, inline=True)
        embed.add_field(name="Code", value=f"`{invite.code}`", inline=True)
        embed.add_field(name="Channel", value=invite.channel.mention if hasattr(invite.channel, "mention") else str(invite.channel), inline=True)
        embed.add_field(name="Max Uses", value=str(invite.max_uses) if invite.max_uses else "Infinite", inline=True)
        embed.add_field(name="Max Age", value=f"{invite.max_age} seconds" if invite.max_age else "Infinite", inline=True)
        
        await self.log_to_server_log_channel(embed)

    async def on_invite_delete(self, invite):
        if ALLOWED_GUILDS and invite.guild.id not in ALLOWED_GUILDS:
            return
            
        guild_id = invite.guild.id
        if guild_id in self.invites:
            try:
                self.invites[guild_id] = await invite.guild.invites()
            except Exception as e:
                print(f"Failed to refresh invites cache on invite delete: {e}")

        embed = discord.Embed(
            title="✉️ Invite Link Deleted",
            color=discord.Color.dark_red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        embed.add_field(name="Code", value=f"`{invite.code}`", inline=True)
        embed.add_field(name="Channel", value=invite.channel.mention if hasattr(invite.channel, "mention") else str(invite.channel), inline=True)
        
        await self.log_to_server_log_channel(embed)

    async def analyze_channel_style(self, channel):
        if isinstance(channel, discord.DMChannel):
            return

        channel_id = str(channel.id)
        print(f"Starting style analysis for channel {channel.name} ({channel_id})...")

        import datetime
        history_msgs = []
        try:
            async for msg in channel.history(limit=100):
                if not msg.author.bot:
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    delta = now_utc - msg.created_at
                    if delta.total_seconds() <= 86400: # 24 hours
                        history_msgs.append(msg)
        except Exception as e:
            print(f"Error fetching channel history for style analysis: {e}", file=sys.stderr)
            return

        if len(history_msgs) < 5:
            print(f"Not enough messages ({len(history_msgs)}) in the last 24h to analyze style for {channel.name}.")
            return

        history_msgs.reverse()
        chat_sample = ""
        for msg in history_msgs[:50]: # Limit to 50 messages to save prompt tokens
            chat_sample += f"[{msg.author.display_name}]: {msg.clean_content}\n"

        prompt = f"""
        You are analyzing the conversation style of a Discord channel based on the messages from the last 24 hours.
        Your goal is to extract formatting, length, tone, internet slang, and greetings from the users so another bot can mimic them.
        
        Recent chat history sample:
        ---
        {chat_sample}
        ---

        Analyze the messages above and respond ONLY with a JSON object containing these exact fields:
        - "tone": a brief description of the channel's tone (e.g. "sarcastic and casual", "enthusiastic", "serious")
        - "avg_length": the typical message length (e.g. "extremely short (under 5 words)", "1-2 short sentences", "long structured text")
        - "lingo": a list of common internet slang words, abbreviations, or keywords frequently used in the chat
        - "greetings": typical greetings used (e.g. "yo", "sup", "hey guys", or none)
        - "style_guidelines": a list of 2-3 specific bullet-point style rules that capture how users communicate in this channel

        Ensure the response is valid JSON.
        """

        try:
            content = await llm_balancer.generate_chat_completion(
                "You are a linguistic analyzer specializing in Discord chat style extraction. Return JSON only.",
                prompt,
                response_json=True
            )
            result = json.loads(content)

            styles = self.load_channel_styles()
            styles[channel_id] = {
                "last_updated": time.time(),
                "tone": result.get("tone", "casual"),
                "avg_length": result.get("avg_length", "short"),
                "lingo": result.get("lingo", []),
                "greetings": result.get("greetings", ""),
                "style_guidelines": result.get("style_guidelines", [])
            }
            self.save_channel_styles(styles)
            print(f"Successfully updated style profile for channel {channel.name} ({channel_id}).")
            DashboardLogger.log_event("STYLE_ANALYSIS", {
                "channel_name": channel.name,
                "profile": styles[str(channel_id)]
            })

        except Exception as e:
            print(f"Error during channel style analysis: {e}", file=sys.stderr)

    async def on_ready(self):
        print(f"Windhelm Guard Discord Bot logged in as {self.user} (ID: {self.user.id})")
        
        if not self.dashboard_started:
            asyncio.create_task(start_dashboard())
            self.dashboard_started = True

        global ALLOWED_GUILDS
        if not ALLOWED_GUILDS:
            print("No ALLOWED_GUILDS specified in .env. Defaulting to allowing ALL guilds.")
        else:
            print(f"Allowed Guilds loaded from .env: {ALLOWED_GUILDS}")

        self.ready_to_chat = True
        print("\nReady for conversational interaction.")

        # Cache server invites for join-tracking
        self.invites = {}
        for guild in self.guilds:
            if ALLOWED_GUILDS and guild.id not in ALLOWED_GUILDS:
                continue
            try:
                self.invites[guild.id] = await guild.invites()
                print(f"Cached {len(self.invites[guild.id])} invites for guild {guild.name}")
            except Exception as e:
                print(f"Failed to cache invites for guild {guild.name}: {e}")

    async def on_relationship_add(self, relationship):
        if getattr(relationship, 'type', None) == discord.RelationshipType.incoming_request:
            try:
                await relationship.accept()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Automatically accepted friend request from {relationship.user}")
            except Exception as e:
                print(f"Failed to accept friend request: {e}")

    async def analyze_image_attachment(self, url):
        prompt = """Analyze this image and return a JSON object with the following fields:
- nsfw_or_sensitive (boolean): true if the image contains explicit NSFW, nudity, heavy gore, or sensitive themes.
- sender_distressed_or_harsh (boolean): true if the image depicts self-harm, dark/grim/depressing content showing the sender might be going through a hard time, or very harsh/distressing themes.
- goes_against_rules (boolean): true if the image violates server/community rules (like explicit NSFW content, heavy gore, hate speech, slurs, doxxing, harassment).
- explanation (string): a brief explanation of the evaluation.
- description (string): a short, clear description of the image content.
"""
        try:
            raw_res = await llm_balancer.generate_vision_completion(url, prompt=prompt, response_json=True)
            if raw_res:
                return json.loads(raw_res)
        except Exception as e:
            print(f"Vision API Error: {e}")
        return None

    async def generate_image_response(self, context, image_description, response_type, channel_style_guidelines, channel_lingo):
        if response_type == "concern":
            instruction = (
                "You noticed that a citizen posted an image that depicts a very harsh or depressing situation, indicating they might be going through a hard time.\n"
                "Even though you are a rugged city guard, you have a soft spot and want to show genuine concern for this citizen.\n"
                "Write a public reply in your Skyrim Guard persona (refer to sweetrolls, taking an arrow to the knee, mead, the Jarl, dragons, or guard duty, but express concern and offer support)."
            )
        else:
            instruction = (
                "You noticed that a citizen posted an image that goes against community rules (contains NSFW, nudity, gore, or other prohibited content in a channel not marked for NSFW).\n"
                "Write a public warning reply in your Skyrim Guard persona (tell them to halt, keep their hands to themselves, no lollygagging, or that the Jarl's dungeon/guards await if they continue to break rules)."
            )

        system_prompt = f"""You are a City Guard from The Elder Scrolls V: Skyrim.
Your objective is to reply to a citizen's message in the channel.
{instruction}

Make sure to match the channel style guidelines and lingo if possible:
CHANNEL STYLE GUIDELINES:
{channel_style_guidelines}
{channel_lingo}

Write a short, natural reply (usually 1-2 short sentences, matching the channel style guidelines). Speak in the first person as the Windhelm Guard. Do not use markdown blocks or say things like 'Guard:' prefix. Just output the reply text.
"""
        user_prompt = f"""CONVERSATION CONTEXT:
{context}

IMAGE IN TARGET MESSAGE DESCRIPTION:
{image_description}
"""
        try:
            return await llm_balancer.generate_chat_completion(system_prompt, user_prompt)
        except Exception as e:
            print(f"Error generating image response: {e}", file=sys.stderr)
            return "Keep your hands to yourself, sneak thief."

    @staticmethod
    def format_discord_message(msg):
        content = msg.clean_content
        if getattr(msg, 'stickers', None):
            content += f" [Sticker: {msg.stickers[0].name}]"
        if getattr(msg, 'attachments', None):
            for a in msg.attachments:
                content += f" [Attachment: {a.filename}]"
        return content.strip()

    async def on_message(self, message):
        if getattr(self, 'ready_to_chat', False) is False:
            return

        if message.author.id == self.user.id:
            return

        MOD_CHANNEL_ID = os.environ.get("MOD_CHANNEL_ID", "")

        # 1. Ignore message if guild is not allowed
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Log all DM conversations
        if is_dm:
            dm_folder = os.path.join(SCRIPT_DIR, "DMs")
            os.makedirs(dm_folder, exist_ok=True)
            
            recipient_name = "Unknown"
            if hasattr(message.channel, 'recipient') and message.channel.recipient:
                recipient_name = message.channel.recipient.name
                
            safe_name = "".join([c for c in recipient_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            log_file = os.path.join(dm_folder, f"dm_{message.channel.id}_{safe_name}.txt")
            
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message.author.display_name}: {message.clean_content}\n")

        # 2. Ignore other bots to prevent loops
        if message.author.bot:
            return

        # 2.4. Check allowed guilds
        if ALLOWED_GUILDS and not is_dm and message.guild:
            if message.guild.id not in ALLOWED_GUILDS:
                return

        # 2.5. Check allowed channels
        if ALLOWED_CHANNELS and not is_dm and message.channel.id not in ALLOWED_CHANNELS:
            return

        clean_text = message.clean_content.strip()
        is_guard_alias = clean_text.lower().startswith("guard ") or clean_text.lower() == "guard"
        is_mentioned = (self.user in message.mentions and not message.mention_everyone) or is_guard_alias
        OWNER_ID = os.environ.get("OWNER_ID", "").strip()
        is_owner = OWNER_ID and str(message.author.id) == OWNER_ID

        # 3. Check channel-specific reply cooldown (8 seconds)
        now = time.time()
        channel_id = message.channel.id
        last_reply = self.cooldowns.get(channel_id, 0)
        if not is_owner:
            if now - last_reply < 8:
                return

        # 3.5. Global spam protection (one channel at a time unprompted)
        if not is_dm and not is_mentioned and not is_owner:
            if now - self.last_global_reply < 45: # 45 seconds global unprompted cooldown
                return

        # 4. Trigger style analysis check asynchronously in background
        if not is_dm:
            chan_id_str = str(channel_id)
            styles = self.load_channel_styles()
            last_updated = styles.get(chan_id_str, {}).get("last_updated", 0)
            if now - last_updated > 3600: # 1 hour
                asyncio.create_task(self.analyze_channel_style(message.channel))

        # 5. Fetch channel history (last 30 messages)
        # 5. Fetch channel history (last 30 messages), filtering out bot messages and old commands to prevent context confusion
        history_msgs = []
        try:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            async for msg in message.channel.history(limit=30):
                if (now_utc - msg.created_at).total_seconds() < 7200:
                    # Exclude bot messages
                    if msg.author.bot:
                        continue
                    # Exclude old commands where the bot was mentioned to prevent context confusion
                    # Keep the current message itself even if it mentions the bot
                    if msg.id != message.id and self.user in msg.mentions:
                        continue
                    history_msgs.append(msg)
        except Exception as e:
            print(f"Error fetching channel history: {e}", file=sys.stderr)
            history_msgs = [message]

        history_msgs.reverse()
        history_str = ""
        for msg in history_msgs:
            author_name = "You" if msg.author.id == self.user.id else msg.author.display_name
            history_str += f"{author_name}: {self.format_discord_message(msg)}\n"

        # 6. Load style guidelines
        channel_style_guidelines = ""
        channel_lingo = ""
        
        if not is_dm:
            chan_id_str = str(channel_id)
            styles = self.load_channel_styles()
            profile = styles.get(chan_id_str)
            if profile:
                tone = profile.get("tone", "")
                avg_length = profile.get("avg_length", "")
                lingo_list = profile.get("lingo", [])
                guidelines = profile.get("style_guidelines", [])

                channel_style_guidelines = (
                    f"Format rules matching this channel:\n"
                    f"- General Tone: {tone}\n"
                    f"- Length guideline: {avg_length}\n"
                )
                if guidelines:
                    channel_style_guidelines += "Specific rules:\n"
                    for rule in guidelines:
                        channel_style_guidelines += f"- {rule}\n"

                if lingo_list:
                    channel_lingo = f"Lingo/Slang you can use if natural: {', '.join(lingo_list)}"

        # 6.5. Rare periodic unprompted fun comment check (0.1% chance when no active conflict)
        if not is_dm and not is_mentioned and not is_owner and message.guild:
            conflict_state = self.load_conflict_state()
            chan_state = conflict_state.get(str(channel_id), {"is_conflict_active": False, "convo_toxicity": 0})
            if not chan_state.get("is_conflict_active", False):
                if random.random() < 0.001:  # 0.1% chance on every message
                    print("Triggering random 0.1% context fun comment...")
                    comment = await self.generate_fun_comment(message.channel, history_str, channel_style_guidelines, channel_lingo)
                    if comment:
                        self.cooldowns[channel_id] = now
                        self.last_global_reply = now
                        await message.reply(comment)
                        state = self.load_random_comment_state()
                        state["last_comment_time"] = now
                        self.save_random_comment_state(state)
                        DashboardLogger.log_event("RANDOM_FUN_COMMENT", {
                            "channel": message.channel.name,
                            "comment": comment
                        })
                        return

        channel_info = "Direct Message" if is_dm else f"Server: {message.guild.name}, Channel: #{message.channel.name}"

        # Analyze images in current message
        current_msg_images = ""
        analyzed_images_list = []
        if message.attachments:
            for a in message.attachments:
                if getattr(a, 'content_type', '') and a.content_type.startswith('image/'):
                    img_data = await self.analyze_image_attachment(a.url)
                    if img_data:
                        analyzed_images_list.append(img_data)
                        desc = img_data.get("description", "")
                        explanation = img_data.get("explanation", "")
                        current_msg_images += f"\n[Image '{a.filename}' Content: {desc}. Analysis: {explanation}]"

        rules = get_rules()
        system_prompt = f"""You are a City Guard from The Elder Scrolls V: Skyrim.
Your objective is to silently observe the conversation, analyze toxicity against community rules, and protect the server.
You must blend your Skyrim Guard persona (arrow to the knee, stolen sweetroll, lollygagging, the Jarl, dragons) with the channel's communication style (casual, lowercase, internet slang). You are a "chronically online" Skyrim Guard.

COMMUNITY RULES (rules.txt):
{rules}

CHANNEL STYLE GUIDELINES:
{channel_style_guidelines}
{channel_lingo}

MODERATION CRITERIA:
- Toxicity Definition: Toxicity is defined as heated arguments between members. Profane or abusive words alone are permissible unless they are part of a heated argument.
- GREETING SAFETY: Casual greetings (such as "hi", "hello", "hey", "good morning") or simple safe messages are 100% NOT toxic. They must always be classified as "none" with toxicity score 0. Under no circumstances can "hi" or casual chat be classified as an offense.
- COMMAND REQUEST SAFETY: Command requests (such as asking the bot to timeout, mute, check a user's profile, set a reminder, or show summaries) are NOT offenses. They are commands. They must always be classified as "none" with toxicity score 0. Even if requested by an unauthorized user, it is NOT an offense (the Python code handles authorization and will simply ignore it).
- CONTEXT ISOLATION: Evaluate ONLY the content and intent of the latest message (the last line of history). Do NOT associate toxic words or insults from previous messages (by other users) to the current author. For example, if User A said something toxic earlier, and User B now says "hi" or "whats my score", User B's message must be classified as "none" with toxicity score 0.
- Evaluate ONLY the latest message (the last line of history) for toxicity, arguments, or commands. Do not re-evaluate or execute past commands in the history.
- Only genuine harassment, severe insults, hate speech, slurs, doxxing, or destructive arguing should be flagged as high toxicity (95+) and warrant intervention or mod reporting.
- INTERACTIVE CHAT MODE: If IS_BEING_ADDRESSED is True, you are being addressed/mentioned! You MUST set "should_intervene_publicly" to true and write a creative, natural conversational response in "public_reply". Speak directly and exactly like a City Guard from Skyrim speaking to a citizen (mention mead, sweetrolls, dragons, taking an arrow to the knee, Jarl, etc.) adapted to the channel's communication style (usually casual, lower-case). Do NOT remain silent, do NOT leave "public_reply" empty, and do NOT just say "No lollygagging" unless it uniquely fits the context.
- SILENT MODE: If you are NOT mentioned or addressed (IS_BEING_ADDRESSED is False), you must strictly stay silent (set "should_intervene_publicly" to false) unless a severe rule violation (toxicity 95+) is occurring.

OFFENSE CLASSIFICATION:
You must classify the message into one of three classifications in "offense_classification":
1. "direct_rule_break": A clear and direct violation of the community rules (such as transphobia, queerphobia, NSFW content in SFW channels, chaser behavior, severe hate speech, slurs, doxxing, or severe insults/harassment). NOTE: Command requests and greetings are NEVER direct rule breaks.
2. "disrespectful": Playing around the boundaries of the rules, being intentionally disrespectful, mean, or using small normal non-abusive words that target someone disrespectfully. NOTE: Command requests and greetings are NEVER disrespectful.
3. "none": Normal conversation, friendly chat, greetings, questions, command requests, or healthy disagreements. Playful banter is defined as friendly jokes or teasing where intent is clear and the recipient does not express discomfort.

You must perform intent and recipient reaction analysis in "intent_analysis":
- Explain the sender's intent: is it a harmless joke, a command request, friendly teasing, or intentional disrespect/harassment?
- Analyze if the recipient or other users in the chat history show any signs of discomfort or objection. If they are joking/laughing along, it is playful banter (none). If they express clear discomfort or ask the sender to stop, and the sender continues, it is disrespectful or a rule break.
- State your reasoning for the offense classification in "offense_reason".

NATURAL LANGUAGE COMMANDS:
- Reminders: If a user asks you to remind them of something (e.g. "@Windhelm Guard remind me in 5 minutes to study"), set:
  - "set_reminder" to true
  - "reminder_delay_seconds" to the delay in seconds (e.g., 300)
  - "reminder_text" to the reminder content (e.g., "study")
- Profile Lookup: If a user asks to see/check/show the profile of a user (e.g. "@Windhelm Guard show profile of @User"), set:
  - "is_requesting_profile" to true
  - "profile_target_user" to the username or mention of the target user.
- Summaries & Judgements:
  - If asked for a summary/log, set "is_requesting_summary" to true.
  - If asked to judge a user/situation, set "is_requesting_judgement" to true.
  - For judgements/summaries, do NOT leak real mod details/names publicly. Keep the public reply anonymous and Skyrim-themed, and put details in "mod_report_reason".
- Moderation Commands: If the message asks you to execute a moderator action (such as timeout, untimeout, kick, ban, unban, clear, lock, unlock, slowmode, warn, or resetting/pardoning/clearing a user's toxicity record), you MUST extract the action and targets:
  - "command_to_execute": set to "timeout", "untimeout", "kick", "ban", "unban", "clear", "lock", "unlock", "slowmode", "warn", "reset_toxicity", or "none"
  - "command_target": the name, mention, ID of the user or channel target
  - "command_args": a dictionary containing:
    - "duration_mins" (int or null): duration in minutes. E.g. "timeout for a day" -> 1440. "timeout for 10 minutes" -> 10.
    - "reason" (string or null): the reason for the mod action.
    - "clear_limit" (int or null): number of messages to clear for clear command.
    - "slowmode_seconds" (int or null): slowmode delay in seconds.

Current Context:
- {channel_info}
{current_msg_images}

Your task:
1. Review the conversation history.
2. Evaluate the last message and overall context for rule violations, commands, or mentions.
3. If should_intervene_publicly is true, write a reply in "public_reply" in your Skyrim Guard persona.
4. If IS_BEING_ADDRESSED is True, you MUST set "should_intervene_publicly" to true and write a creative, natural conversational response in "public_reply" matching your Skyrim Guard persona. Speak directly as the Windhelm Guard himself talking directly to the citizen. Do not speak in third person about guards, and avoid repeating "No lollygagging" unless it uniquely fits the conversation context.
5. If should_report_to_mods is true, write the reason in "mod_report_reason".

You MUST respond ONLY with a JSON object matching this schema:
{{
  "toxicity_level": 0,
  "offense_classification": "none",
  "offense_reason": "explanation of classification/offense if any, else empty",
  "intent_analysis": "analysis of the user's intent and target user's reaction",
  "is_argument": false,
  "is_requesting_summary": false,
  "is_requesting_judgement": false,
  "is_requesting_profile": false,
  "profile_target_user": null,
  "command_to_execute": "none",
  "command_target": null,
  "command_args": {{
    "duration_mins": null,
    "reason": null,
    "clear_limit": null,
    "slowmode_seconds": null
  }},
  "should_intervene_publicly": false,
  "public_reply": "your response/de-escalation/command confirmation (empty if should_intervene_publicly is false)",
  "should_report_to_mods": false,
  "mod_report_reason": "",
  "set_reminder": false,
  "reminder_delay_seconds": null,
  "reminder_text": null
}}
"""
        user_msg = f"""CONVERSATION CONTEXT (last 30 messages, oldest to newest):
{history_str}

==================================================
TARGET MESSAGE TO ANALYZE (Evaluate ONLY this message for toxicity and commands):
Sender: {message.author.display_name} (ID: {message.author.id})
Message Content: {message.clean_content}
IS_BEING_ADDRESSED (Bot was mentioned or addressed as 'guard'): {is_mentioned}
==================================================

Analyze the TARGET MESSAGE content and the sender's intent. Do not attribute any text, toxicity, or commands from the CONVERSATION CONTEXT to this sender. If the target message itself does not violate rules, toxicity must be 0 and offense_classification must be "none".
"""

        try:
            content = await llm_balancer.generate_chat_completion(
                system_prompt,
                user_msg,
                response_json=True
            )
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error from LLM: {e}")
                return

            toxicity_level = result.get("toxicity_level", 0)
            is_argument = result.get("is_argument", False)
            is_requesting_summary = result.get("is_requesting_summary", False)
            is_requesting_judgement = result.get("is_requesting_judgement", False)
            is_requesting_profile = result.get("is_requesting_profile", False)
            should_intervene = result.get("should_intervene_publicly", False)
            public_reply = result.get("public_reply", "")
            should_report = result.get("should_report_to_mods", False)
            mod_reason = result.get("mod_report_reason", "")
            offense_classification = result.get("offense_classification", "none")
            offense_reason = result.get("offense_reason", "")
            intent_analysis = result.get("intent_analysis", "")

            # Execute commands if mentioned, in DM, or if requestor is the owner (Jarl)
            command_processed = False
            if is_mentioned or is_dm or is_owner:
                if result.get("command_to_execute") and result.get("command_to_execute") != "none":
                    await self.execute_command(message, result)
                    should_intervene = False
                    public_reply = ""
                    command_processed = True
                elif result.get("set_reminder"):
                    await self.handle_reminder(message, result)
                    command_processed = True
                elif is_requesting_profile:
                    await self.handle_profile_request(message, result)
                    should_intervene = False
                    public_reply = ""
                    command_processed = True

            # If user requested a summary
            if is_requesting_summary and MOD_CHANNEL_ID:
                members = self.load_critical_members()
                mod_channel = self.get_channel(int(MOD_CHANNEL_ID))
                if mod_channel:
                    report = "**🚨 Jarl's Intelligence Summary 🚨**\n"
                    if not members:
                        report += "No critical members logged yet."
                    else:
                        for uid, data in members.items():
                            report += f"- **{data['name']}**: Cumulative Toxicity Score: {data['toxicity_score']}, Offenses: {data.get('total_offenses', len(data.get('offenses', [])))}\n"
                    await mod_channel.send(report)
                
                # Make sure the bot acknowledges it publicly if asked publicly
                if str(message.channel.id) != MOD_CHANNEL_ID:
                    should_intervene = True
                    if not public_reply:
                        public_reply = "The intelligence report has been delivered securely to the Jarl's steward."
                command_processed = True

            # If user requested a judgement
            if is_requesting_judgement:
                should_report = True
                if not mod_reason:
                    mod_reason = "A citizen requested a judgement on this situation, but no detailed assessment was provided by the guard."
                if str(message.channel.id) != MOD_CHANNEL_ID:
                    should_intervene = True
                    if not public_reply:
                        public_reply = "We have our eyes on a certain citizen. Keep your hands to yourself and there won't be any trouble."
                command_processed = True

            # Stateful Conflict Tracking
            chan_id_str = str(message.channel.id)
            conflict_state = self.load_conflict_state()
            chan_state = conflict_state.get(chan_id_str, {"is_conflict_active": False, "convo_toxicity": 0})
            
            was_conflict_active = chan_state.get("is_conflict_active", False)

            # If a new severe violation (95+) is detected, activate conflict state
            if toxicity_level >= 95:
                chan_state["is_conflict_active"] = True
                chan_state["convo_toxicity"] = toxicity_level
            
            # If conflict is active, we continue to check the decay / new toxicity level
            if chan_state["is_conflict_active"]:
                chan_state["convo_toxicity"] = toxicity_level
                # If toxicity drops below 90, deactivate the conflict state
                if chan_state["convo_toxicity"] < 90:
                    chan_state["is_conflict_active"] = False
                    
            # Check if conflict was resolved
            if was_conflict_active and not chan_state["is_conflict_active"]:
                print(f"[Conflict Resolution] Conflict in channel {message.channel.name} has been resolved! Dropping toxicity scores.")
                members = self.load_critical_members()
                for uid, udata in members.items():
                    old_score = udata.get("toxicity_score", 0.0)
                    new_score = max(0.0, old_score - 30.0)
                    udata["toxicity_score"] = new_score
                    if new_score < 90.0:
                        udata["last_warned_threshold"] = 0
                    elif new_score < 95.0:
                        udata["last_warned_threshold"] = min(udata.get("last_warned_threshold", 0), 90)
                self.save_critical_members(members)

            # Save the updated conflict state
            conflict_state[chan_id_str] = chan_state
            self.save_conflict_state(conflict_state)

            # Enforce public silence unless there is an active conflict OR bot is mentioned/DM'd
            if not is_mentioned and not is_dm:
                if not chan_state["is_conflict_active"]:
                    should_intervene = False
                    public_reply = ""

            # 6.6 Image sensitive probabilistic checks at high-stake situations
            is_nsfw_ok = False
            if is_dm:
                is_nsfw_ok = True
            elif hasattr(message.channel, 'is_nsfw'):
                if message.channel.is_nsfw or message.channel.id in NSFW_ALLOWED_CHANNELS:
                    is_nsfw_ok = True

            if analyzed_images_list and not is_nsfw_ok:
                has_sensitive = any(img.get("nsfw_or_sensitive", False) for img in analyzed_images_list)
                if has_sensitive:
                    is_high_stake = (toxicity_level >= 95 or chan_state.get("is_conflict_active", False))
                    if is_high_stake:
                        distressed_images = [img for img in analyzed_images_list if img.get("sender_distressed_or_harsh", False)]
                        rule_breaking_images = [img for img in analyzed_images_list if img.get("goes_against_rules", False)]
                        
                        distress_triggered = False
                        if distressed_images:
                            roll = random.random()
                            print(f"[Image Reader] High-stake distress check. Roll: {roll:.3f} (threshold: 0.10)")
                            if roll < 0.10:
                                img_desc = distressed_images[0].get("description", "")
                                reply_text = await self.generate_image_response(
                                    context=history_str,
                                    image_description=img_desc,
                                    response_type="concern",
                                    channel_style_guidelines=channel_style_guidelines,
                                    channel_lingo=channel_lingo
                                )
                                if reply_text:
                                    should_intervene = True
                                    public_reply = reply_text
                                    distress_triggered = True
                                    DashboardLogger.log_event("IMAGE_SENSITIVE_ALERT", {
                                        "channel": channel_info,
                                        "user": f"{message.author.display_name} ({message.author.id})",
                                        "image_description": img_desc,
                                        "reply": reply_text
                                    })
                        
                        if rule_breaking_images and not distress_triggered:
                            roll = random.random()
                            print(f"[Image Reader] High-stake rules violation check. Roll: {roll:.3f} (threshold: 0.50)")
                            if roll < 0.50:
                                img_desc = rule_breaking_images[0].get("description", "")
                                reply_text = await self.generate_image_response(
                                    context=history_str,
                                    image_description=img_desc,
                                    response_type="warning",
                                    channel_style_guidelines=channel_style_guidelines,
                                    channel_lingo=channel_lingo
                                )
                                if reply_text:
                                    should_intervene = True
                                    public_reply = reply_text
                                    DashboardLogger.log_event("IMAGE_WARNING_ALERT", {
                                        "channel": channel_info,
                                        "user": f"{message.author.display_name} ({message.author.id})",
                                        "image_description": img_desc,
                                        "reply": reply_text
                                    })
            
            DashboardLogger.log_event("MODERATOR_EVALUATION", {
                "channel": channel_info,
                "message": self.format_discord_message(message),
                "toxicity_level": toxicity_level,
                "offense_classification": offense_classification,
                "offense_reason": offense_reason,
                "intent_analysis": intent_analysis,
                "is_argument": is_argument,
                "should_intervene": should_intervene,
                "should_report": should_report
            })

            # Track Critical Members
            author_id = str(message.author.id)
            author_name = message.author.display_name
            
            members = self.load_critical_members()
            fd = members.get(author_id, {
                "name": author_name,
                "toxicity_score": 0.0,
                "last_toxic_time": None,
                "last_warned_threshold": 0,
                "offenses": [],
                "total_offenses": 0
            })
            if "offenses" not in fd or not isinstance(fd["offenses"], list):
                fd["offenses"] = []
            if "toxicity_score" not in fd:
                fd["toxicity_score"] = 0.0
            else:
                fd["toxicity_score"] = float(fd["toxicity_score"])
            if "last_toxic_time" not in fd:
                fd["last_toxic_time"] = None
            if "last_warned_threshold" not in fd:
                fd["last_warned_threshold"] = 0

            # 1. Apply Decay if user has previous score & toxic timestamp
            now = time.time()
            old_score = fd["toxicity_score"]
            if fd["last_toxic_time"] is not None and old_score > 0:
                elapsed_hours = (now - float(fd["last_toxic_time"])) / 3600.0
                if old_score >= 50.0:
                    decay_amount = 3.0 * (elapsed_hours // 12.0)
                else:
                    decay_amount = 1.0 * (elapsed_hours // 6.0)
                
                if decay_amount > 0:
                    new_score = max(0.0, old_score - decay_amount)
                    fd["toxicity_score"] = new_score
                    
                    if old_score >= 50.0:
                        decayed_seconds = (decay_amount / 3.0) * 12.0 * 3600.0
                    else:
                        decayed_seconds = (decay_amount / 1.0) * 6.0 * 3600.0
                    fd["last_toxic_time"] = float(fd["last_toxic_time"]) + decayed_seconds
                    
                    # Update threshold tracker based on decayed score
                    if fd["toxicity_score"] < 90.0:
                        fd["last_warned_threshold"] = 0
                    elif fd["toxicity_score"] < 95.0:
                        fd["last_warned_threshold"] = min(fd["last_warned_threshold"], 90)
                    
                    old_score = fd["toxicity_score"] # update old_score to decayed value

            # 2. Map LLM offense classification to points
            points_to_add = 0.0
            recorded_reason = ""
            
            if offense_classification == "direct_rule_break":
                points_to_add = 10.0
                should_report = True  # force report to mod channel
                recorded_reason = f"[Rule Break] {offense_reason or mod_reason or 'Direct rule violation'}"
            elif offense_classification == "disrespectful":
                points_to_add = 2.0
                should_report = False # do NOT report small offenses
                recorded_reason = f"[Disrespectful] {offense_reason or 'Disrespectful behavior'}"
            else:
                # none
                points_to_add = 0.0
                should_report = False
            
            if points_to_add > 0:
                fd["toxicity_score"] += points_to_add
                fd["last_toxic_time"] = now
                
                # Record detailed offense
                timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                fd["offenses"].append({
                    "timestamp": timestamp_str,
                    "toxicity_level": toxicity_level,
                    "reason": recorded_reason,
                    "message": message.clean_content
                })
                fd["total_offenses"] = len(fd["offenses"])

            # 3. Check Threshold crossings
            new_score = fd["toxicity_score"]
            crossed_threshold = None
            
            if new_score >= 95.0 and fd["last_warned_threshold"] < 95:
                crossed_threshold = 95
                should_intervene = True
                public_reply = f"Enough of this insolence, {message.author.mention}! You've been warned before, and now you're playing with fire. The Jarl's guards are ready to haul you away. Behave, or else."
                fd["last_warned_threshold"] = 95
            elif new_score >= 90.0 and new_score < 95.0 and fd["last_warned_threshold"] < 90:
                crossed_threshold = 90
                should_intervene = True
                public_reply = f"Halt right there, {message.author.mention}! Your behavior in this hold is pushing the boundaries of the Jarl's patience. Keep it civil, or you'll find yourself in the Dragonsreach dungeon."
                fd["last_warned_threshold"] = 90
                
            # Save updated profile
            members[author_id] = fd
            self.save_critical_members(members)
            
            # Send summary to mod channel if threshold was crossed
            if crossed_threshold is not None:
                await self.send_threshold_mod_summary(fd, author_id, crossed_threshold, message.jump_url)

            # Private Mod Report (Embed format with Context)
            if should_report and MOD_CHANNEL_ID:
                mod_channel = self.get_channel(int(MOD_CHANNEL_ID))
                if mod_channel:
                    embed = discord.Embed(
                        title="🚨 Threat Evaluation: Mod Alert",
                        color=discord.Color.red(),
                        timestamp=datetime.datetime.now(datetime.timezone.utc)
                    )
                    embed.add_field(name="User", value=f"{author_name} (<@{author_id}>)", inline=True)
                    embed.add_field(name="Channel", value=f"<#{message.channel.id}>", inline=True)
                    embed.add_field(name="Toxicity Score", value=f"**{toxicity_level}/100**", inline=True)
                    embed.add_field(name="Violation Reason", value=recorded_reason or mod_reason or "Direct rule violation", inline=False)
                    embed.add_field(name="Triggering Message", value=f"\"{message.clean_content}\"", inline=False)
                    
                    # Add conversation context (last 4 messages before this one)
                    context_msgs = [m for m in history_msgs if m.id != message.id][-4:]
                    recent_context = ""
                    for m in context_msgs:
                        recent_context += f"**{m.author.display_name}**: {m.clean_content[:150]}\n"
                    if not recent_context:
                        recent_context = "*No previous messages in context.*"
                    embed.add_field(name="Recent Chat Context", value=recent_context, inline=False)
                    
                    embed.add_field(name="Action Link", value=f"[Jump to Message]({message.jump_url})", inline=False)
                    embed.set_footer(text=f"User ID: {author_id}")
                    
                    await mod_channel.send(embed=embed)

            if is_mentioned and not command_processed:
                should_intervene = True
                if not public_reply:
                    GUARD_QUOTES = [
                        "I used to be an adventurer like you. Then I took an arrow in the knee...",
                        "Let me guess. Someone stole your sweetroll?",
                        "No lollygagging.",
                        "Disrespect the law, and you disrespect me.",
                        "What is it? Dragons?",
                        "Watch the skies, traveler.",
                        "Got to thinking, maybe I'm the Dragonborn, and I just don't know it yet?",
                        "My cousin's out fighting dragons, and what do I get? Guard duty.",
                        "Wait... I know you.",
                        "Hands to yourself, sneak thief.",
                        "Citizen.",
                        "Fear not, for the guards of Windhelm are ever vigilant."
                    ]
                    public_reply = random.choice(GUARD_QUOTES)

            # Public De-escalation or Reply
            if should_intervene and public_reply:
                self.cooldowns[channel_id] = time.time()
                if not is_dm and not is_mentioned:
                    self.last_global_reply = time.time()
                
                await message.reply(public_reply)
        return
                
        except Exception as e:
            print(f"Error in Discord response generation: {e}", file=sys.stderr)
            if is_dm or is_mentioned:
                await message.channel.send("Focus and consistency are the keys to mastery.")

def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        print("Error: DISCORD_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("Initializing Standalone Windhelm Guard Discord Bot...")
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    discord_client = WindhelmGuardDiscordClient(intents=intents)
    
    try:
        discord_client.run(token)
    except KeyboardInterrupt:
        print("\nBot shutting down.")
    except Exception as e:
        print(f"Error running Discord bot: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
