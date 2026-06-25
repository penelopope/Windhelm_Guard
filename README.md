# Windhelm Guard Discord Bot

A standalone Discord moderation and interaction bot themed as a Skyrim Windhelm Guard, powered by Groq/Gemini/Deepseek/OpenAI load balancer and discord.py.

## Prerequisites

Make sure you have Python 3 installed. You will also need to install the required Python packages:

```bash
pip install discord.py groq
```

## Setup

1. **Environment Variables**: Create a `.env` file in the same directory as the bot script (or copy `.env.example` if available) and add the following keys:
   ```env
   # Your Discord Bot Token
   DISCORD_TOKEN="your_discord_bot_token_here"
   
   # Your Groq API Key for LLM generation
   GROQ_API_KEY="your_groq_api_key_here"
   
   # (Optional) Comma-separated list of channel IDs where the bot is allowed to chat. 
   # If left empty, the bot will chat in all channels it has access to.
   ALLOWED_CHANNELS="123456789012345678,987654321098765432"
   ```

2. **Persona & Context**:
   - `persona.txt`: Describe the bot's core personality here.
   - `convo_data.txt`: Provide text examples or chat logs representing how the bot should speak.

## Features

- **Server (Guild) Whitelisting**: Use `ALLOWED_GUILDS` to restrict the bot's activity to entire specific servers.
- **Channel Whitelisting**: Use `ALLOWED_CHANNELS` to restrict the bot's activity to specific channels. It will still respond to DMs by default.
- **Personal Mentions**: The bot will specifically answer when its user is directly mentioned (ignores `@everyone` and `@here`).
- **Human-like Delays**: Responses have a calculated typing delay based on message length, while triggering the "is typing..." status in Discord.
- **Dynamic Channel Styles**: The bot analyzes channel history every hour to adapt its tone, length, and slang to match the local chat culture.

## Running the Bot

Start the bot by running:
```bash
python3 windhelm_guard_discord_bot.py
```
