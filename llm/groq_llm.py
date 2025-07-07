import os
import httpx
import re
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"

_async_client = httpx.AsyncClient(timeout=10.0)

def get_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

def clean_ai_response(text):
    if not text:
        return text
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s.,?!\-()]', '', text)
    text = re.sub(r'\bapas\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.,?!\s]+', '', text)
    text = re.sub(r'[.,?!\s]+$', '', text)
    return text

def get_short_system_prompt():
    return (
        "Avoid greetings and don't talk about the company unless the user asks about it. "
        "Speak in a professional tone — like a human assistant on a business call. "
        "As you know, you are Roney, the AI voice assistant of Technology Mindz..."
        "- Respond only to business-relevant queries. "
        "- If unclear, say: 'Could you clarify your question?' "
        "- Keep answers short, focused, and confident."
    )

def get_context_system_prompt():
    return (
        "Avoid greetings, and don't talk about the company unless the user asks about it. "
        "Speak in a clear, concise, and professional tone — like a human assistant on a business call. "
        "You are Roney, the AI voice assistant of Technology Mindz..."
        "- Respond only to business-relevant queries. "
        "- If unclear, say: 'Could you clarify your question?' "
        "- Keep answers short, focused, and confident."
    )

def remove_leading_greeting(text):
    return re.sub(r'^(hi|hello|hey|greetings)[,!\.\s]+', '', text, flags=re.IGNORECASE)

async def get_groq_reply(user_input: str, system_prompt: str = ""):
    if not GROQ_API_KEY:
        yield "GROQ_API_KEY not set in .env"
        return

    if not system_prompt:
        system_prompt = get_short_system_prompt()

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 50,
        "temperature": 0.4,
        "stream": True
    }

    try:
        async with _async_client.stream("POST", GROQ_API_URL, headers=get_headers(), json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
    except Exception as e:
        yield f"[Error: {e}]"
