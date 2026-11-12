"""LLM wrapper: Gemini (GEMINI_API_KEY) or Anthropic (ANTHROPIC_API_KEY).
Any failure raises LLMUnavailable; the pipeline then routes the item to a pastor."""
import json, os, re, time
import httpx

# Tried in order; a model that stays rate-limited or unavailable falls through to the next.
GEMINI_MODELS = os.environ.get("PLUMBLINE_GEMINI_MODEL", "gemini-3.5-flash-lite,gemini-3.8-flash,gemini-2.5-flash").split(",")
ANTHROPIC_MODEL = os.environ.get("PLUMBLINE_MODEL", "claude-sonnet-5-5")

def _load_env():
    """Load KEY=value lines from the project's .env (real environment variables win)."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    except FileNotFoundError:
        pass

_load_env()

class LLMUnavailable(Exception):
    pass

def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    if not isinstance(data, dict):
        raise ValueError("model returned JSON that is not an object")
    return data

def _gemini(key: str, system: str, user: str, max_tokens: int) -> dict:
    last = None
    for model in GEMINI_MODELS:
        for attempt in range(3):
            r = _post(key, model, system, user, max_tokens)
            if r.status_code not in (429, 503):
                break
            time.sleep(float(r.headers.get("retry-after", 0)) or min(10 * 2 ** attempt, 40))
        if r.status_code in (404, 429, 503):  # unknown model or still limited: next model
            last = r
            continue
        r.raise_for_status()
        return _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    last.raise_for_status()

def _post(key, model, system, user, max_tokens):
    return httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"systemInstruction": {"parts": [{"text": system}]},
              "contents": [{"role": "user", "parts": [{"text": user}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0,
                                   "maxOutputTokens": max_tokens}},
        timeout=60)

def _anthropic(key: str, system: str, user: str, max_tokens: int) -> dict:
    import anthropic
    resp = anthropic.Anthropic(api_key=key).messages.create(
        model=ANTHROPIC_MODEL, max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}])
    return _parse("".join(b.text for b in resp.content if b.type == "text"))

def call_json(system: str, user: str, max_tokens: int = 4000) -> dict:
    gk, ak = os.environ.get("GEMINI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")
    if not (gk or ak):
        raise LLMUnavailable("Set GEMINI_API_KEY or ANTHROPIC_API_KEY")
    last = None
    for _ in range(2):  # one retry covers a malformed or transient response
        try:
            return _gemini(gk, system, user, max_tokens) if gk else _anthropic(ak, system, user, max_tokens)
        except Exception as e:
            last = e
    raise LLMUnavailable(str(last)) from last
