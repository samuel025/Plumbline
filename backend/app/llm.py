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

_last_ok = 0.0    # time of the last successful model call
_last_fail = 0.0  # time of the last failed one
_dead: dict = {}  # (key index, model) -> time until which we skip it (daily quota exhausted)

def _gemini_keys() -> list[str]:
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY{i}" for i in range(2, 6)]
    return [os.environ[n] for n in names if os.environ.get(n)]

def _gemini(keys: list[str], system: str, user: str, max_tokens: int) -> dict:
    """Try each key x model. A daily-quota 429 retires that pair for 30 minutes; other 429/503 get short retries."""
    last = None
    for model in GEMINI_MODELS:
        for ki, key in enumerate(keys):
            if _dead.get((ki, model), 0) > time.time():
                continue
            for attempt in range(3):
                r = _post(key, model, system, user, max_tokens)
                if r.status_code == 429 and "PerDay" in r.text:
                    _dead[(ki, model)] = time.time() + 1800
                    break
                if r.status_code not in (429, 503):
                    break
                time.sleep(float(r.headers.get("retry-after", 0)) or 3 * (attempt + 1))
            if r.status_code in (404, 429, 503):
                last = r
                continue
            r.raise_for_status()
            return _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    if last is None:
        raise RuntimeError("all Gemini keys/models are out of daily quota")
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
    gk, ak = _gemini_keys(), os.environ.get("ANTHROPIC_API_KEY")
    if not (gk or ak):
        raise LLMUnavailable("Set GEMINI_API_KEY or ANTHROPIC_API_KEY")
    global _last_ok, _last_fail
    last = None
    for _ in range(2):  # one retry covers a malformed or transient response
        try:
            out = _gemini(gk, system, user, max_tokens) if gk else _anthropic(ak, system, user, max_tokens)
            _last_ok = time.time()
            return out
        except Exception as e:
            last = e
    _last_fail = time.time()
    raise LLMUnavailable(str(last)) from last
