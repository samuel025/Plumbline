"""LLM wrapper: Gemini (GEMINI_API_KEY) or Anthropic (ANTHROPIC_API_KEY).
Any failure raises LLMUnavailable; the pipeline then routes the item to a pastor."""
import json, logging, os, re, time
import httpx

log = logging.getLogger("plumbline.llm")

# A stalled free-tier request is almost always stuck for good (observed: 46s spikes, normal is 1-3s),
# so give up quickly and retry instead of waiting a long time once.
TIMEOUT = float(os.environ.get("PLUMBLINE_LLM_TIMEOUT", "20"))
ATTEMPTS = 3

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
    """Take the FIRST complete JSON object in the text, ignoring any chatter or a second object after it."""
    i = text.find("{")
    if i < 0:
        raise ValueError("model returned no JSON object")
    data, _ = json.JSONDecoder().raw_decode(text[i:])
    if not isinstance(data, dict):
        raise ValueError("model returned JSON that is not an object")
    return data

_last_ok = 0.0    # time of the last successful model call
_last_fail = 0.0  # time of the last failed one
_dead: dict = {}       # (key index, model) -> time until which we skip it (daily quota exhausted)
_dead_note: dict = {}  # (key index, model) -> "resets in 9h43m" (from the API's own message)
_last_trail: list = []  # what the most recent model call tried, per key and model

def _gemini_keys() -> list[str]:
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY{i}" for i in range(2, 6)]
    return [os.environ[n] for n in names if os.environ.get(n)]

def _label(r) -> str:
    t = r.text
    if r.status_code == 200: return "ok"
    if r.status_code == 429: return "daily limit reached" if "PerDay" in t else "rate limited"
    if r.status_code == 503: return "overloaded"
    if r.status_code == 404: return "model not available"
    return f"error {r.status_code}"

def key_states() -> list[dict]:
    """Per-key status for the health badge, from what recent calls learned."""
    out = []
    for ki in range(len(_gemini_keys())):
        dead = [(m, _dead_note.get((ki, m), "")) for m in GEMINI_MODELS if _dead.get((ki, m), 0) > time.time()]
        out.append({"key": ki + 1, "state": "daily limit reached" if dead else "available",
                    "detail": dead[0][1] if dead else ""})
    return out

def _gemini(keys: list[str], system: str, user: str, max_tokens: int) -> dict:
    """Try each model x key. A daily-quota 429 retires that pair for 30 minutes; other 429/503 get short retries."""
    global _last_trail
    trail, last = [], None
    for model in GEMINI_MODELS:
        for ki, key in enumerate(keys):
            if _dead.get((ki, model), 0) > time.time():
                trail.append(f"key {ki+1} · {model}: skipped (daily limit reached earlier)")
                continue
            for attempt in range(3):
                r = _post(key, model, system, user, max_tokens)
                if r.status_code == 429 and "PerDay" in r.text:
                    _dead[(ki, model)] = time.time() + 1800
                    m = re.search(r"retry in ([0-9hms.]+)", r.text)
                    _dead_note[(ki, model)] = ("resets in " + re.sub(r"\.\d+s", "s", m.group(1))) if m else ""
                    break
                if r.status_code not in (429, 503):
                    break
                time.sleep(float(r.headers.get("retry-after", 0)) or 3 * (attempt + 1))
            trail.append(f"key {ki+1} · {model}: {_label(r)}")
            if r.status_code in (404, 429, 503):
                last = r
                continue
            r.raise_for_status()
            _last_trail = trail
            return _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    _last_trail = trail
    raise RuntimeError("KEYS: " + "; ".join(trail))

def _post(key, model, system, user, max_tokens):
    return httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"systemInstruction": {"parts": [{"text": system}]},
              "contents": [{"role": "user", "parts": [{"text": user}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0,
                                   "maxOutputTokens": max_tokens}},
        timeout=TIMEOUT)

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
    for _ in range(ATTEMPTS):  # retries cover a stalled, malformed or transient response
        try:
            out = _gemini(gk, system, user, max_tokens) if gk else _anthropic(ak, system, user, max_tokens)
            _last_ok = time.time()
            return out
        except Exception as e:
            last = e
            log.warning("model call failed: %s: %s", type(e).__name__, str(e)[:300])
    _last_fail = time.time()
    # include the exception class: some (e.g. httpx timeouts) have an empty message
    raise LLMUnavailable(f"{type(last).__name__}: {last}") from last
