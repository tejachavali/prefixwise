"""Measure the Gemini API's prompt cache from the API itself (issue #170).

Prefixwise needs Gemini's cache rules before it can judge Gemini's own API: which part of a request
the cached prefix starts with, the smallest prompt that is ever read, how reads are rounded, how
often an unchanged prefix comes back short, how long an idle cache lasts, and what the usage fields
report for an explicit cache and on the OpenAI-compatible endpoint. Nothing here assumes them.

Every call is a plain REST request (no SDK), and every response's raw `usageMetadata` (or `usage`)
is printed and written to the output JSON. Each run starts its prompts with a random nonce, so no
earlier run's cache can serve it.

  python gemini_cache_probe.py out.json --model gemini-2.5-flash [--parts growth,order,minimum,
      short,stream,explicit,openai,idle] [--idle-gaps 60,180,300] [--base URL]

The key comes from GEMINI_API_KEY and is sent as the x-goog-api-key header (never printed). About
60 calls without `idle`; a few cents on a Flash model.

Parts:
  growth    an unchanged conversation growing ~600 tokens a turn for 8 turns: what each turn reads
  stream    the same, streamed (streamGenerateContent?alt=sse): is the read the same?
  order     a large system instruction, large tools and contents; change only the first tool, then
            only the start of the system instruction: where the read stops gives the prefix order
  minimum   prompts of ~1k to ~6k tokens, each sent 3 times --pause apart: the smallest that is read
  short     one ~8k-token request sent 6 times --pause apart: how often an unchanged prefix comes back short
  explicit  a cachedContents entry, then generateContent with cachedContent and new contents
  openai    the OpenAI-compatible endpoint, twice: does usage carry prompt_tokens_details.cached_tokens?
  idle      one conversation with idle gaps (--idle-gaps, seconds): when does the read drop to 0?
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

PAUSE = 2.0  # seconds between calls (--pause)
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone cloud ember maple "
         "harbor lantern meadow orbit pixel quartz saddle timber velvet willow anchor basalt cobalt dune "
         "falcon garnet hazel iris jasper kelp lotus marble nectar onyx pepper quill russet sage thistle").split()


def text(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


class Gemini:
    def __init__(self, base: str, key: str, model: str):
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.log: list[dict] = []

    def _post(self, url: str, body: dict, bearer=False, stream=False):
        headers = {"content-type": "application/json"}
        headers.update({"authorization": "Bearer " + self.key} if bearer else {"x-goog-api-key": self.key})
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read().decode()
                status = r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read().decode(errors="replace")[:600], e.code
        except Exception as e:  # noqa: BLE001 - report and go on
            raw, status = f"{type(e).__name__}: {e}"[:600], 0
        return status, raw, time.time() - t0

    def generate(self, label: str, body: dict, stream=False) -> dict:
        verb = "streamGenerateContent?alt=sse" if stream else "generateContent"
        status, raw, secs = self._post(f"{self.base}/v1beta/models/{self.model}:{verb}", body, stream=stream)
        usage = None
        if status == 200:
            if stream:  # usage rides on the last chunk(s) that carry it
                for line in raw.splitlines():
                    if line.startswith("data:"):
                        try:
                            chunk = json.loads(line[5:])
                        except ValueError:
                            continue
                        usage = chunk.get("usageMetadata") or usage
            else:
                usage = json.loads(raw).get("usageMetadata")
        row = {"part": label, "t": round(time.time(), 1), "status": status, "secs": round(secs, 2),
               "usage": usage, "error": None if status == 200 else raw[:300]}
        self.log.append(row)
        u = usage or {}
        print(f"  {label:34s} {status}  prompt {u.get('promptTokenCount')}  cached {u.get('cachedContentTokenCount')}"
              + (f"  ERROR {row['error'][:160]}" if row["error"] else ""), flush=True)
        return row

    def call(self, label: str, url: str, body: dict, bearer=False) -> dict:
        status, raw, secs = self._post(url, body, bearer=bearer)
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        row = {"part": label, "t": round(time.time(), 1), "status": status, "secs": round(secs, 2),
               "response": {k: v for k, v in (data or {}).items() if k in ("name", "usage", "usageMetadata",
                                                                            "expireTime", "model")} or None,
               "error": None if 200 <= status < 300 else raw[:300]}
        self.log.append(row)
        print(f"  {label:34s} {status}  {json.dumps(row['response'])[:200] if row['response'] else row['error']}",
              flush=True)
        return data or {}


def conversation(nonce: str, system: str, turns: list[str], tools=None) -> dict:
    contents = []
    for i, t in enumerate(turns):
        contents.append({"role": "user", "parts": [{"text": t}]})
        if i < len(turns) - 1:
            contents.append({"role": "model", "parts": [{"text": f"Noted, step {i + 1}."}]})
    body = {"systemInstruction": {"parts": [{"text": f"[{nonce}] " + system}]}, "contents": contents,
            "generationConfig": {"maxOutputTokens": 16}}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
    return body


def tool_decls(rng: random.Random, n=6, words=500) -> list[dict]:
    return [{"name": f"tool_{i}", "description": text(words, rng),
             "parameters": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}}, "required": ["path"]}}
            for i in range(n)]


def growth(g: Gemini, rng, stream=False):
    nonce = uuid.uuid4().hex[:8]
    system, turns = text(4500, rng), []
    for turn in range(8):
        turns.append(text(600, rng))
        g.generate(f"{'stream' if stream else 'growth'} turn {turn + 1}", conversation(nonce, system, turns), stream)
        time.sleep(PAUSE)


def order(g: Gemini, rng):
    nonce = uuid.uuid4().hex[:8]
    system, tools, turns = text(3000, rng), tool_decls(rng), [text(1500, rng)]
    base = conversation(nonce, system, turns, tools)
    for i in range(3):
        g.generate(f"order base {i + 1}", base)
        time.sleep(PAUSE)
    first_tool = json.loads(json.dumps(base))
    first_tool["tools"][0]["functionDeclarations"][0]["description"] = "CHANGED " + \
        first_tool["tools"][0]["functionDeclarations"][0]["description"]
    g.generate("order first tool changed", first_tool)
    time.sleep(PAUSE)
    sys_start = json.loads(json.dumps(base))
    sys_start["systemInstruction"]["parts"][0]["text"] = "CHANGED " + sys_start["systemInstruction"]["parts"][0]["text"]
    g.generate("order system start changed", sys_start)
    time.sleep(PAUSE)
    first_turn = json.loads(json.dumps(base))
    first_turn["contents"][0]["parts"][0]["text"] = "CHANGED " + first_turn["contents"][0]["parts"][0]["text"]
    g.generate("order first user turn changed", first_turn)


def minimum(g: Gemini, rng):
    for words in (800, 1600, 2400, 3200, 4400, 6000):
        nonce = uuid.uuid4().hex[:8]
        body = conversation(nonce, text(words, rng), ["Say ok."])
        for i in range(3):
            g.generate(f"minimum ~{words} words try {i + 1}", body)
            time.sleep(PAUSE)


def short(g: Gemini, rng):
    body = conversation(uuid.uuid4().hex[:8], text(8000, rng), ["Say ok."])
    for i in range(6):
        g.generate(f"short repeat {i + 1}", body)
        time.sleep(PAUSE)


def explicit(g: Gemini, rng):
    nonce = uuid.uuid4().hex[:8]
    system = f"[{nonce}] " + text(5000, rng)
    created = g.call("explicit create cachedContents", f"{g.base}/v1beta/cachedContents",
                     {"model": f"models/{g.model}", "systemInstruction": {"parts": [{"text": system}]},
                      "contents": [{"role": "user", "parts": [{"text": text(1500, rng)}]}], "ttl": "300s"})
    name = created.get("name")
    if not name:
        return
    for i in range(2):
        g.generate(f"explicit with cachedContent {i + 1}",
                   {"cachedContent": name, "contents": [{"role": "user", "parts": [{"text": text(200, rng)}]}],
                    "generationConfig": {"maxOutputTokens": 16}})
        time.sleep(PAUSE)
    try:  # tidy up; storage is billed by the hour
        req = urllib.request.Request(f"{g.base}/v1beta/{name}", method="DELETE", headers={"x-goog-api-key": g.key})
        urllib.request.urlopen(req, timeout=60).read()
        print("  explicit cache deleted", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  explicit cache not deleted: {type(e).__name__}", flush=True)


def openai_compat(g: Gemini, rng):
    nonce = uuid.uuid4().hex[:8]
    body = {"model": g.model, "max_tokens": 16,
            "messages": [{"role": "system", "content": f"[{nonce}] " + text(6000, rng)},
                         {"role": "user", "content": "Say ok."}]}
    for i in range(3):
        g.call(f"openai-compatible {i + 1}", f"{g.base}/v1beta/openai/chat/completions", body, bearer=True)
        time.sleep(PAUSE)


def idle(g: Gemini, rng, gaps):
    nonce = uuid.uuid4().hex[:8]
    system, turns = text(6000, rng), [text(300, rng)]
    g.generate("idle start", conversation(nonce, system, turns))
    for gap in gaps:
        print(f"  ... idle {gap}s", flush=True)
        time.sleep(gap)
        turns.append(text(300, rng))
        g.generate(f"idle after {gap}s", conversation(nonce, system, turns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--parts", default="growth,stream,order,minimum,short,explicit,openai")
    p.add_argument("--idle-gaps", default="60,180,300")
    p.add_argument("--base", default="https://generativelanguage.googleapis.com")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pause", type=float, default=2.0, help="seconds between calls")
    a = p.parse_args()
    global PAUSE
    PAUSE = a.pause
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    g, rng = Gemini(a.base, key, a.model), random.Random(a.seed)
    parts = [x.strip() for x in a.parts.split(",") if x.strip()]
    runs = {"growth": lambda: growth(g, rng), "stream": lambda: growth(g, rng, stream=True),
            "order": lambda: order(g, rng), "minimum": lambda: minimum(g, rng), "short": lambda: short(g, rng),
            "explicit": lambda: explicit(g, rng), "openai": lambda: openai_compat(g, rng),
            "idle": lambda: idle(g, rng, [int(x) for x in a.idle_gaps.split(",")])}
    unknown = [x for x in parts if x not in runs]
    if unknown:
        sys.exit(f"unknown parts: {unknown}; choose from {sorted(runs)}")
    print(f"model {a.model}, parts {parts}", flush=True)
    for part in parts:
        print(f"== {part}", flush=True)
        runs[part]()
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "parts": parts, "calls": g.log}, f, indent=1)
    print(f"{len(g.log)} calls written to {a.out}")


if __name__ == "__main__":
    main()
