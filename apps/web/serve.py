"""A single-process web front end for the measured champion topology.

The repository could already serve the champion -- but only as two processes
(a neuro-san server plus a separate web client from another repository), on a
developer's machine, reachable by nobody. That is not "online", and the winner
of a search that nobody can talk to is still just a hash in a report.

This is one process with no external client: FastAPI serves a page, the page
posts a question, and the question runs through the champion genome via
neuro-san's direct session -- the same code path the evaluator measures with,
so what a visitor talks to is exactly what was scored.

It binds 7860 because that is what Hugging Face Spaces expects, and Spaces is
the cheapest way to put this behind a URL somebody can click. It works the same
under `docker run -p 7860:7860` or bare `python apps/web/serve.py`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from esp.config import bootstrap  # noqa: E402

# Load .env BEFORE anything downstream reads env at import time. The rate
# limiter builds its keyring at module load from GOOGLE_API_KEYS; leaving
# bootstrap for main() froze that keyring empty for anyone who only set
# their keys in .env, so the first per-day 429 stopped the page instead
# of rotating.
bootstrap()

from esp.eval.runner import _ask, write_network  # noqa: E402
from esp.eval.tasks import TASKS, score  # noqa: E402
from esp.genome.definition import Genome  # noqa: E402
from esp.genome.seeds import SEEDS  # noqa: E402
from esp.service.state import Evaluated, ServiceState  # noqa: E402
from esp.surrogate.predictor import MIN_SAMPLES  # noqa: E402

# A visitor is not a benchmark run. One question at a time, and a hard cap on
# how many a single deployment will answer before it stops -- the free tier is
# 500 requests a day and one careless loop would spend all of it.
MAX_QUESTIONS = int(os.environ.get("ESP_WEB_MAX_QUESTIONS", "40"))
_asked = {"count": 0}


def champion():
    """The measured best, or the designer-shaped seed when nothing is measured.

    Priority: state/champion.json (evolved mutant winner with full genome +
    measurement), then ServiceState (seed matches only), then the designer-shaped
    seed as last-resort fallback. Falling back rather than failing is deliberate:
    a fresh deployment with no state should still answer, and the page says
    which case it is in.
    """
    state_dir = Path(os.environ.get("ESP_STATE", ROOT / "state"))
    manifest = state_dir / "champion.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        genome = Genome.from_canonical(payload["genome"])
        measured = payload["measured"]
        record = Evaluated(
            genome_hash=payload["hash"], origin=payload.get("origin", "measured"),
            fitness=measured["fitness"], accuracy=measured["accuracy"],
            tokens=measured["tokens"], agents=measured["agents"],
            depth=len(genome.reachable()), generation=0, measured_at="",
            model=genome.default_model, genome=payload["genome"])
        return payload.get("origin", "measured"), genome, record

    state = ServiceState.load(state_dir)
    best = state.best()
    if best is not None:
        for name, build in SEEDS.items():
            genome = build()
            if genome.genome_hash() == best.genome_hash:
                return name, genome, best
    return "designer_shaped", SEEDS["designer_shaped"](), None


NAME, GENOME, RECORD = champion()
HOCON = str(write_network(GENOME))

# Declared at module scope, and that is not a style choice. This file uses
# `from __future__ import annotations`, so every annotation is a string that
# FastAPI resolves against module globals -- a request model defined inside
# build_app() is invisible there, and FastAPI silently reclassifies the
# parameter as a query field. The symptom is a 422 saying the body field is a
# missing query parameter, which points nowhere near the cause.
class Question(BaseModel):
    question: str


PAGE = """<!doctype html><meta charset=utf-8>
<title>neuro-san-esp | measured champion</title>
<style>
 :root{color-scheme:light dark}
 body{font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      max-width:760px;margin:0 auto;padding:32px 20px;background:#fbfcfd;color:#101418}
 @media(prefers-color-scheme:dark){body{background:#0f1115;color:#e6e9ee}}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#5b6672;margin:0 0 22px;font-size:14px}
 .card{border:1px solid #d7dde3;border-radius:10px;padding:16px;margin:16px 0}
 @media(prefers-color-scheme:dark){.card{border-color:#2a2f38}}
 textarea{width:100%;box-sizing:border-box;padding:10px;font:inherit;border-radius:8px;
          border:1px solid #c6ced6;background:transparent;color:inherit}
 button{margin-top:10px;padding:9px 18px;border:0;border-radius:8px;background:#1a4fa0;
        color:#fff;font:inherit;cursor:pointer}
 button[disabled]{opacity:.5;cursor:not-allowed}
 pre{white-space:pre-wrap;background:#12161b;color:#d6e2ee;padding:12px;border-radius:8px;
     font:13px/1.5 ui-monospace,monospace;overflow-x:auto}
 .ex{font-size:13px;color:#5b6672} .ex a{color:#1a4fa0;cursor:pointer;display:block;margin:3px 0}
 .warn{background:#fff6e5;border-color:#e8d9b0;font-size:14px}
 @media(prefers-color-scheme:dark){.warn{background:#2a2313;border-color:#4a3f22}}
</style>
<h1>Talking to a measured agent network</h1>
<p class=sub>__SUB__</p>

<div class=card>
<textarea id=q rows=3 placeholder="Ask about Meridian Logistics..."></textarea>
<button id=go onclick=ask()>Ask</button>
<div class=ex style="margin-top:10px">Try one of these — each has a known correct answer:
__EXAMPLES__
</div>
</div>
<pre id=out>Ready.</pre>

<div class="card warn">
<b>What this is.</b> Meridian Logistics is invented — 24 depots, 40 contracts, 124 documents.
Nothing about it is in any model's training data, so the only way to answer is to go and read,
which is what makes the score a measurement of the <i>topology</i> rather than of recall.
<br><br>
<b>What it is not.</b> __CAVEAT__
</div>
<p class=sub>Source: <a href="https://github.com/Sivakumarraj/neuro-san-esp">github.com/Sivakumarraj/neuro-san-esp</a></p>
<script>
function fill(t){document.getElementById('q').value=t}
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const b=document.getElementById('go'), o=document.getElementById('out');
  b.disabled=true; o.textContent='Running the network... multi-hop questions take a minute.';
  const t0=Date.now();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
                               body:JSON.stringify({question:q})});
    const d=await r.json();
    o.textContent = d.error ? ('error: '+d.error)
      : `answer:   ${d.answer}\\nagents:   ${d.agents}\\nseconds:  ${d.seconds}`
        + (d.expected ? `\\nexpected: ${d.expected}   ->  ${d.correct?'CORRECT':'WRONG'}` : '');
  }catch(e){o.textContent='error: '+e}
  finally{b.disabled=false}
}
</script>"""


def measurement_count() -> int:
    """How many real evaluations this deployment can see."""
    cache = ROOT / "tests" / "fixtures" / "cache"
    return len(list(cache.glob("*.json"))) if cache.is_dir() else 0


def surrogate_quality() -> dict | None:
    """The most recent cross-validated quality report, if a run produced one.

    Read from the run history rather than recomputed, so the page reports the
    predictor that was actually used rather than one fitted here for display.
    """
    history = ROOT / "results" / "history.json"
    if not history.exists():
        return None
    try:
        entries = json.loads(history.read_text(encoding="utf-8")).get(
            "surrogate_quality") or []
    except (json.JSONDecodeError, OSError):
        return None
    return entries[-1] if entries else None


def caveat() -> str:
    """What this deployment has *not* shown, derived rather than written down.

    The previous version was prose fixed in the template, and it went stale the
    moment an evolved candidate won: the page claimed no mutant had beaten the
    baselines directly beneath a header naming one as the best-measured
    topology. A caveat that contradicts the page above it is worse than none.
    """
    parts: list[str] = []
    evolved = RECORD is not None and RECORD.origin.startswith("mut:")
    if not evolved:
        parts.append("no evolved candidate has beaten the hand-designed "
                     "baselines yet")

    measured = measurement_count()
    if measured < MIN_SAMPLES:
        parts.append(f"the surrogate has not trained &mdash; it needs "
                     f"{MIN_SAMPLES} measurements and there are {measured}, so "
                     f"the search is still exploring rather than ranking")
    else:
        # Having enough samples to train is not the same as training usefully.
        # Reporting only the count would let the page imply a working predictor
        # on the strength of a threshold it merely cleared.
        quality = surrogate_quality()
        if quality is not None and not quality.get("beats_random", False):
            parts.append(
                f"the surrogate trained on {quality.get('samples', measured)} "
                f"samples and did not rank better than chance "
                f"(Spearman {quality.get('spearman', float('nan')):+.3f}), so "
                f"the free search explores rather than selects")

    if not parts:
        parts.append("the search has run few generations, so the Pareto front "
                     "is thin")
    body = "; ".join(parts)
    return (body[0].upper() + body[1:] +
            ". Stated in the repository rather than left out.")


def page() -> str:
    if RECORD is not None:
        sub = (f"<b>{NAME}</b> &mdash; the best-measured topology: "
               f"{RECORD.accuracy:.0%} correct on {len(TASKS)} multi-hop questions, "
               f"{RECORD.tokens:,} tokens, {RECORD.agents} agents. "
               "Chosen by measurement, not by guessing.")
    else:
        sub = (f"<b>{NAME}</b> &mdash; the shape neuro-san's own designer produces. "
               "No measurements are loaded in this deployment, so this is the "
               "baseline rather than a measured winner.")
    examples = "".join(
        f'<a onclick="fill(this.textContent)">{t.question}</a>'
        for t in (TASKS[0], TASKS[1], TASKS[10]))
    return (PAGE.replace("__SUB__", sub)
                .replace("__EXAMPLES__", examples)
                .replace("__CAVEAT__", caveat()))


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="neuro-san-esp champion")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "champion": NAME,
                "measured": RECORD is not None,
                "asked": _asked["count"], "cap": MAX_QUESTIONS}

    @app.post("/ask")
    def ask(payload: Question) -> JSONResponse:
        if _asked["count"] >= MAX_QUESTIONS:
            return JSONResponse(
                {"error": "this deployment has answered its question limit for "
                          "now -- the free provider tier is 500 requests a day "
                          "and one candidate costs about 165"}, status_code=429)
        question = payload.question.strip()[:500]
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)

        _asked["count"] += 1
        started = time.monotonic()
        try:
            answer, _, seconds = _ask(HOCON, question)
        except Exception as exc:      # a provider 429 must not 500 the page
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"[:300]}, status_code=502)

        # If it happens to be one of the graded questions, grade it in front of
        # the visitor. Claiming correctness without showing the expected answer
        # would be the same as not claiming it.
        expected = next((t.answer for t in TASKS
                         if t.question.strip() == question), None)
        return JSONResponse({
            "answer": answer[:2000],
            "agents": len(GENOME.reachable()),
            "seconds": round(seconds or (time.monotonic() - started), 1),
            "expected": expected,
            "correct": score(expected, answer) if expected else None,
        })

    return app


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set -- the page would load and every "
              "question would fail", file=sys.stderr)
        return 1
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"champion={NAME} measured={RECORD is not None} port={port}")
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
