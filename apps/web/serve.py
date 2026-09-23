"""A single-process web front end for the measured champion topology.

The repository could already serve the champion -- but only as two processes
(a neuro-san server plus a separate web client from another repository), on a
developer's machine, reachable by nobody. That is not "online", and the winner
of a search that nobody can talk to is still just a hash in a report.

This is one process with no external client: FastAPI serves a page, the page
posts a question, and the question runs through the champion via neuro-san's
direct session -- the same code path the evaluator measures with. What a
visitor talks to is the measured topology with one line changed: the front man
explains its answer instead of returning the bare value the scorer needs. If the
deployment holds a key for a different provider from the one the champion was
measured on, the models move rung for rung and the page says so
(`esp/serving.py`).

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

import html  # noqa: E402

from esp.config import key_name_for, key_problem  # noqa: E402
from esp.eval import measurements  # noqa: E402
from esp.eval.runner import _ask, write_network  # noqa: E402
from esp.eval.tasks import TASKS, score  # noqa: E402
from esp.genome.definition import Genome  # noqa: E402
from esp.genome.seeds import SEEDS  # noqa: E402
from esp.service.state import Evaluated, ServiceState  # noqa: E402
from esp.serving import SHOWCASE, display_question, graded, presentable  # noqa: E402
from esp.surrogate.predictor import MIN_SAMPLES  # noqa: E402

# A visitor is not a benchmark run, and every question is paid for: a multi-hop
# answer is ten or more model calls. A public deployment caps how many it will
# answer before it stops, so one careless loop cannot run up the key's bill.
MAX_QUESTIONS = int(os.environ.get("ESP_WEB_MAX_QUESTIONS", "40"))
_asked = {"count": 0}


def champion():
    """The measured best, or the designer-shaped seed when nothing is measured.

    Priority: this deployment's own `state/champion.json`, then its service
    population, then the evaluation cache committed with the repository, then
    the designer-shaped seed. Falling back rather than failing is deliberate: a
    fresh deployment with no state should still answer, and the page says which
    case it is in.

    The committed cache was missing from that list, and the omission was not
    cosmetic. A fresh clone has no state, so every deployment anybody has ever
    started from one served `designer_shaped` at +0.7761 -- the *worst* of the
    eleven measured networks -- under a page headed "measured champion", while
    the network that actually won at +0.8453 sat in the repository with its
    genome beside its score.
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

    committed = measurements.best()
    if committed is not None:
        return committed.name(), committed.genome, Evaluated(
            genome_hash=committed.genome_hash, origin=committed.origin,
            fitness=committed.fitness, accuracy=committed.accuracy,
            tokens=committed.tokens, agents=committed.agents,
            depth=committed.depth, generation=0, measured_at="",
            model=committed.genome.default_model,
            genome=committed.genome.canonical())

    return "designer_shaped", SEEDS["designer_shaped"](), None


NAME, GENOME, RECORD = champion()
SERVED = presentable(GENOME)
HOCON = str(write_network(SERVED.genome))

# Declared at module scope, and that is not a style choice. This file uses
# `from __future__ import annotations`, so every annotation is a string that
# FastAPI resolves against module globals -- a request model defined inside
# build_app() is invisible there, and FastAPI silently reclassifies the
# parameter as a query field. The symptom is a 422 saying the body field is a
# missing query parameter, which points nowhere near the cause.
class Question(BaseModel):
    question: str


PAGE = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<link rel=icon href="data:,">
<title>neuro-san-esp | measured champion</title>
<style>
 :root{color-scheme:light dark;--bg:#fbfcfd;--fg:#101418;--mute:#5b6672;--line:#d7dde3;
       --accent:#1a4fa0;--code:#12161b;--codefg:#d6e2ee;--warn:#fff6e5;--warnline:#e8d9b0;
       --ok:#1f7a3a;--bad:#a4262c;--onaccent:#fff}
 @media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e9ee;--mute:#98a2ae;
       --line:#2a2f38;--accent:#6ea0ff;--warn:#2a2313;--warnline:#4a3f22;--ok:#5cc47a;--bad:#ff7b72;--onaccent:#0f1115}}
 body{font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      max-width:780px;margin:0 auto;padding:32px 16px;background:var(--bg);color:var(--fg)}
 h1{font-size:22px;margin:0 0 4px} .sub{color:var(--mute);margin:0 0 18px;font-size:14px}
 .card{border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
 textarea{width:100%;box-sizing:border-box;padding:10px;font:inherit;border-radius:8px;
          border:1px solid var(--line);background:transparent;color:inherit}
 button{padding:9px 18px;border:0;border-radius:8px;background:var(--accent);color:var(--onaccent);
        font:inherit;cursor:pointer} button[disabled]{opacity:.5;cursor:not-allowed}
 .row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
 .ex{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin-top:12px}
 .ex button{background:transparent;color:inherit;border:1px solid var(--line);text-align:left;
            font-size:14px;line-height:1.4;padding:10px}
 .ex b{display:block;font-size:12px;color:var(--mute);font-weight:600;margin-bottom:2px}
 #answer{white-space:pre-wrap;line-height:1.65}
 .meta{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute);margin-top:12px}
 .ok{color:var(--ok);font-weight:600} .bad{color:var(--bad);font-weight:600}
 .warn{background:var(--warn);border-color:var(--warnline);font-size:14px}
 .served{font-size:13px;color:var(--mute)}
</style>
<h1>Talking to a measured agent network</h1>
<p class=sub>__SUB__</p>
<p class=served>__SERVED__</p>

<div class=card>
<label for=q class=sub>Ask anything about Meridian Logistics &mdash; or start with one of these.
Each has a known correct answer, and the page grades it.</label>
<div class=ex>__EXAMPLES__</div>
<textarea id=q rows=3 style="margin-top:12px"
 placeholder="e.g. Which depot services contract C-2117, and who manages it?"></textarea>
<div class=row><button id=go onclick=ask()>Ask</button>
<span id=status class=sub style="margin:0"></span></div>
</div>

<div class=card id=result hidden>
<div id=answer></div>
<div class=meta id=meta></div>
</div>

<div class="card warn">
<b>What this is.</b> Meridian Logistics is invented &mdash; 24 depots, 40 contracts, 124 documents.
Nothing about it is in any model's training data, so the only way to answer is to go and read,
which is what makes the score a measurement of the <i>topology</i> rather than of recall.
<br><br>
<b>What it is not.</b> __CAVEAT__
</div>
<p class=sub>Source: <a href="https://github.com/Sivakumarraj/neuro-san-esp">github.com/Sivakumarraj/neuro-san-esp</a></p>
<script>
function pick(b){document.getElementById('q').value=b.dataset.q;ask()}
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const go=document.getElementById('go'), st=document.getElementById('status');
  const box=document.getElementById('result'), ans=document.getElementById('answer'),
        meta=document.getElementById('meta');
  go.disabled=true; box.hidden=true;
  st.textContent='Running the network — a multi-hop question takes a minute or two.';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
                               body:JSON.stringify({question:q})});
    const d=await r.json();
    box.hidden=false; meta.textContent='';
    if(d.error){ans.textContent='Error: '+d.error;return}
    ans.textContent=d.answer||'(the network returned no answer)';
    const line=document.createElement('div');
    line.textContent=`${d.agents} agents · ${d.seconds}s · ${d.provider}`
      +` · router ${d.router_model}, workers ${d.worker_model}`;
    meta.appendChild(line);
    if(d.expected){
      const g=document.createElement('div');
      g.append('Benchmark answer: '+d.expected+' — ');
      const v=document.createElement('span');
      v.className=d.correct?'ok':'bad'; v.textContent=d.correct?'CORRECT':'WRONG';
      g.appendChild(v); meta.appendChild(g);
    }
  }catch(e){box.hidden=false;ans.textContent='Error: '+e}
  finally{go.disabled=false;st.textContent=''}
}
</script></html>"""


def measurement_count() -> int:
    """How many real evaluations this deployment can see.

    Counted through the same reader the champion comes from, so the page cannot
    report a population it did not actually resolve -- a cache file whose
    genome will not rebuild is not a measurement this deployment can see.
    """
    return len(measurements.load())


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
    labels = ("Direct lookup", "Two documents", "Deepest chain", "Hop, then calculate")
    examples = "".join(
        f'<button data-q="{html.escape(display_question(task), quote=True)}" '
        f'onclick="pick(this)"><b>{label} &middot; {task.hops} hop'
        f'{"s" if task.hops != 1 else ""}</b>{html.escape(display_question(task))}</button>'
        for label, task in zip(labels, SHOWCASE, strict=False))
    served = html.escape(SERVED.note(GENOME.default_model))
    return (PAGE.replace("__SUB__", sub)
                .replace("__SERVED__", served)
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
                {"error": f"this deployment has answered its limit of "
                          f"{MAX_QUESTIONS} questions -- every answer is paid "
                          f"for, so a public page stops rather than running up "
                          f"the key's bill. ESP_WEB_MAX_QUESTIONS raises it."},
                status_code=429)
        question = payload.question.strip()[:1000]
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)

        _asked["count"] += 1
        started = time.monotonic()
        try:
            answer, _, seconds = _ask(HOCON, question)
        except Exception as exc:      # a provider 429 must not 500 the page
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"[:300]}, status_code=502)

        # If it is one of the benchmark questions, grade it in front of the
        # visitor. Claiming correctness without showing the expected answer
        # would be the same as not claiming it. Matched as displayed, because
        # the page shows benchmark questions without their "number only"
        # suffix -- the network is now asked to explain, not to be terse.
        task = graded(question)
        expected = task.answer if task else None
        return JSONResponse({
            # Never truncated. The old 2,000-character cut existed to protect a
            # one-word benchmark answer from nothing, and it clipped exactly the
            # explanation a person came to read.
            "answer": answer,
            "agents": len(SERVED.genome.reachable()),
            "seconds": round(seconds or (time.monotonic() - started), 1),
            "provider": SERVED.provider,
            "router_model": SERVED.genome.agents[SERVED.genome.top].model
                            or SERVED.genome.default_model,
            "worker_model": SERVED.genome.default_model,
            "expected": expected,
            "correct": score(expected, answer) if expected else None,
        })

    return app


def main() -> int:
    # The key this deployment needs is the one for the provider it serves on,
    # which is the configured one -- not the one the champion was measured on.
    # This used to demand GOOGLE_API_KEY unconditionally, so a deployment
    # holding only a Claude key refused to start at all.
    wanted = key_name_for(SERVED.genome.default_model)
    problem = key_problem(os.environ.get(wanted or "", ""), wanted or "")
    if wanted is None or problem:
        print(f"{wanted or 'no provider key'} is {problem or 'unknown'} -- the "
              "page would load and every question would fail. Put the key in "
              ".env; see .env.example.", file=sys.stderr)
        return 1
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"champion={NAME} measured={RECORD is not None} "
          f"provider={SERVED.provider} retargeted={SERVED.retargeted} port={port}")
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
