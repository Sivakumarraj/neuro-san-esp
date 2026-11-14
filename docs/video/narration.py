"""The film's narration, and the timeline the picture is cut to.

    python3 narration.py OUT.wav TIMELINE.js

Every scene's lines are synthesised on this machine with Kokoro, and their measured lengths
set the film's timing: a scene lasts as long as its lines need, plus a breath either side.
TIMELINE.js tells film.html when each scene starts and when each line begins, so text and
pictures land on the words. OUT.wav is the voice alone, aligned to that timeline.

Needs `pip install kokoro-onnx soundfile` and Kokoro's two model files, kokoro-v1.0.onnx
and voices-v1.0.bin, from the kokoro-onnx releases on GitHub, in models/ beside this file or
at the paths in KOKORO_MODEL and KOKORO_VOICES. Lines already spoken are kept in cache/.

Every figure here is from the README, docs/RESULTS.md, docs/FINDINGS.md, docs/DESIGN.md,
SERVING.md or SECURITY.md. Numbers are written as they should be spoken.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
VOICE = "af_heart"
SPEED = 1.08
LEAD = 1.0        # seconds from a scene's cut to its first word
GAP = 0.5         # between lines
TAIL = 1.15       # after a scene's last word, before its cut

# (scene id, chapter, shortest it may be in seconds, [lines])
SCENES = [
    ("title", 0, 11, [
        "This is neuro san E.S.P.: a fitness function for neuro san agent networks, "
        "and an evolutionary search that uses it.",
        "It is reviewed here the way an author of E.S.P. would, with every figure from the "
        "repository.",
    ]),
    ("lens", 0, 12, [
        "E.S.P., evolutionary surrogate-assisted prescription, comes from Cognizant A.I. Lab.",
        "It learns to predict outcomes, and searches for the actions those predictions favour.",
        "Its authors would ask five questions of a system like this.",
    ]),
    ("gap", 0, 11, [
        "neuro san can design a multi-agent network from one sentence, but it never measures "
        "one.",
        "Nothing scores one network against another.",
        "This project adds the missing half: measure, search, and serve.",
    ]),
    ("which", 0, 11, [
        "So which E.S.P. is this? neuro san already ships an E.S.P. decision assistant, whose "
        "agents help a person decide.",
        "There, E.S.P. is what the network does. Here, E.S.P. designs the network.",
    ]),
    ("q1", 1, 12, [
        "Question one. What is optimised?",
        "The context is a question with a checkable answer.",
        "The action is a network, with a model for each agent.",
        "The outcomes, accuracy, tokens and dollars, feed a fixed fitness formula.",
    ]),
    ("genome", 1, 12, [
        "A candidate is neuro san's own HOCON, plus a model per agent, so whatever the search "
        "finds, neuro san can serve.",
        "The model is part of the genome's hash, so scores never cross models.",
        "And order matters: neuro san enters through the first agent listed. Written "
        "alphabetically, one seed put its arithmetic agent in front, and scored zero.",
    ]),
    ("world", 1, 12, [
        "Every question is about Meridian Logistics: a hundred and twenty-four invented "
        "documents on depots, contracts and incidents.",
        "Answers are correct by construction, and the only tool is a deterministic search, so "
        "a score measures the network, not a drifting retriever.",
    ]),
    ("sets", 1, 11, [
        "Networks are chosen on sixty select questions, and judged on two hundred from the "
        "other half of the company.",
        "Even always answering not stated scores a quarter of the newest hundred. That is "
        "chance level.",
    ]),
    ("loop", 2, 14, [
        "Question two starts with the search.",
        "Phase A measures a few seed networks for real.",
        "Phase B trains one predictor per outcome.",
        "Phase C breeds mutants, and ranks thousands with no model calls.",
        "Phase D measures only the most promising, and feeds them back.",
    ]),
    ("operators", 2, 16, [
        "Seven operators make the mutants.",
        "Add an agent, remove one, rewire, split, merge, switch an agent's search on or off, "
        "or reassign its model.",
        "A broken mutant, with a cycle or an unreachable agent, is thrown away, never "
        "repaired.",
    ]),
    ("predictor", 2, 11, [
        "The predictor is one gradient-boosted regressor per outcome, on thirteen structural "
        "features of the network.",
        "It never sees a measured quantity, and below eight measurements it refuses to fit.",
    ]),
    ("split", 2, 13, [
        "Asked in review what the surrogate predicts, the first version had a weak answer: "
        "one model, trained on the combined fitness.",
        "Now each outcome has its own predictor, and fitness is derived.",
        "The split exposed a hidden defect: token cost is ranked backwards in all twenty "
        "seeds. So it is left out of the ranking, and published, not patched.",
    ]),
    ("pareto", 2, 11, [
        "Parents come from the Pareto front: more accurate, cheaper per right answer, and "
        "smaller.",
        "The front was once plotted and never used. Now it breeds.",
    ]),
    ("holdout", 3, 12, [
        "Question three. Does it hold on questions it never chose on?",
        "The first search measured twelve networks on seventeen questions, and looked like a "
        "win. But across two hundred half splits, the winner on one half never came first on "
        "the other.",
        "Seventeen questions cannot rank networks. Every claim since is held out.",
    ]),
    ("bank", 3, 11, [
        "On twenty-four fresh questions, the evolved network answered two fewer than the "
        "designer's shape.",
        "And in the first paid run, a stronger router bought nothing: sixty-eight against "
        "sixty-eight, at twice the cost.",
    ]),
    ("stages", 3, 13, [
        "The paid runs moved everything to OpenAI: nano workers, and one promoted agent on "
        "mini.",
        "A hand-written seed called flat beat the designer's shape, a hundred and fifty-five "
        "to a hundred and thirty-five.",
        "One mutation of it, two C C four, without its arithmetic agent, led again, just short "
        "of significance.",
    ]),
    ("cap", 3, 11, [
        "Then the misses pointed at the benchmark: the search returned three documents, and "
        "mis-picked appears in eight.",
        "At ten documents, the designer's shape rose from sixty-nine to eighty-four percent. "
        "Three and ten are never compared.",
    ]),
    ("placement", 3, 11, [
        "Most remaining misses were arithmetic. Promoting the researcher, the agent that reads "
        "and adds, took two C C four from fifty-two to fifty-seven of sixty.",
        "Where the model goes matters more than which model.",
    ]),
    ("q3", 3, 14, [
        "The headline. On a hundred judge questions, asked twice, two C C four answered "
        "ninety-four and a half percent; the designer's shape, eighty-four.",
        "Seventeen questions to two: p equals zero point zero zero zero seven.",
        "Thirty-seven percent fewer tokens, half the agents, and the predictor was not used "
        "in any paid run.",
    ]),
    ("rep", 3, 12, [
        "It replicated on forty fresh questions, against a control with every specialist "
        "promoted.",
        "Ninety percent, against seventy-seven and a half, and seventy-five.",
        "Nine to one against the designer. Seven to one against the control, not yet "
        "significant.",
    ]),
    ("cost", 4, 11, [
        "Question four. What does a right answer cost?",
        "Its reader runs a model three point seven times dearer per token, so despite fewer "
        "tokens, a correct answer costs seventy-two percent more.",
    ]),
    ("route", 4, 15, [
        "Here E.S.P. finds a context: the question is the context, the network the action, "
        "and answers and dollars the outcomes.",
        "A router sends a question to the strong network only when its text predicts a gain: "
        "ninety-three percent, for twenty-nine percent less. At random, eighty-eight.",
        "But it does not beat a one-line rule, which keeps ninety-four and a half percent for "
        "fourteen percent less.",
    ]),
    ("agree", 4, 11, [
        "Every question was asked twice. When the designer's runs agree, they are right "
        "ninety-six percent of the time; when they disagree, twenty-eight.",
        "A good flag. But as a gate, it loses to the rule.",
    ]),
    ("autopsy", 4, 11, [
        "One kind of question, multi-incident penalties, holds thirty of the designer's "
        "fifty-two wrong runs.",
        "Not retrieval, arithmetic: both confident misses subtracted the hourly rates, and "
        "never multiplied by the hours.",
    ]),
    ("defects", 5, 15, [
        "Most of the work was finding what made good networks look bad.",
        "Timeouts scored as wrong answers. A rate limiter that froze every agent at once. A "
        "cost term that clipped every network to one penalty.",
        "Incident four four two nine tripping the quota guard. A grader that gave marks for "
        "listing numbers.",
        "And every token figure before the tenth of October, counted twice.",
    ]),
    ("audit", 5, 10, [
        "Before any money was spent, every command was rerun from a clean install, and every "
        "committed number reproduced.",
        "Four statements stronger than their evidence were corrected.",
    ]),
    ("q5", 5, 15, [
        "Question five. Can it run unattended?",
        "neuro san's own server fires the optimizer on a schedule. The model never chooses or "
        "spends, and most wakes say nothing.",
        "A preflight checks the key with the provider, a lease stops overlaps, state is saved "
        "after every candidate, and every paid run stops at its cap, and resumes.",
    ]),
    ("providers", 5, 10, [
        "It runs on Anthropic, OpenAI or Gemini, by whichever key is present.",
        "Measurements never cross providers: every hash changes, and the search restarts from "
        "the seeds.",
    ]),
    ("ci", 5, 11, [
        "Every push runs pinned CI: a dependency audit, linters, neuro san's HOCON validator, "
        "and eighty-five percent coverage.",
        "A test recomputes every README figure from the logs, so a claim that drifts fails "
        "the build.",
    ]),
    ("serve", 5, 12, [
        "The winner is served as a web page, in neuro san's accelerator U.I., as a registry, "
        "and on Hugging Face.",
        "The page caps spending, can require a token, tags every request, and never accepts "
        "an uploaded network.",
    ]),
    ("timeline", 0, 11, [
        "Forty-six commits, from the twenty-fourth of August to the twenty-ninth of October.",
        "The search in August, held-out judging and the audit in September, paid runs and "
        "routing in October.",
    ]),
    ("limits", 0, 11, [
        "Where is this not E.S.P.? There is no learned prescriptor: seven operators and "
        "Pareto selection, closer to LEAF.",
        "And the paid result does not test the predictor.",
    ]),
    ("prior", 0, 11, [
        "Searching agent designs is not new: A.D.A.S., A-Flow, G.P.T.-Swarm, and Agent Square, "
        "which already uses a predictor.",
        "What is new is narrower: a measured fitness function for neuro san, run on a budget, "
        "and the defects only running it could find.",
    ]),
    ("next", 0, 11, [
        "Next is already built: a same-budget experiment, predictor against random choice, "
        "reported whichever way it falls.",
        "And a real prescriptor starts by deciding what the context is.",
    ]),
    ("numbers", 0, 11, [
        "Forty-six commits, over eighteen thousand lines of Python, seven hundred and "
        "eighty-four test functions.",
        "A hundred and eighty-nine of two hundred held-out answers, for three dollars "
        "seventy-three across nine paid stages.",
    ]),
    ("end", 0, 11, [
        "neuro san E.S.P. Measure any neuro san network. Search for a better one. "
        "Serve the winner.",
        "Built on neuro san, from Cognizant A.I. Lab. Directed and debugged by Siva Kumar "
        "Raju, and written with Claude Code.",
    ]),
]


def synthesise(kokoro, text, cache):
    key = hashlib.sha256(f"{VOICE}|{SPEED}|{text}".encode()).hexdigest()[:20]
    path = cache / f"{key}.wav"
    if path.exists():
        samples, rate = sf.read(path, dtype="float32")
        return samples, rate
    samples, rate = kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-us")
    sf.write(path, samples, rate, subtype="FLOAT")
    return samples, rate


def main():
    out_wav, out_js = sys.argv[1], sys.argv[2]
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(os.environ.get("KOKORO_MODEL", str(HERE / "models" / "kokoro-v1.0.onnx")),
                    os.environ.get("KOKORO_VOICES", str(HERE / "models" / "voices-v1.0.bin")))

    timeline, clips, start, rate = [], [], 0.0, 24000
    for name, chapter, shortest, lines in SCENES:
        cues, t = [], LEAD
        for text in lines:
            samples, rate = synthesise(kokoro, text, cache)
            cues.append(round(t, 2))
            clips.append((start + t, samples))
            t += len(samples) / rate + GAP
        duration = round(max(shortest, t - GAP + TAIL), 1)
        timeline.append({"id": name, "chap": chapter, "start": round(start, 2),
                         "dur": duration, "cues": cues})
        print(f"{name:10s} {start:7.1f} s  {duration:5.1f} s  {len(lines)} lines")
        start += duration

    track = np.zeros(int(start * rate) + rate, dtype=np.float32)
    for at, samples in clips:
        begin = int(at * rate)
        track[begin:begin + len(samples)] += samples
    track /= max(float(np.abs(track).max()), 1e-6) / 0.9
    sf.write(out_wav, track[:int(start * rate)], rate, subtype="PCM_16")
    data = json.dumps({"total": round(start, 2), "scenes": timeline})
    Path(out_js).write_text("// Written by narration.py from the measured length of every line.\n"
                            f"window.TIMELINE = {data};\n")
    words = sum(len(t.split()) for _, _, _, lines in SCENES for t in lines)
    print(f"total {start:.1f} s ({start / 60:.2f} min), {len(SCENES)} scenes, {words} words")


if __name__ == "__main__":
    main()
