"""The ESP loop.

    A  seed population is evaluated for real
    B  a Predictor is trained on (genome -> each outcome objective)
    C  thousands of candidates are evolved against the Predictor, free
    D  only the elite are evaluated for real, and feed back into B

Phase C is the point. A plain genetic algorithm would have to run every
candidate through a language model; here the search is free and only the
promising few are paid for.

Three words get used loosely about this loop and mean different things:

* **Predictor** -- the surrogate. One model per measured outcome objective
  (accuracy, token cost), learned from real evaluations. It never sees the
  weights below, and it never sees a fitness.
* **Fitness** -- `scalarise` below. A fixed weighting, not a learned thing,
  applied to outcomes. Over *measured* outcomes it scores Phase A and D; over
  *predicted* outcomes it ranks Phase C. Same function either way.
* **Prescription** -- what proposes the next candidate. In the ESP paper this
  is a neural network evolved against the surrogate. Here it is not: it is
  seven mutation operators and elite selection, run against the Predictor.
  That departure is real and is stated in the README rather than smoothed over.

**And there is a deeper departure than the missing Prescriptor, which explains
it.** ESP is a *context to actions to outcomes* loop: a Prescriptor is a model
mapping a **context** to the actions to take in it. This project has no context
variable anywhere -- every candidate is evaluated against the same fixed task
set, so the mapping a Prescriptor would learn has nothing to take as input.
Mutation operators are not a lazy stand-in for a Prescriptor here; with no
context they are the only thing that can occupy that slot at all. Adding a
learned Prescriptor therefore starts with deciding what the context *is*, and
that decision is the research, not the network.

Phase B used to train one model directly on the scalarised fitness, which put
the weighting inside the surrogate and made the three indistinguishable. See
`esp/surrogate/outcomes.py` for why that was wrong and what it concealed.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from esp.eval import failover
from esp.eval.runner import Evaluation, QuotaExhausted, evaluate
from esp.genome.definition import Genome
from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.surrogate.outcomes import NULL_TRIALS, Outcome, OutcomeSurrogate
from esp.surrogate.predictor import MIN_SAMPLES

# Accuracy dominates: a cheap network that answers nothing is worthless. Cost
# and size break ties between networks that are equally right, which is exactly
# when they should matter.
WEIGHTS = {"accuracy": 1.0, "tokens": 0.06, "agents": 0.02}

# The token count that costs a candidate the full token penalty.
#
# Set from measurement rather than taste. A scale low enough to put every
# candidate past the cap makes min() clip them to the same maximum penalty: a
# network costing 300k and one costing 900k would score identically on cost,
# and the objective would carry no gradient at all -- multi-objective in shape
# while optimising accuracy alone.
#
# The measured candidates span roughly 250k to 475k tokens. 600k keeps all of
# them inside the range, with room to reward a cheaper topology and penalise a
# profligate one. The cap remains because past this point a topology is runaway
# rather than merely expensive.
TOKEN_SCALE = 600_000.0


def scalarise(accuracy: float, tokens: int, agents: int) -> float:
    """Outcomes in, fitness out. The one place the weighting lives.

    Kept separate from `fitness` because the same weighting has to be applied
    to *predicted* outcomes during Phase C, where there is no Evaluation to
    hand -- only the Predictor's estimates. Three modules had grown their own
    copy of this arithmetic; a weighting that disagrees with itself between the
    surrogate that ranks and the loop that selects is the quietest way to make
    a search meaningless.
    """
    return (
        WEIGHTS["accuracy"] * accuracy
        - WEIGHTS["tokens"] * min(tokens / TOKEN_SCALE, 1.0)
        - WEIGHTS["agents"] * (agents / 9.0)
    )


def fitness(evaluation: Evaluation) -> float:
    """Scalarised for selection. The Pareto front is kept separately, because
    the trade-off is the honest result and a single number hides it."""
    return scalarise(evaluation.accuracy, evaluation.tokens, evaluation.agents)


def non_dominated(points: list[tuple[float, float, float]]) -> list[int]:
    """Indices of the non-dominated points on (accuracy up, tokens down,
    agents down).

    Selection used pure scalarised fitness while the Pareto front was computed,
    plotted in three documents, and never consulted. That is multi-objective in
    the report and single-objective in the search: a candidate that is the
    cheapest network measured gets no say in breeding if one weighting puts it
    mid-table. The front is now what breeds.
    """
    front: list[int] = []
    for i, (accuracy, tokens, agents) in enumerate(points):
        dominated = any(
            other_a >= accuracy and other_t <= tokens and other_g <= agents
            and (other_a > accuracy or other_t < tokens or other_g < agents)
            for j, (other_a, other_t, other_g) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


@dataclass
class Record:
    genome_hash: str
    generation: int
    origin: str
    fitness: float
    accuracy: float
    tokens: int
    agents: int
    depth: int
    seconds: float
    predicted: float | None = None


@dataclass
class History:
    records: list[Record] = field(default_factory=list)
    surrogate_quality: list[dict] = field(default_factory=list)
    real_evaluations: int = 0
    surrogate_evaluations: int = 0
    # Set when a run ends before its generations are done -- currently only
    # when the provider's daily budget runs out.
    stopped_early: str = ""

    def best(self) -> Record | None:
        return max(self.records, key=lambda r: r.fitness, default=None)

    def best_per_generation(self) -> list[Record]:
        out: list[Record] = []
        for generation in sorted({r.generation for r in self.records}):
            upto = [r for r in self.records if r.generation <= generation]
            out.append(max(upto, key=lambda r: r.fitness))
        return out

    def pareto(self) -> list[Record]:
        """Non-dominated on (accuracy up, tokens down, agents down).

        Shares `non_dominated` with the selection step, so the front the
        reports draw and the front the search breeds from cannot drift apart.
        """
        points = [(r.accuracy, float(r.tokens), float(r.agents))
                  for r in self.records]
        front = [self.records[i] for i in non_dominated(points)]
        seen: set[str] = set()
        unique = []
        for record in sorted(front, key=lambda r: (-r.accuracy, r.tokens)):
            if record.genome_hash not in seen:
                seen.add(record.genome_hash)
                unique.append(record)
        return unique


class Evolution:
    def __init__(self, seed: int = 20260821, elite: int = 3,
                 surrogate_pool: int = 400, real_per_generation: int = 4,
                 out_dir: str = "results"):
        self.rng = random.Random(seed)
        self.seed = seed
        self.elite = elite
        self.surrogate_pool = surrogate_pool
        self.real_per_generation = real_per_generation
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.history = History()
        self.surrogate = OutcomeSurrogate(seed=seed)
        self.pool: dict[str, Genome] = {}       # hash -> genome, everything seen
        self.scored: dict[str, float] = {}      # hash -> real fitness
        # What the Predictor actually learns. Fitness is derived from these,
        # so it is not stored as a target -- storing it as one is the mistake
        # this replaced.
        self.outcomes: dict[str, Outcome] = {}  # hash -> measured outcomes

    # ------------------------------------------------------------------ phases

    def _evaluate_real(self, genome: Genome, generation: int, origin: str,
                       predicted: float | None = None) -> Record:
        digest = genome.genome_hash()
        started = time.monotonic()
        evaluation = evaluate(genome)
        value = fitness(evaluation)

        self.pool[digest] = genome
        self.scored[digest] = value
        self.outcomes[digest] = Outcome(accuracy=evaluation.accuracy,
                                        tokens=evaluation.tokens)
        if not evaluation.from_cache:
            self.history.real_evaluations += 1

        record = Record(
            genome_hash=digest, generation=generation, origin=origin,
            fitness=round(value, 4), accuracy=evaluation.accuracy,
            tokens=evaluation.tokens, agents=evaluation.agents,
            depth=evaluation.depth, seconds=round(time.monotonic() - started, 1),
            predicted=None if predicted is None else round(predicted, 4),
        )
        self.history.records.append(record)
        cached = " (cached)" if evaluation.from_cache else ""
        print(f"  gen{generation} {origin:16s} {digest} "
              f"acc={evaluation.accuracy:.2f} tok={evaluation.tokens:6d} "
              f"agents={evaluation.agents} fit={value:+.4f}{cached}", flush=True)
        return record

    def _parents(self) -> list[Genome]:
        """Who is allowed to breed: the Pareto front, then the best scalarised.

        The front comes first because a network that is non-dominated is
        provably not beaten on every objective at once, which is a stronger
        claim than sitting high under one particular weighting. When the front
        is smaller than `elite` the remaining slots go to the best fitness, so
        a one-point front does not collapse the search onto a single parent.
        """
        digests = list(self.scored)
        if not digests:
            return []
        points = [(self.outcomes[h].accuracy, float(self.outcomes[h].tokens),
                   float(len(self.pool[h].reachable()))) for h in digests]
        chosen = [digests[i] for i in non_dominated(points)][:self.elite]

        if len(chosen) < self.elite:
            for digest in sorted(digests, key=lambda h: -self.scored[h]):
                if digest not in chosen:
                    chosen.append(digest)
                if len(chosen) >= self.elite:
                    break
        return [self.pool[h] for h in chosen]

    def _breed(self, parents: list[Genome], count: int) -> list[tuple[Genome, str]]:
        """Mutate parents until `count` distinct unseen genomes exist.

        Distinctness is checked by genome hash: re-proposing something already
        evaluated wastes a slot in the elite batch, which is the expensive one.
        """
        produced: dict[str, tuple[Genome, str]] = {}
        attempts = 0
        while len(produced) < count and attempts < count * 60:
            attempts += 1
            parent = self.rng.choice(parents)
            try:
                child, operator = mutate(parent, self.rng)
            except InvalidMutant:
                continue
            digest = child.genome_hash()
            if digest in self.scored or digest in produced:
                continue
            produced[digest] = (child, operator)
        return list(produced.values())

    def run(self, generations: int = 4) -> History:
        print(f"ESP run: seed={self.seed} generations={generations} "
              f"surrogate_pool={self.surrogate_pool} elite={self.real_per_generation}",
              flush=True)

        # --- Phase A: seed the Predictor with real measurements
        print("\nPhase A -- seed population, real evaluation", flush=True)
        try:
            for name, make in SEEDS.items():
                self._evaluate_real(make(), 0, f"seed:{name}")

            seed_parents = [self.pool[h] for h in self.scored]
            for genome, operator in self._breed(seed_parents, 6):
                self._evaluate_real(genome, 0, f"mut:{operator}")
        except QuotaExhausted as exc:
            # Running out of provider budget is an expected end to a run, not a
            # crash. Everything measured before this point is real and has been
            # paid for; losing it to a traceback would mean paying again.
            self._stop("provider budget exhausted during the seed population", exc)
            return self.history

        # --- Phases B, C, D, repeated
        for generation in range(1, generations + 1):
            genomes = [self.pool[h] for h in self.scored]
            outcomes = [self.outcomes[h] for h in self.scored]

            print(f"\nGeneration {generation}", flush=True)
            # The null costs `NULL_TRIALS` extra cross-validations per
            # objective -- seconds of CPU against a generation that costs
            # real money and minutes. It is measured here because without it
            # the gate below has no evidence to act on.
            quality = self.surrogate.report_quality(genomes, outcomes,
                                                    seed=self.seed,
                                                    null_trials=NULL_TRIALS)
            print(f"  Phase B -- {quality}", flush=True)
            # Named *and* acted on. Until this gate existed, an objective the
            # Predictor ranked worse than its own null still contributed its
            # full weight to every Phase C decision, and the combined figure
            # above did not show it: on the committed population the token
            # model is reliably *anti*-correlated and the accuracy model
            # carries the total. Steering by a predictor that orders
            # backwards is worse than not predicting that objective at all.
            gated = quality.gated_outcomes()
            for useless in quality.useless_outcomes():
                margin = quality.margin(useless)
                if useless in gated:
                    print(f"  Phase B -- the {useless} model ranks no better "
                          f"than its own null ({margin:+.3f} against it). "
                          f"EXCLUDED from Phase C; held at the population "
                          f"mean so it cannot order candidates.", flush=True)
                else:
                    print(f"  Phase B -- WARNING: the {useless} model ranks "
                          f"no better than chance, but its own null was not "
                          f"measured, so it is not safe to exclude. Phase C "
                          f"is still weighting it.", flush=True)
            record = {"generation": generation, **quality.as_record()}
            record["gated_outcomes"] = gated
            self.history.surrogate_quality.append(record)
            self.surrogate.fit(genomes, outcomes, gated=gated)

            # Phase C: search wide, for free.
            parents = self._parents()
            candidates = self._breed(parents, self.surrogate_pool)
            self.history.surrogate_evaluations += len(candidates)
            if not candidates:
                print("  Phase C -- no new candidates; search exhausted", flush=True)
                break
            predictions = self.surrogate.predict([c for c, _ in candidates])
            ranked = sorted(zip(candidates, predictions, strict=True),
                            key=lambda pair: -pair[1])
            if self.surrogate.ranks():
                print(f"  Phase C -- {len(candidates)} candidates scored by "
                      f"surrogate, best predicted {ranked[0][1]:+.4f}", flush=True)
            else:
                # Every prediction is the same constant, so the sort above did
                # nothing and Phase D is about to buy the first few candidates
                # in generation order. That is a random search, and printing a
                # "best predicted" score over it would hide the one fact that
                # matters about this generation.
                print(f"  Phase C -- {len(candidates)} candidates generated, but "
                      f"the surrogate is UNTRAINED ({len(genomes)} of "
                      f"{MIN_SAMPLES} samples): every prediction is the same "
                      f"constant, so the elite below is an arbitrary slice, not "
                      f"a selection. This generation is a random search.",
                      flush=True)

            # Phase D: pay for the elite only.
            print("  Phase D -- real evaluation of the elite", flush=True)
            try:
                for (genome, operator), predicted in ranked[:self.real_per_generation]:
                    self._evaluate_real(genome, generation, f"mut:{operator}",
                                        predicted=float(predicted))
            except QuotaExhausted as exc:
                self._stop(f"provider budget exhausted in generation {generation}", exc)
                return self.history

            self.save()

        self.save()
        return self.history

    # ------------------------------------------------------------------- output

    def _stop(self, why: str, exc: Exception) -> None:
        """End a run early, keeping everything already measured.

        Also records why, in the history itself. A run that stopped because the
        provider ran out of budget is a different artefact from one that
        finished, and a reader who cannot tell them apart will read a short
        fitness curve as a converged search.
        """
        self.history.stopped_early = why
        print(f"\n  STOPPED: {why}\n  {str(exc)[:200]}", flush=True)
        swapped = failover.swaps()
        if swapped:
            print(f"  model swaps: {swapped}", flush=True)
        self.save()

    def save(self) -> None:
        payload = {
            "seed": self.seed,
            "weights": WEIGHTS,
            "real_evaluations": self.history.real_evaluations,
            "surrogate_evaluations": self.history.surrogate_evaluations,
            "surrogate_quality": self.history.surrogate_quality,
            "stopped_early": self.history.stopped_early,
            "model_swaps": failover.swaps(),
            "records": [asdict(r) for r in self.history.records],
            "pareto": [asdict(r) for r in self.history.pareto()],
        }
        (self.out_dir / "history.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

        best = self.history.best()
        if best:
            (self.out_dir / "best.hocon").write_text(
                self.pool[best.genome_hash].to_hocon(), encoding="utf-8")
