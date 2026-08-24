"""The ESP loop.

    A  seed population is evaluated for real
    B  a Predictor is trained on (genome, fitness)
    C  thousands of candidates are evolved against the Predictor, free
    D  only the elite are evaluated for real, and feed back into B

Phase C is the point. A plain genetic algorithm would have to run every
candidate through a language model; here the search is free and only the
promising few are paid for.
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
from esp.surrogate.predictor import MIN_SAMPLES, Surrogate

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


def fitness(evaluation: Evaluation) -> float:
    """Scalarised for selection. The Pareto front is kept separately, because
    the trade-off is the honest result and a single number hides it."""
    return (
        WEIGHTS["accuracy"] * evaluation.accuracy
        - WEIGHTS["tokens"] * min(evaluation.tokens / TOKEN_SCALE, 1.0)
        - WEIGHTS["agents"] * (evaluation.agents / 9.0)
    )


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
        """Non-dominated on (accuracy up, tokens down, agents down)."""
        front: list[Record] = []
        for candidate in self.records:
            dominated = any(
                other.accuracy >= candidate.accuracy
                and other.tokens <= candidate.tokens
                and other.agents <= candidate.agents
                and (other.accuracy > candidate.accuracy
                     or other.tokens < candidate.tokens
                     or other.agents < candidate.agents)
                for other in self.records
            )
            if not dominated:
                front.append(candidate)
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
        self.surrogate = Surrogate(seed=seed)
        self.pool: dict[str, Genome] = {}       # hash -> genome, everything seen
        self.scored: dict[str, float] = {}      # hash -> real fitness

    # ------------------------------------------------------------------ phases

    def _evaluate_real(self, genome: Genome, generation: int, origin: str,
                       predicted: float | None = None) -> Record:
        digest = genome.genome_hash()
        started = time.monotonic()
        evaluation = evaluate(genome)
        value = fitness(evaluation)

        self.pool[digest] = genome
        self.scored[digest] = value
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
            values = [self.scored[h] for h in self.scored]

            print(f"\nGeneration {generation}", flush=True)
            quality = self.surrogate.report_quality(genomes, values, seed=self.seed)
            print(f"  Phase B -- {quality}", flush=True)
            self.history.surrogate_quality.append(
                {"generation": generation, **asdict(quality)})
            self.surrogate.fit(genomes, values)

            # Phase C: search wide, for free.
            parents = [g for _, g in sorted(
                ((self.scored[h], self.pool[h]) for h in self.scored),
                key=lambda pair: -pair[0])[:self.elite]]
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
