.PHONY: install test lint lint-docs check check-key smoke measure figures ablation probe baseline search holdout offline null-sweep report proofs dossier primer explainer verify service-report champion studio docker clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests -q

lint:
	ruff check esp tests scripts apps

# The docs, with the markdown linter and settings neuro-san-studio gates on.
lint-docs:
	pymarkdown --config .pymarkdownlint.yaml scan *.md docs/*.md

# Everything CI enforces, in one command.
check: lint lint-docs test

# Does the configured provider accept the key? One free call -- listing models
# costs nothing -- asked before anything long or public starts.
check-key:
	PYTHONPATH=$$PWD python scripts/check_key.py

# The whole path on your key: preflight, then the four showcase questions put
# to the champion as the web page serves it. Real calls, real cost -- a few
# dozen model calls in all. Everything else here is offline by design.
smoke:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/smoke_live.py

# Measure any neuro-san network on any question file -- the capability neuro-san
# lacks. Real calls. NETWORK is a HOCON path or a manifest name; TASKS is a JSON
# Lines file of {"question", "answer"} (default: the built-in 17 questions).
#   make measure NETWORK=registries/my_network.hocon TASKS=my_questions.jsonl
measure:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$${AGENT_TOOL_PATH:-$$PWD} python -m esp.measure $(NETWORK) --tasks $${TASKS:-meridian}

# Every published figure about the Predictor, regenerated from committed data.
figures:
	PYTHONPATH=$$PWD python scripts/surrogate_figures.py

# Does the Predictor pick better networks than chance? Every way of holding out
# three of the committed measurements, trained on the rest exactly as a wake
# is. About 45 CPU-minutes, spread over every core; no key, no calls.
ablation:
	PYTHONPATH=$$PWD python scripts/selection_ablation.py

# Google's free tier only: which Gemini models answer today, and what each one's
# daily cap is. A search that starts on an exhausted model scores every
# candidate zero. Paid providers have no daily cap, so there is nothing to probe.
probe:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/probe_models.py

# Measure the seed topologies only, on the configured provider. Three
# candidates, roughly 500 model calls.
baseline:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/run_esp.py --generations 0 --out results

# The full loop: measure, train, search free, pay for the elite.
search:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/run_esp.py --generations 3 --elite 3 --out results

# Select a winner on half the tasks, then judge it on the half it was not
# selected on. Costs nothing: every evaluation recorded the outcome of each
# individual task, so the question is already answerable from what is committed.
holdout:
	PYTHONPATH=$$PWD python scripts/holdout_report.py

# Phase B and C only: train the Predictor on whatever real evaluations are
# cached, then evolve against it. Zero provider calls, so this runs with no
# budget and no key at all.
offline:
	PYTHONPATH=$$PWD python scripts/offline_search.py --pool 2000

# Is the gate's verdict stable, or an artifact of one fold seed? Sweeps seeds
# and shuffle counts over the committed measurements. Costs nothing, and exists
# because two different nulls for one objective were once documented at once.
null-sweep:
	PYTHONPATH=$$PWD python scripts/null_sweep.py

# Start a real neuro-san server and prove it fires the optimiser by itself.
# The one claim in this repository a unit test cannot check: that an
# invocation:"event" agent on a cron schedule is started by the framework with
# no user and no client attached. Cron is shortened to once a minute for the
# duration; a verification that takes an hour is one nobody runs.
verify:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/verify_periodic.py

# Run the commands whose output appears in the dossier, and record it. The
# transcripts in the PDF are captured, never typed -- a claim that cannot drift
# away from what the repository actually does.
proofs:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/capture_proofs.py

# The same project for a reader who knows nothing at all -- not agents, not
# models, not tokens. Every idea anchored to something ordinary: a restaurant
# kitchen, a wind tunnel, an electricity bill. Makes the same admissions as
# every other document here, because the reader least able to check is the one
# who most deserves them.
explainer:
	PYTHONPATH=$$PWD python -m esp.report.explainer

# The same project explained to somebody who has never seen it: no neuro-san,
# no evolutionary computation, no jargon. Makes the same admissions as the
# dossier -- a beginner's version that drops the parts that did not work is not
# a simpler document, it is a less true one.
primer:
	PYTHONPATH=$$PWD python -m esp.report.primer

# The whole project as one PDF: the gap it fills, every file, the service, the
# captured proofs, and the run report carried in full.
dossier: proofs
	PYTHONPATH=$$PWD python -m esp.report.dossier
	PYTHONPATH=$$PWD python -m esp.report.primer

# Serve the best-measured topology so a person can talk to it. Until this
# existed, the winner of the search was a hash in a report -- one step short of
# the point, which is that one of these topologies is better to actually use.
champion:
	PYTHONPATH=$$PWD python scripts/serve_champion.py --state $${ESP_STATE:-state}

# Every measured network at once, in neuro-san's own accelerator UI, so the
# same question can be put to the designer's shape and to the network that beat
# it. The UI lands on 4173 and talks to a neuro-san server on 8080; nsflow
# starts both.
studio:
	PYTHONPATH=$$PWD python scripts/serve_studio.py
	@# Checked before it is used, because the failure is otherwise a bare
	@# ModuleNotFoundError from inside a make recipe, which says nothing about
	@# what to install.
	@python -c "import nsflow" 2>/dev/null || { \
	  echo ""; \
	  echo "make studio needs nsflow, neuro-san's accelerator UI, and it is not"; \
	  echo "installed. It is an extra because it is a large install:"; \
	  echo ""; \
	  echo "    pip install -e \".[studio]\"      # or .[dev], which includes it"; \
	  echo ""; \
	  exit 1; \
	}
	@# The key, before the UI rather than after. Without this the first failure
	@# is an agent replying "API key not valid" in the chat panel, which is a bad
	@# place to find out and a worse one in front of somebody. Non-fatal: twelve
	@# topologies are worth looking at even with no key, so it warns and opens.
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=$$PWD python scripts/check_key.py --warn --quiet-when-fine
	@# `python -m nsflow.run`, not the `nsflow` console script: the script
	@# installed by nsflow 0.6.19 imports a `main` its own run module does not
	@# define, so it fails on import. The module runs fine, and 0.7 fixes the
	@# script -- the module form works on both.
	@#
	@# The key is exported here because nsflow loads .env relative to its own
	@# install directory, not the working tree, so a key in this repo's .env
	@# never reaches the agents otherwise.
	set -a; [ -f .env ] && . ./.env; set +a; \
	AGENT_MANIFEST_FILE=$$PWD/registries/studio_manifest.hocon \
	AGENT_TOOL_PATH=$$PWD PYTHONPATH=$$PWD \
	python -m nsflow.run

# Turn the service's accumulated population into the report inputs. Without
# this the service is invisible: history.json is written by the batch run, and
# an optimiser could accumulate for weeks while the report still showed the last
# afternoon the batch script finished.
service-report:
	PYTHONPATH=$$PWD python scripts/service_report.py --state $${ESP_STATE:-state} --out results
	PYTHONPATH=$$PWD python -m esp.report.plots
	PYTHONPATH=$$PWD python -c "from esp.report.build import Report; print(Report('results').build())"

report:
	PYTHONPATH=$$PWD python scripts/baseline_report.py results
	PYTHONPATH=$$PWD python -m esp.report.plots
	PYTHONPATH=$$PWD python -c "from esp.report.build import Report; print(Report('results').build())"

docker:
	docker build -t neuro-san-esp .
	docker run --rm neuro-san-esp

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info .esp-networks
