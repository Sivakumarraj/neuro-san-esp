.PHONY: install test lint check probe baseline search report proofs dossier primer explainer verify service-report champion docker clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests -q

lint:
	ruff check esp tests scripts apps

# Everything CI enforces, in one command.
check: lint test

# Which models are usable, and what daily budget is left. Costs one call per
# model, and is worth running before any search: the free tier caps requests
# per day per model, and a search that starts on an exhausted model scores
# every candidate zero.
probe:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/probe_models.py

# Measure the seed topologies only. Three candidates, roughly 500 provider
# requests -- one model's entire daily free-tier allowance.
baseline:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/run_esp.py --generations 0 --out results

# The full loop. Needs budget on at least one model in esp/eval/failover.py.
search:
	PYTHONPATH=$$PWD AGENT_TOOL_PATH=$$PWD python scripts/run_esp.py --generations 3 --elite 3 --out results

# Phase B and C only: train the Predictor on whatever real evaluations are
# cached, then evolve against it. Zero provider calls, so this runs with no
# budget and no key at all.
offline:
	PYTHONPATH=$$PWD python scripts/offline_search.py --pool 2000

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

# The same project explained to somebody who has never seen it: no neuro-san,
# no evolutionary computation, no jargon. Makes the same admissions as the
# dossier -- a beginner's version that drops the parts that did not work is not
# a simpler document, it is a less true one.
# The same project for a reader who knows nothing at all -- not agents, not
# models, not tokens. Every idea anchored to something ordinary: a restaurant
# kitchen, a wind tunnel, an electricity bill. Makes the same admissions as
# every other document here, because the reader least able to check is the one
# who most deserves them.
explainer:
	PYTHONPATH=$$PWD python -m esp.report.explainer

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
