"""A measured fitness function for neuro-san agent networks, and the
surrogate-assisted evolutionary search that uses it."""

from esp.config import bootstrap as _bootstrap

# Load .env before any module reads a setting from the environment.
#
# The model, the model ladder and the pacing are fixed when their modules are
# imported. Five entry points -- the unattended optimiser and the studio among
# them -- imported those modules before calling bootstrap(), so a provider
# chosen in .env was silently ignored: a machine holding only a Claude key
# rendered Gemini networks, and every call inside them failed. Loading here
# makes the order impossible to get wrong. A variable already in the
# environment still wins, and ESP_NO_DOTENV=1 turns this off.
_bootstrap()
