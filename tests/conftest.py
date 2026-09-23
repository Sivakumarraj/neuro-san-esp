"""Keep the suite independent of the machine it runs on.

Set before any `esp` module is imported, because `esp` loads .env on import and
fixes its model settings from the environment at the same moment.

A developer's own .env -- a Claude key and a Claude model ladder, say -- would
otherwise reach tests that pin the Gemini population the committed
measurements were taken on, and fail them for a reason unrelated to the code.
Provider tests that need a specific configuration set it themselves.
"""

import os

os.environ["ESP_NO_DOTENV"] = "1"
# Provider keys too: the provider is chosen from whichever key is present, so a
# key exported in the developer's shell would otherwise pick it for the suite.
for _name in ("ESP_DEFAULT_MODEL", "ESP_MODEL_TIERS", "ESP_RPM", "ESP_MODELS",
              "ESP_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
              "GOOGLE_API_KEYS", "OPENROUTER_API_KEY"):
    os.environ.pop(_name, None)
