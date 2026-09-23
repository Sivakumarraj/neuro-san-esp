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
for _name in ("ESP_DEFAULT_MODEL", "ESP_MODEL_TIERS", "ESP_RPM", "ESP_MODELS"):
    os.environ.pop(_name, None)
