#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """
    Load local environment variables from a .env file.

    Existing environment variables always take precedence.
    Empty lines and comments are ignored.
    """

    if not path.is_file():
        return

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key:
                os.environ.setdefault(key, value)


load_env_file()


DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

ADB_DEVICE = os.environ.get("ADB_DEVICE", "")
