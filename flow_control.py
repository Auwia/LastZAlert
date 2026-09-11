#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


FLOW_NAMES = (
    "treasure",
    "hq",
    "heal",
    "donation",
    "ministry",
    "forziere",
    "generic",
    "research",
    "rally",
    "hero",
    "bounty",
)


FLOW_ENABLED_PATHS = {
    flow_name: BASE_DIR / f"{flow_name}_enabled.txt"
    for flow_name in FLOW_NAMES
}


def get_flow_enabled_path(flow_name: str) -> Path:
    if flow_name not in FLOW_ENABLED_PATHS:
        raise ValueError(f"workflow non valido: {flow_name}")

    return FLOW_ENABLED_PATHS[flow_name]


def is_flow_enabled(flow_name: str) -> bool:
    path = get_flow_enabled_path(flow_name)

    try:
        value = path.read_text(encoding="utf-8").strip()
        return value != "0"

    except FileNotFoundError:
        return True

    except Exception as exc:
        print(
            f"[FLOW ENABLE] errore lettura {path}: {exc}",
            flush=True,
        )
        return True


def set_flow_enabled(flow_name: str, enabled: bool) -> bool:
    path = get_flow_enabled_path(flow_name)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "1\n" if enabled else "0\n",
        encoding="utf-8",
    )

    return enabled


def get_all_flow_states() -> dict:
    return {
        flow_name: is_flow_enabled(flow_name)
        for flow_name in FLOW_NAMES
    }
