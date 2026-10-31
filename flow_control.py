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
    "workers",
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


# ------------------------------------------------------------
# Ministry sub-options
# ------------------------------------------------------------


# ------------------------------------------------------------
# Treasure workflow mode
# ------------------------------------------------------------


# ------------------------------------------------------------
# Research workflow mode
# ------------------------------------------------------------

RESEARCH_FLOW_MODE_PATH = BASE_DIR / "research_flow_mode.txt"
RESEARCH_FLOW_MODES = ("recommended", "image_scan")


def get_research_flow_mode() -> str:
    try:
        mode = RESEARCH_FLOW_MODE_PATH.read_text(
            encoding="utf-8"
        ).strip().lower()
    except FileNotFoundError:
        return "recommended"

    if mode not in RESEARCH_FLOW_MODES:
        return "recommended"

    return mode


def set_research_flow_mode(mode: str) -> str:
    mode = str(mode).strip().lower()

    if mode not in RESEARCH_FLOW_MODES:
        raise ValueError(f"Invalid research flow mode: {mode}")

    RESEARCH_FLOW_MODE_PATH.write_text(
        mode + "\n",
        encoding="utf-8",
    )
    return mode


TREASURE_FLOW_MODE_PATH = BASE_DIR / "treasure_flow_mode.txt"
TREASURE_FLOW_MODES = ("simplified", "original")


def get_treasure_flow_mode() -> str:
    try:
        mode = TREASURE_FLOW_MODE_PATH.read_text(
            encoding="utf-8"
        ).strip().lower()
    except FileNotFoundError:
        return "simplified"

    if mode not in TREASURE_FLOW_MODES:
        return "simplified"

    return mode


def set_treasure_flow_mode(mode: str) -> str:
    mode = str(mode).strip().lower()

    if mode not in TREASURE_FLOW_MODES:
        raise ValueError(f"Invalid treasure flow mode: {mode}")

    TREASURE_FLOW_MODE_PATH.write_text(
        mode + "\n",
        encoding="utf-8",
    )

    return mode


MINISTRY_NAMES = (
    "construction",
    "science",
    "agriculture",
)


MINISTRY_ENABLED_PATHS = {
    ministry_name: BASE_DIR / f"ministry_{ministry_name}_enabled.txt"
    for ministry_name in MINISTRY_NAMES
}


def get_ministry_enabled_path(ministry_name: str) -> Path:
    if ministry_name not in MINISTRY_ENABLED_PATHS:
        raise ValueError(f"ministero non valido: {ministry_name}")

    return MINISTRY_ENABLED_PATHS[ministry_name]


def is_ministry_enabled(ministry_name: str) -> bool:
    path = get_ministry_enabled_path(ministry_name)

    try:
        value = path.read_text(encoding="utf-8").strip()
        return value != "0"

    except FileNotFoundError:
        return True

    except Exception as exc:
        print(
            f"[MINISTRY ENABLE] errore lettura {path}: {exc}",
            flush=True,
        )
        return True


def set_ministry_enabled(ministry_name: str, enabled: bool) -> bool:
    path = get_ministry_enabled_path(ministry_name)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "1\n" if enabled else "0\n",
        encoding="utf-8",
    )

    return enabled


def get_all_ministry_states() -> dict:
    return {
        ministry_name: is_ministry_enabled(ministry_name)
        for ministry_name in MINISTRY_NAMES
    }


# ============================================================
# RALLY MONSTER SELECTION
# ============================================================

import json
import threading

RALLY_MONSTERS_DIR = BASE_DIR / "rally" / "monsters"
RALLY_MONSTERS_CONFIG = BASE_DIR / "rally_monsters_enabled.json"
_RALLY_MONSTERS_LOCK = threading.RLock()


def list_rally_monsters():
    """Discover monster templates dynamically."""
    if not RALLY_MONSTERS_DIR.is_dir():
        return []

    return sorted(
        p.name for p in RALLY_MONSTERS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )


def _read_rally_monster_config():
    try:
        data = json.loads(
            RALLY_MONSTERS_CONFIG.read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}


def get_all_rally_monster_states():
    """Unknown monsters default to enabled."""
    with _RALLY_MONSTERS_LOCK:
        saved = _read_rally_monster_config()
        return {
            name: saved.get(name, True) is True
            for name in list_rally_monsters()
        }


def set_rally_monster_enabled(name, enabled):
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")

    if name not in list_rally_monsters():
        raise ValueError("Unknown monster")

    with _RALLY_MONSTERS_LOCK:
        saved = _read_rally_monster_config()
        saved[name] = enabled

        temporary = RALLY_MONSTERS_CONFIG.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(saved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
        temporary.replace(RALLY_MONSTERS_CONFIG)

    return enabled
