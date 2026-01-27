# --------------------
# ROI DEFINITIONS
# --------------------

SIMPLE_EVENTS = {
    "colleague": {
        "templates": "colleague",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.75,
        "cooldown": 3,
    },
    "heal": {
        "templates": "heal",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.72,
        "cooldown": 1,
    },
    "heal_help": {
        "templates": "heal_help",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.88,
        "cooldown": 1,
    },
    "heal_finished": {
        "templates": "heal_finished",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.88,
        "cooldown": 1,
    },
#    "back_home": {
#        "templates": "back_home",
#        "roi": (0.0, 1.00, 0.0, 1.0),
#        "threshold": 0.9,
#        "cooldown": 10,
#        "tap": "center",
#        "log": "[BACK_AT_HOME]",
#    },
    "confirm_popup": {
        "templates": "confirm_popup",
        "roi": (0.0, 1.00, 0.0, 1.0),
        "threshold": 0.9,
        "cooldown": 10,
        "tap": "center",
        "log": "[CONFIRM_POPUP]",
    },
    "cancel_popup": {
        "templates": "cancel_popup",
        "roi": (0.0, 1.00, 0.0, 1.0),
        "threshold": 0.9,
        "cooldown": 2,
        "tap": "center",
        "log": "[CANCEL_POPUP]",
    },
    "electricity": {
        "templates": "electricity",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.85,
        "cooldown": 600,
    },
    "alloy": {
        "templates": "alloy",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.80,
        "cooldown": 500,
    },
    "zelt": {
        "templates": "zelt",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.80,
        "cooldown": 700,
    },
    "meal": {
        "templates": "meal",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.80,
        "cooldown": 400,
    },
    "wood": {
        "templates": "wood",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.80,
        "cooldown": 450,
    },
    "experience": {
        "templates": "experience",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.80,
        "cooldown": 1000,
    },
    "forziere": {
        "templates": "forziere",
         "roi": (0.0, 1.00, 0.0, 1.0),
        "threshold": 0.91,
        "cooldown": 10,
    },
}
