# --------------------
# ROI DEFINITIONS
# --------------------
# ROI tutto schermo:
# "roi": (0.0, 1.0, 0.0, 1.0),

SIMPLE_EVENTS = {
    "heal_finished": {
        "templates": "heal_finished",
        "roi": (0.697, 0.937, 0.581, 0.693),
        "threshold": 0.88,
        "cooldown": 1,
    },
    "colleague": {
        "templates": "colleague",
        "roi": (0.719, 0.857, 0.730, 0.793),
        "threshold": 0.75,
        "cooldown": 1,
    },
    "heal_help": {
        "templates": "heal_help",
        "roi": (0.697, 0.937, 0.581, 0.693),
        "threshold": 0.88,
        "cooldown": 1,
    },
    "back_home": {
        "templates": "back_home",
        "roi": (0.75, 1.0, 0.75, 1.0),
        "threshold": 0.9,
        "cooldown": 10,
        "tap": "bottom_right",
        "log": "[BACK_AT_HOME]",
    },
    "confirm_popup": {
        "templates": "confirm_popup",
        "roi": (0.0, 1.0, 0.0, 1.0),
        "threshold": 0.9,
        "cooldown": 2,
        "tap": "OUTSIDE",
        "log": "[CONFIRM_POPUP]",
    },
    "cancel_popup": {
        "templates": "cancel_popup",
        "roi": (0.0, 1.00, 0.0, 1.0),
        "threshold": 0.9,
        "cooldown": 2,
        "log": "[CANCEL_POPUP]",
    },
    #"electricity": {
    #    "templates": "electricity",
    #    "roi": (0.131, 0.610, 0.161, 0.439),
    #    "threshold": 0.85,
    #    "cooldown": 600,
    #},
    #"alloy": {
    #    "templates": "alloy",
    #    "roi": (0.131, 0.610, 0.161, 0.439),
    #    "threshold": 0.80,
    #    "cooldown": 500,
    #},
    "zelt": {
        "templates": "zelt",
        "roi": (0.131, 0.610, 0.161, 0.439),
        "threshold": 0.80,
        "cooldown": 700,
    },
    #"meal": {
    #    "templates": "meal",
    #    "roi": (0.131, 0.610, 0.161, 0.439),
    #    "threshold": 0.80,
    #    "cooldown": 400,
    #},
    #"wood": {
    #    "templates": "wood",
    #    "roi": (0.131, 0.610, 0.161, 0.439),
    #    "threshold": 0.80,
    #    "cooldown": 450,
    #},
    #"experience": {
    #    "templates": "experience",
    #    "roi": (0.131, 0.610, 0.161, 0.439),
    #    "threshold": 0.80,
    #    "cooldown": 1000,
    #},
}
