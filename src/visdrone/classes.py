VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

VALID_CATEGORIES = set(range(1, 11))
CATEGORY_TO_CLASS = {category: category - 1 for category in VALID_CATEGORIES}
CLASS_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_CLASS.items()}

