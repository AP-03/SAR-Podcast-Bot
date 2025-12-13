PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging",
]

TOOLS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]

PHASE_TO_IDX = {name: i for i, name in enumerate(PHASES)}
TOOL_TO_IDX = {name: i for i, name in enumerate(TOOLS)}
