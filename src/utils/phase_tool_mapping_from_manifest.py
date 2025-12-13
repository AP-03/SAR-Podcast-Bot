import pandas as pd
from collections import Counter, defaultdict

# Path to manifest CSV (update if needed)
MANIFEST_CSV = "/Users/omarahmed/Desktop/UCL/Year 3/Term 1/Deep Learning/SAR-Podcast-Bot/src/dataset/cholec80/cholec80_manifest.csv"

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
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
    "specimen_bag",
]
TOOL_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]

def build_phase_tool_mapping_from_manifest(min_tool_ratio=0.1):
    df = pd.read_csv(MANIFEST_CSV)
    print("First 5 rows of manifest CSV:")
    print(df.head())
    print("\nColumn names:", df.columns.tolist())
    phase_tool_counts = defaultdict(Counter)
    phase_counts = Counter()
    for _, row in df.iterrows():
        phase_id = int(row['phase'])
        if 0 <= phase_id < len(PHASES):
            phase = PHASES[phase_id]
            phase_counts[phase] += 1
            for tool, tool_name in zip(TOOLS, TOOL_NAMES):
                if int(row[tool]) == 1:
                    phase_tool_counts[phase][tool_name] += 1
    print("\nPhase counts:", dict(phase_counts))
    for phase in PHASES:
        print(f"Tool counts for {phase}: {dict(phase_tool_counts[phase])}")
    mapping = {}
    for phase in PHASES:
        total = phase_counts[phase]
        if total == 0:
            mapping[phase] = []
            continue
        tools = [tool for tool, count in phase_tool_counts[phase].items() if count / total >= min_tool_ratio]
        mapping[phase] = tools
    return mapping

def main():
    mapping = build_phase_tool_mapping_from_manifest(min_tool_ratio=0.1)
    print("\nPhase-to-tool mapping (tools used in >1% of frames for each phase):\n")
    for phase, tools in mapping.items():
        print(f"{phase}: {tools}")
    print("\nYou can copy this mapping into your test script.")

if __name__ == "__main__":
    main()
