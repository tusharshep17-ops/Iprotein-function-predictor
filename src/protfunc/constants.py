"""Project constants and enzyme-class metadata."""

from __future__ import annotations

EC_CLASSES = {
    "EC1_Oxidoreductase": {
        "number": 1,
        "name": "Oxidoreductase",
        "description": "Catalyses oxidation-reduction reactions.",
    },
    "EC2_Transferase": {
        "number": 2,
        "name": "Transferase",
        "description": "Transfers functional groups between molecules.",
    },
    "EC3_Hydrolase": {
        "number": 3,
        "name": "Hydrolase",
        "description": "Cleaves bonds through hydrolysis.",
    },
    "EC4_Lyase": {
        "number": 4,
        "name": "Lyase",
        "description": "Adds or removes groups to form double bonds without hydrolysis.",
    },
    "EC5_Isomerase": {
        "number": 5,
        "name": "Isomerase",
        "description": "Catalyses intramolecular rearrangements.",
    },
    "EC6_Ligase": {
        "number": 6,
        "name": "Ligase",
        "description": "Joins molecules, usually coupled to nucleotide hydrolysis.",
    },
    "EC7_Translocase": {
        "number": 7,
        "name": "Translocase",
        "description": "Catalyses movement of ions or molecules across membranes.",
    },
}

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ALLOWED_AMINO_ACIDS = frozenset(CANONICAL_AMINO_ACIDS + "BXZJUO")
MIN_SEQUENCE_LENGTH = 20
MAX_SEQUENCE_LENGTH = 10_000

