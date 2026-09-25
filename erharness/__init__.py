"""erharness — local scoring harness for the Amazon ML Challenge entity-resolution task.

Mirrors the leaderboard metric (macro F0.5 per Source-1 entity, singletons
all-or-nothing) and adds the tooling to decide what to submit:
decomposition, bootstrap significance, threshold sweeps, oracle ceilings,
blocking audits, entity-level CV splits and an experiment log.
"""
from .decode import DecodeConfig, Prepared, decode
from .io import label_candidates, read_candidates, read_match_sets, write_match_sets
from .metric import evaluate, per_entity, project_score, report_from_table
from .sweep import Scorer, best_config, grid_search

__all__ = [
    "DecodeConfig", "Prepared", "decode", "label_candidates", "read_candidates", "read_match_sets",
    "write_match_sets", "evaluate", "per_entity", "project_score", "report_from_table",
    "Scorer", "best_config", "grid_search",
]
__version__ = "0.1.0"
