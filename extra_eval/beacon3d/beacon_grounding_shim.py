#!/usr/bin/env python3
"""Run Beacon3D's ``evaluate_grounding.py`` with one guard, without editing it.

Its ``tag_score`` divides by the number of cases carrying a knowledge tag, and
3RScan / MultiScan have no ``functionality`` cases, so the upstream script dies with
``ZeroDivisionError`` after printing the tags that do exist. This wrapper imports the
upstream module from the cwd (the Beacon3D clone), skips tags with no cases, and
otherwise runs ``main()`` unchanged -- every printed number is the upstream code's.
"""

import os
import sys

sys.path.insert(0, os.getcwd())  # the Beacon3D dir; to_beacon3d.py runs this with cwd there
import evaluate_grounding as eg  # noqa: E402

_orig = eg.eval_score


def eval_score(infer_results_mapping, processed_grounding):
    present = {k for v0 in infer_results_mapping.values() for _, tag in v0.values() for k in tag}
    kept = eg.KNOWLEDGE_TYPES
    eg.KNOWLEDGE_TYPES = [k for k in kept if k in present]
    try:
        _orig(infer_results_mapping, processed_grounding)
    finally:
        eg.KNOWLEDGE_TYPES = kept


eg.eval_score = eval_score

if __name__ == "__main__":
    sys.exit(eg.main())
