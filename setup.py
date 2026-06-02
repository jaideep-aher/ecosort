"""End-to-end project setup: data -> features -> train -> evaluate -> experiment.

Running ``python setup.py`` reproduces the entire pipeline from scratch:

1. Download & materialize the TrashNet subset (``scripts/make_dataset.py``)
2. Extract classical colour+HOG features (``scripts/build_features.py``)
3. Train all three models (``scripts/model.py::train_all``)
4. Evaluate on the test set + write metrics/plots (``scripts/evaluate.py``)
5. Run the robustness experiment (``scripts/experiment.py``)

Use ``--quick`` to train the deep model for fewer epochs (smoke test).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def run_pipeline(skip_experiment: bool = False) -> None:
    """Execute the full training/evaluation pipeline."""
    import make_dataset
    import build_features
    import model
    import evaluate

    print("\n########## STEP 1: dataset ##########")
    make_dataset.main()

    print("\n########## STEP 2: classical features ##########")
    build_features.main()

    print("\n########## STEP 3: train all models ##########")
    model.train_all()

    print("\n########## STEP 4: evaluate ##########")
    evaluate.evaluate_all()

    if not skip_experiment:
        print("\n########## STEP 5: robustness experiment ##########")
        import experiment
        experiment.run_experiment()

    print("\nPipeline complete. Artifacts in models/ and data/outputs/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoSort project setup")
    parser.add_argument("--quick", action="store_true",
                        help="Train the deep model for 2 epochs (smoke test).")
    parser.add_argument("--skip-experiment", action="store_true",
                        help="Skip the robustness experiment step.")
    args = parser.parse_args()

    if args.quick:
        os.environ["ECOSORT_EPOCHS"] = "2"

    run_pipeline(skip_experiment=args.skip_experiment)


if __name__ == "__main__":
    main()
