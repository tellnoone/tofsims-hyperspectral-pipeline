#!/usr/bin/env python3
"""
run_pipeline.py
Command-line entry point for the ToF-SIMS analysis pipeline.

Examples
--------
Process everything in data/raw with the default configuration:

    python scripts/run_pipeline.py

Process a different dataset without touching the config file:

    python scripts/run_pipeline.py --raw-dir /path/to/new_data --output-dir results/new_run

Re-run just one stage after changing a setting:

    python scripts/run_pipeline.py --stage segment

List the stages and exit:

    python scripts/run_pipeline.py --list-stages
"""
import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import STAGES, run_stages  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ToF-SIMS hyperspectral analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        default="all",
        help=f"Stage to run: all, or one of {', '.join(STAGES)} "
             f"(comma-separate to run several). Default: all.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a pipeline config YAML. Default: configs/pipeline_config.yaml",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Override dataset.raw_dir."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override dataset.processed_dir (and figures_dir unless --figures-dir is given).",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=None, help="Override dataset.figures_dir."
    )
    parser.add_argument(
        "--nmf-k",
        default=None,
        help="Override the NMF component count ('auto' to pick from the error elbow).",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP and HDBSCAN (much faster; the other stages are unaffected).",
    )
    parser.add_argument(
        "--list-stages", action="store_true", help="Print the stage names and exit."
    )
    return parser


def resolve_stages(value: str) -> list:
    """Turn the --stage argument into an ordered list of stage names."""
    if value.strip().lower() == "all":
        return list(STAGES)

    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in STAGES]
    if unknown:
        raise SystemExit(
            f"Unknown stage(s): {', '.join(unknown)}. Valid stages: {', '.join(STAGES)}"
        )
    return requested


def build_overrides(args: argparse.Namespace) -> dict:
    """Translate CLI flags into a config override tree."""
    dataset = {}
    if args.raw_dir:
        dataset["raw_dir"] = str(args.raw_dir)
    if args.output_dir:
        dataset["processed_dir"] = str(args.output_dir)
        dataset["figures_dir"] = str(args.figures_dir or args.output_dir)
    if args.figures_dir:
        dataset["figures_dir"] = str(args.figures_dir)

    overrides = {}
    if dataset:
        overrides["dataset"] = dataset
    if args.nmf_k is not None:
        value = args.nmf_k if args.nmf_k == "auto" else int(args.nmf_k)
        overrides.setdefault("decomposition", {})["nmf"] = {"n_components": value}
    if args.no_umap:
        overrides.setdefault("decomposition", {})["umap"] = {"enabled": False}
        overrides.setdefault("segmentation", {})["hdbscan"] = {"enabled": False}
    return overrides


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_stages:
        print("Pipeline stages, in order:")
        for stage in STAGES:
            print(f"  {stage}")
        return 0

    stages = resolve_stages(args.stage)
    config = load_config(args.config, overrides=build_overrides(args))

    print(f"Config      : {config.config_path}")
    print(f"Raw data    : {config.raw_dir}")
    print(f"Processed   : {config.processed_dir}")
    print(f"Figures     : {config.figures_dir}")
    print(f"Stages      : {', '.join(stages)}")

    try:
        results = run_stages(config, stages)
    except FileNotFoundError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    except Exception:  # pragma: no cover - surfaced to the user verbatim
        traceback.print_exc()
        return 1

    total = sum(len(result) for result in results.values())
    print(f"\nPipeline complete: {total} artefact(s) written to {config.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
