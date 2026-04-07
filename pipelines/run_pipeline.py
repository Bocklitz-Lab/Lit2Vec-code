"""Shared pipeline runner with repro and production profiles."""

from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import contextmanager
from typing import Iterator

import yaml

from pipelines.pipeline_registry import (
    normalize_stage_list,
    resolve_stage_module,
    stage_args_for,
)


@contextmanager
def temporary_argv(argv: list[str]) -> Iterator[None]:
    """Run stage main() functions with a temporary sys.argv."""
    previous = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_stage(stage_name: str, config_path: str) -> None:
    module_path = resolve_stage_module(stage_name)
    module = importlib.import_module(module_path)
    stage_cli_args = ["--config", config_path, *stage_args_for(stage_name)]

    if not hasattr(module, "main"):
        raise AttributeError(f"Stage module '{module_path}' has no main() entrypoint.")

    print(f"\n=== Running stage: {stage_name} ({module_path}) ===")
    with temporary_argv([module_path, *stage_cli_args]):
        module.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Lit2Vec pipeline profile with shared stage orchestration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["repro", "production"],
        help="Override the config profile and use the matching default stage list.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        help=(
            "Optional explicit stage names or module paths. "
            "If omitted, the runner uses PIPELINE.stages or the selected profile default."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline_config = config.get("PIPELINE", {})
    profile = args.profile or pipeline_config.get("profile", "repro")
    stages = normalize_stage_list(args.stages or pipeline_config.get("stages"), profile)

    print(f"Pipeline profile: {profile}")
    print(f"Stages: {', '.join(stages)}")

    for stage_name in stages:
        run_stage(stage_name, args.config)


if __name__ == "__main__":
    main()
