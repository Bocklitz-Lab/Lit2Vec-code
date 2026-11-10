"""Shared pipeline registry for reproducibility and production profiles."""

from __future__ import annotations

from typing import Iterable


DEFAULT_STAGES = [
    "metadata",
    "normalize",
    "chunk",
    "embed",
    "validate",
    "compare",
]


STAGE_MODULES = {
    "metadata": "pipelines.01_Metadata.run",
    "normalize": "pipelines.02_Normalize.run",
    "chunk": "pipelines.03_Chunk.run",
    "embed": "pipelines.04_Embed.run",
    "validate": "pipelines.05_Validate.run",
    "compare": "pipelines.06_Compare.run",
}


STAGE_ARGS = {
    "metadata": [],
    "normalize": [],
    "chunk": [],
    "embed": ["--mode", "both"],
    "validate": [],
    "compare": [],
}


PROFILE_DEFAULTS = {
    "repro": DEFAULT_STAGES,
    "production": DEFAULT_STAGES,
}


def resolve_stage_module(stage_name: str) -> str:
    """Return a Python module path for a built-in stage or custom module."""
    return STAGE_MODULES.get(stage_name, stage_name)


def default_stages_for_profile(profile: str) -> list[str]:
    """Return the configured default stage names for a known profile."""
    return list(PROFILE_DEFAULTS.get(profile, DEFAULT_STAGES))


def stage_args_for(stage_name: str) -> list[str]:
    """Return default CLI args for built-in stages."""
    return list(STAGE_ARGS.get(stage_name, []))


def normalize_stage_list(stages: Iterable[str] | None, profile: str) -> list[str]:
    """Pick explicit stages from config or fall back to a profile default."""
    if stages:
        return [stage for stage in stages]
    return default_stages_for_profile(profile)
