"""qGridX — Hybrid quantum-classical power-grid storage planning.

Attribution:
  Architecture partially adapted from the gqco reference implementation
  (https://github.com/shunyaist/generative-quantum-combinatorial-optimization,
  MIT License). See NOTICE for full attribution.
"""
__version__ = "0.1.0"

from qgridx.config import Config, ProblemSpec, load_config  # noqa: F401
from qgridx.pipeline import Pipeline, PipelineResult  # noqa: F401
from qgridx.registry import register_component, get_component, list_components  # noqa: F401
