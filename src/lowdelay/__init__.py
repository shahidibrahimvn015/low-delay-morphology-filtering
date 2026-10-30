"""lowdelay: low-group-delay, morphology-preserving digital filter design and evaluation.

ED5017 DSP course research project - Low-Group-Delay Morphology-Preserving Digital Filtering: Constrained FIR versus Phase-Equalized IIR.
All frequencies inside the package are digital angular frequencies (rad/sample)
unless a variable name ends in ``_hz``.
"""
from .specs import FilterSpec, load_spec
from .filters import Filter

__all__ = ["FilterSpec", "load_spec", "Filter"]
