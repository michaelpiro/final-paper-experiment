"""Importing this package registers every detector via @register side-effects."""

from . import classical          # noqa: F401  AMF, Reg-AMF, CEM, GMM-Levin
from . import gmm_variants       # noqa: F401  Self-GMM, Spatial-GMM
from . import ours               # noqa: F401  CF-Attn(-CFAR), NeighborMLP, DSM

# MCLT (deep template) is optional — it pulls heavier deps; import lazily.
try:
    from . import mclt           # noqa: F401
except Exception as _e:          # pragma: no cover
    import warnings
    warnings.warn(f"MCLT detector not loaded: {_e}")
