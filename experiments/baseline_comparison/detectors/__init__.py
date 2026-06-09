"""Importing this package registers every detector via @register side-effects."""

from . import classical          # noqa: F401  AMF, Reg-AMF, CEM, GMM-Levin
from . import gmm_variants       # noqa: F401  Self-GMM, Spatial-GMM
from . import ecem               # noqa: F401  E-CEM (ensemble cascaded CEM)
from . import ours               # noqa: F401  CF-Attn(-CFAR), NeighborMLP, DSM

# Deep baselines are optional — they pull heavier deps; import each lazily so a
# missing dependency disables only that detector.
for _m in ["mclt", "htd_irn", "tsttd"]:
    try:
        __import__(f"{__name__}.{_m}")
    except Exception as _e:       # pragma: no cover
        import warnings
        warnings.warn(f"detector '{_m}' not loaded: {_e}")
