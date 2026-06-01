"""
baselines/lrao_mlp.py — MLP adaptation of the CNN-LRao detector for i.i.d. data.

The original CNN-LRao repo (LRao-detector-main/) uses 1D Conv layers designed
for sequential time-series data. For i.i.d. hyperspectral pixels there is no
temporal structure, so we replace the Conv1D layers with a standard MLP while
keeping the same training objective: maximize Linear Fisher Information (LFI).

Both architectures optimize the same criterion:
    J = ĝ^T Ĉ_Ψ^{-1} ĝ   (LFI w.r.t. target direction s)

and use the same LLMP detection statistic. The difference is architectural only:
- LRao-CNN (original): 1D Conv + Tanh, designed for correlated time-series
- LRao-MLP (this file): Linear + Tanh, appropriate for i.i.d. feature vectors

Since the objective and detection statistic are identical, LRao-MLP is
mathematically equivalent to our ScoreNet + train_lfi from dsm_model.py.
This class provides a drop-in wrapper using the same interface as the Trafo
class from the original repo, for traceability and fair comparison.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsm_model import train_lfi, compute_lfi_detector_scores


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrafoMLP(nn.Module):
    """
    MLP replacement for the Trafo (Conv1D) class.

    Maps R^d → R^d, trained to maximize LFI at its output.

    Compared to ScoreNet in dsm_model.py, this class uses Tanh activations
    (matching the original CNN-LRao) rather than SiLU. Otherwise identical.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None,
                 activation: str = 'tanh'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        act_map = {'tanh': nn.Tanh, 'silu': nn.SiLU, 'relu': nn.ReLU}
        act_cls = act_map[activation]

        dims   = [input_dim] + list(hidden_dims) + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)
        self._apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lrao_mlp(model: TrafoMLP,
                   train_data: np.ndarray,
                   s: np.ndarray,
                   config: dict = None) -> TrafoMLP:
    """
    Train a TrafoMLP by maximizing the Linear Fisher Information (LFI).

    This wraps dsm_model.train_lfi with a config-dict interface matching
    the style of the original CNN-LRao training functions.

    Parameters
    ----------
    model      : TrafoMLP instance
    train_data : (n, d) background training samples
    s          : (d,) target signature (unit-norm)
    config     : dict with keys (all optional, sensible defaults):
        lr            : float   = 1e-3
        weight_decay  : float   = 1e-4
        batch_size    : int     = 64
        epochs        : int     = 1000
        delta_theta   : float   = 0.01
        device        : str     = 'cpu'
        print_every   : int     = 200
        checkpointer  : Checkpointer or None

    Returns
    -------
    Trained model (on CPU).
    """
    if config is None:
        config = {}

    return train_lfi(
        model        = model,
        data         = train_data,
        s            = s,
        delta_theta  = config.get('delta_theta',  0.01),
        lr           = config.get('lr',           1e-3),
        batch_size   = config.get('batch_size',   64),
        epochs       = config.get('epochs',       1000),
        device       = config.get('device',       'cpu'),
        print_every  = config.get('print_every',  200),
        weight_decay = config.get('weight_decay', 1e-4),
        checkpointer = config.get('checkpointer', None),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_lrao_mlp(test_data: np.ndarray,
                    train_data: np.ndarray,
                    model: TrafoMLP,
                    s: np.ndarray,
                    delta_theta: float = 0.01) -> np.ndarray:
    """
    LLMP detection statistic using a trained TrafoMLP.

    Identical formula to LRao-IID (uses compute_lfi_detector_scores):
        T(y) = ĝ^T Ĉ_Ψ^{-1}(Ψ(y) - μ̂_Ψ) / √Ĵ

    Parameters
    ----------
    test_data  : (n_test, d)
    train_data : (n_train, d)
    model      : trained TrafoMLP
    s          : (d,) target signature
    delta_theta: finite-difference step for Jacobian estimate

    Returns
    -------
    scores : (n_test,)
    """
    return compute_lfi_detector_scores(model, train_data, test_data, s, delta_theta)
