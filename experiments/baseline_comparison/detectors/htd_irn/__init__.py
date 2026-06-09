"""
HTD-IRN adapter — faithful to the authors' Train_Test.py (identical model, loss,
hyperparameters and training loop). Only the I/O is adapted: the "image" is our
(planted) test box and the prior target spectrum is our signature (replacing
their ts_generation). Transductive: trains on the test image each call.

Reference: Shen et al., "Hyperspectral Target Detection Based on Interpretable
Representation Network", IEEE TGRS 2023. Upstream:
  potential_spatial_baselines_code/code/HTD-IRN-main/{Model,Train_Test,utils}.py
"""

from __future__ import annotations
import os, random
import numpy as np
import torch

from ...framework.detector_api import Detector, DetectorInput
from ...framework.registry import register
from .srn import SRN


def _seed_torch(seed=1):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _standard(X):
    mn, mx = float(X.min()), float(X.max())
    return X if mx == mn else (X - mn) / (mx - mn)


def _calculate_loss(RX, X, F_map, t_tensor, t_proj, S, eta1):
    recon_loss = torch.mean(torch.abs(RX - X))
    weight = torch.sum(torch.square(X - t_tensor), dim=1)
    mapping_loss = torch.mean(torch.abs(t_proj - 1))
    energy_loss = torch.mean(weight * torch.square(F_map))
    CEM_loss = 0.1 * energy_loss + mapping_loss
    spar_loss = torch.mean(torch.sum(torch.abs(S), dim=1))
    return recon_loss + CEM_loss + eta1 * spar_loss


@register("HTD-IRN")
class HTDIRN(Detector):
    needs_spatial = True
    transductive = True
    space = "raw"

    def fit(self, ctx):           # transductive: nothing to pre-fit
        return self

    def score(self, ctx: DetectorInput) -> np.ndarray:
        c = self.cfg
        dev = ctx.device if torch.cuda.is_available() else "cpu"
        m = int(c.get("m", 30)); eta1 = float(c.get("eta1", 10))
        iters = int(c.get("iters", 5000))
        lr = float(c.get("lr", 1e-3)); wd = float(c.get("weight_decay", 2e-5))
        _seed_torch(1)

        # --- preprocessing (mirror data_preprocessing) ---
        img = ctx.test_image(raw=True).astype(np.float32)     # (H, W, C)
        img[img < 0] = 0
        # standardize image + signature with the SAME image min/max so the prior
        # lives in the image's scale (their ts_generation runs on standardized hs).
        mn, mx = float(img.min()), float(img.max())
        rng = (mx - mn) if mx != mn else 1.0
        img = (img - mn) / rng
        H, W, C = img.shape
        sig = ((ctx.signature_raw.astype(np.float32) - mn) / rng).reshape(C, 1)

        hs = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(dev)   # (1,C,H,W)
        t = torch.from_numpy(sig).unsqueeze(-1)                                # (C,1,1)
        t_atom = t.unsqueeze(-1).to(dev)                                       # (C,1,1,1)
        t_tensor = t.unsqueeze(0).to(dev)                                      # (1,C,1,1)
        et = torch.tile(t_tensor, (1, 1, H, W))                                # (1,C,H,W)

        model = SRN(C, m).to(dev); model.train()
        print(f"      [HTD-IRN] training on {next(model.parameters()).device} "
              f"| {iters} iters | image {H}x{W}x{C}", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        for i in range(1, iters + 1):
            opt.zero_grad()
            predict_x, det_map, S = model(hs, t_atom)
            _, t_proj, _ = model(et, t_atom)
            loss = _calculate_loss(predict_x, hs, det_map, t_tensor, t_proj, S, eta1)
            loss.backward(); opt.step()
            if i % 1000 == 0:
                for g in opt.param_groups:
                    g['lr'] *= 0.5
                print(f"        iter {i}/{iters}  loss={float(loss.detach()):.4f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            _, det_map, _ = model(hs, t_atom)
        dm = det_map.squeeze(0).detach().cpu().numpy()        # (H, W)
        dm = np.clip(_standard(dm), 0, 1)
        self._log = {"final_loss": float(loss.detach())}
        return dm.reshape(-1)                                  # row-major -> matches pixels
