"""
checkpointing.py — Per-epoch model saving with best-AUC and best-loss tracking.
"""

import os
import torch
import torch.nn as nn


class Checkpointer:
    """
    Saves model + optimizer state every `save_every` epochs.
    Separately tracks and overwrites the best AUC and best loss checkpoints.

    Checkpoint format (each .pt file):
        {
            'epoch':          int,
            'state_dict':     model.state_dict(),
            'optimizer':      optimizer.state_dict(),  (optional)
            'metrics':        dict,                    (optional)
        }
    """

    def __init__(self, run_dir: str, save_every: int = 100):
        self.ckpt_dir   = os.path.join(run_dir, 'checkpoints')
        self.save_every = save_every
        self._best_auc  = -1.0
        self._best_loss = float('inf')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _pack(self, epoch, model, optimizer=None, metrics=None):
        payload = {
            'epoch':      epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        if metrics is not None:
            payload['metrics'] = metrics
        return payload

    # ------------------------------------------------------------------
    def save(self, epoch: int, model: nn.Module,
             optimizer=None, metrics: dict = None):
        """Save periodic checkpoint every `save_every` epochs."""
        if epoch % self.save_every == 0:
            path = os.path.join(self.ckpt_dir, f'epoch_{epoch:05d}.pt')
            torch.save(self._pack(epoch, model, optimizer, metrics), path)

    def save_best_auc(self, model: nn.Module, auc: float,
                      epoch: int, optimizer=None):
        """Overwrite best_auc.pt if `auc` improves."""
        if auc > self._best_auc:
            self._best_auc = auc
            torch.save(
                self._pack(epoch, model, optimizer, {'auc': auc}),
                os.path.join(self.ckpt_dir, 'best_auc.pt')
            )

    def save_best_loss(self, model: nn.Module, loss: float,
                       epoch: int, optimizer=None):
        """Overwrite best_loss.pt if `loss` improves."""
        if loss < self._best_loss:
            self._best_loss = loss
            torch.save(
                self._pack(epoch, model, optimizer, {'loss': loss}),
                os.path.join(self.ckpt_dir, 'best_loss.pt')
            )

    def save_final(self, model: nn.Module, epoch: int,
                   optimizer=None, metrics: dict = None):
        """Save final.pt at end of training."""
        torch.save(
            self._pack(epoch, model, optimizer, metrics),
            os.path.join(self.ckpt_dir, 'final.pt')
        )

    # ------------------------------------------------------------------
    def load(self, model: nn.Module, which: str = 'final',
             optimizer=None, device: str = 'cpu'):
        """
        Load a checkpoint into `model` (and optionally `optimizer`).

        Parameters
        ----------
        which : 'final' | 'best_auc' | 'best_loss' | 'epoch_NNNNN'
        """
        name = f'{which}.pt' if not which.endswith('.pt') else which
        path = os.path.join(self.ckpt_dir, name)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('metrics', {}), ckpt.get('epoch', None)

    def list_checkpoints(self):
        return sorted(os.listdir(self.ckpt_dir))
