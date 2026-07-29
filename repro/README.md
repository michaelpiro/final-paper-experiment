# repro — clean camera-ready experiment package (branch `rebuttal`)

One self-contained implementation of every experiment in the paper. No vendored
`src` duplicates, no hardcoded hyperparameters (everything in `configs/`),
every trainer seeds torch/numpy **before** model construction (the seed-order
bug found in the camera-ready LRao and THANTD trainers is fixed here).

```
repro/
  configs/    spatial.yaml (all spatial knobs), iid_single.yaml, iid_multi.yaml
  data/       pavia-u.mat, Sandiego{,2}.mat, region jsons, manual_boxes.json
  core/       verbatim ports of the paper pipeline's src/{data,models,detectors,metrics}.py
              + seeding.py (the seed contract) + normalization.py (robust whitenings)
  scenes/     pavia4 / sandiego / sandiego2 builders, fingerprint-asserted
  models/     dart/  darts/  lrao/  classical/  deep/{thantd,htdnet,tsttd,osvae}
              (each: trainer + detector; deep wrappers share a uniform fit/score API)
  protocols/  spatial.py (scene x seed x theta sweep, 12 detectors)
              iid.py     (the verbatim published run_iid; robust LRao native)
  analysis/   tables.py, verify.py + reference.json (published numbers), figures.py
RunSpatial.ipynb / RunIID.ipynb — the two verification notebooks (Colab).
```

LRao protocol = the published `lrao_val2` configuration: 20% validation split,
early stopping on the validation LFI cost (its authors' prescribed usage),
[128] hidden (DART's architecture), robust median/MAD whitening. The IID fixed
LRao uses the published median/IQR variant (`lrao_input_norm: robust`).

## Verification workflow
1. Run RunSpatial.ipynb and RunIID.ipynb on Colab (each cell resumable).
2. `verify.py` prints fresh-vs-published deltas (training-free detectors must
   match to ~1e-3; trained detectors within seed-level stds).
3. Return the zips for archiving into `camera_ready/`.

## Pending before first full run (blocked on local file access when created)
- Diff `configs/spatial.yaml` against `tsp_repro/configs/spatial.yaml`
  (pre-clean branch) and rename to `spatial.yaml`.
- Add `lrao_input_norm: robust` default into the two IID yamls (the notebooks
  set it explicitly meanwhile).
- Adapt `analysis/figures.py` (copied from camera_ready/scripts/make_figures.py)
  to read these runs' output layout.
- Delete legacy dirs (tsp_repro/, colab_deep/, experiments/, archive/,
  final_paper_experiments/, potential_spatial_baselines_code/, paper's_cites/,
  old notebooks, root scripts) and push.
- CPU smoke test both protocols with tiny budgets + equivalence spot-checks
  (plant labels vs archived npz, scene fingerprints, AMF/GMM score equality).
