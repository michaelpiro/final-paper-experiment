# Colab notebook log — best epochs & results (from RunIIDConfigurable, 08-13)

User-run sessions on Colab T4, std front, noise-before, wd=0, clip 1.0,
eval every 10 ep, best-epoch by Pd@Pfa=.1. single = b2t4 (meadows/trees),
theta=.15. Results files on the Colab side: results_iid_128.json,
ckpt_iid_128/ (best-epoch ckpts + score npz per run).

## single, rho=.03, 5000 ep — best (best_epoch) per seed 42..46

| n | s42 | s43 | s44 | s45 | s46 | mean best |
|---|---|---|---|---|---|---|
| 32   | .655 (3970) | .540 (1650) | .620 (290)  | .555 (320)  | .650 (370)  | .604 |
| 64   | .740 (550)  | .625 (170)  | .550 (4810) | .540 (500)  | .660 (610)  | .623 |
| 128  | .670 (290)  | .620 (220)  | .565 (730)  | .565 (420)  | .655 (130)  | .615 |
| 256  | .695 (250)  | .590 (140)  | .570 (120)  | .580 (200)  | .620 (100)  | .611 |
| 512  | .700 (210)  | .605 (150)  | .600 (4520) | .560 (360)  | .630 (70)   | .619 |
| 1024 | .735 (120)  | .700 (4740) | .715 (4050) | .680 (4970) | .700 (4510) | .706 |
| 2048 | .700 (4160) | .740 (3340) | .725 (3300) | .705 (3440) | .720 (4920) | .718 |

Best-epoch pattern: n<=512 mostly 100-700 (overfit clock, luck-heavy —
scattered locations); n>=1024 late (3300-5000, near cap, consistent).

## single, rho=.03, 50000 ep (partial, n=32/64) — budget does NOT help small n

n=32: .510@46800 / .520@50000 / .565@49100 / .550@300 / .755@29500(outlier)
      mean .580 < the 5k-budget .604 (5000 evals -> bigger max-bias, no gain)
n=64: .550@38800 / .570@300 / .480@44200 (s45+ interrupted)

Verdict: small-n single saturates ~.6; extra budget only adds selection
noise (the .755@29500 is a lucky eval, not a regime).

## LRao-BE (full-batch, no cutoff, robust IQR, detach, wd=0, no ES,
## best-epoch @eval/10, 5000 ep) — single, from session output

n=32 .657±.03 | n=64 .592 | n=128 .714 | n=256 .807 | n=512 .811 (partial)
(vs published-ES recipe rows: n=32 .129, n=2048 .777; ES was stopping at
epoch 4-12 on small n — val-ES with patience 3/check-every-1 is broken
at small n.)
Best-epoch locations scattered (140-4910) — oracle discount applies.

## AMF singular below n≈128 (n<D=103): pd=nan on most seeds at n=32/64
(pure sample-covariance AMF; the paper's small-n AMF values live on a
different implementation path).
