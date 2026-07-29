"""Paper tables from a spatial run's metrics.json files.

make_tables(out_root, dst) writes table_pavia.tex/.md and
table_sandiego.tex/.md (combined SD1+SD2) with the published column sets.
"""
import json
import os

import numpy as np

ORDER = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR', 'AMF-global',
         'AMF-local', 'GMM-Levin', 'LRao', 'THANTD', 'HTDNet', 'TSTTD',
         'OSVAE']
PAVIA_COLS = ['pauc', 'auc', 'pd05', 'pd_cfar', 'pfa_avg', 'pfa_max',
              'unlab', 'asph', 'trees']
SD_COLS = ['pauc', 'auc', 'pd05', 'pd_cfar', 'pfa']


def _agg(out_root, scn, theta):
    p = os.path.join(out_root, scn, 'metrics.json')
    rows = json.load(open(p))['rows'][str(theta)]
    out = {}
    for det in ORDER:
        vals = {}
        for s in rows:
            if det in rows[s]:
                for k, v in rows[s][det].items():
                    vals.setdefault(k, []).append(v)
        if vals:
            out[det] = {k: float(np.mean(v)) for k, v in vals.items()}
    return out


def _emit(path, header, lines):
    with open(path, 'w') as f:
        f.write('\n'.join([header] + lines) + '\n')
    print('wrote', path)


def make_tables(out_root, dst='tables_out', theta=0.15):
    os.makedirs(dst, exist_ok=True)
    pav = _agg(out_root, 'pavia4', theta)
    md = ['| Detector | ' + ' | '.join(PAVIA_COLS) + ' |',
          '|' + '---|' * (len(PAVIA_COLS) + 1)]
    tex = []
    for d in ORDER:
        if d not in pav:
            continue
        cells = [f'{pav[d].get(c, float("nan")):.3f}' for c in PAVIA_COLS]
        md.append('| ' + d + ' | ' + ' | '.join(cells) + ' |')
        tex.append(d + ' & -- & ' + ' & '.join(cells) + r'\\')
    _emit(os.path.join(dst, 'table_pavia.md'), md[0], md[1:])
    _emit(os.path.join(dst, 'table_pavia.tex'),
          '% pavia4 rows (paper Table 1 column order)', tex)

    sd1 = _agg(out_root, 'sandiego', theta)
    sd2 = _agg(out_root, 'sandiego2', theta)
    md = ['| Detector | ' + ' | '.join('SD1 ' + c for c in SD_COLS)
          + ' | ' + ' | '.join('SD2 ' + c for c in SD_COLS) + ' |',
          '|' + '---|' * (2 * len(SD_COLS) + 1)]
    tex = []
    for d in ORDER:
        if d not in sd1 and d not in sd2:
            continue
        c1 = [f'{sd1.get(d, {}).get(c, float("nan")):.3f}' for c in SD_COLS]
        c2 = [f'{sd2.get(d, {}).get(c, float("nan")):.3f}' for c in SD_COLS]
        md.append('| ' + d + ' | ' + ' | '.join(c1 + c2) + ' |')
        tex.append(d + ' & ' + ' & '.join(c1) + ' & & ' + ' & '.join(c2) + r'\\')
    _emit(os.path.join(dst, 'table_sandiego.md'), md[0], md[1:])
    _emit(os.path.join(dst, 'table_sandiego.tex'),
          '% combined SD1+SD2 rows (paper tab:sandiego column order)', tex)
