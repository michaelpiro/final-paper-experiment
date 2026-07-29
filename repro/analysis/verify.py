"""Verification: compare a fresh spatial run against the published numbers.

Usage:
    from repro.analysis.verify import verify_spatial
    verify_spatial('results/spatial')          # after run_scene() on the scenes
"""
import json
import os

import numpy as np

_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference.json')


def verify_spatial(out_root, table_theta=0.15, tol_free=2e-3):
    ref = json.load(open(_REF))['spatial']
    print(f'=== verification vs published numbers (θ={table_theta}) ===')
    print('Δ = fresh - published.  Training-free rows should be |Δ| < '
          f'{tol_free}; trained rows within seed-level std.')
    for scn, rref in ref.items():
        p = os.path.join(out_root, scn, 'metrics.json')
        if not os.path.exists(p):
            print(f'[{scn}] no fresh metrics — skipped')
            continue
        rows = json.load(open(p))['rows'].get(str(table_theta), {})
        if not rows:
            print(f'[{scn}] θ={table_theta} not in fresh metrics — skipped')
            continue
        print(f'\n[{scn}]')
        print(f'{"detector":12s} {"metric":8s} {"published":>9s} '
              f'{"fresh":>9s} {"Δ":>8s}')
        for det, mref in rref.items():
            fresh = {}
            for s in rows:
                if det in rows[s]:
                    for k, v in rows[s][det].items():
                        fresh.setdefault(k, []).append(v)
            if not fresh:
                print(f'{det:12s} -- missing from fresh run --')
                continue
            for k, vref in mref.items():
                if k.startswith('_') or k not in fresh:
                    continue
                vf = float(np.mean(fresh[k]))
                flag = ''
                if det in ('AMF-global', 'AMF-local', 'GMM-Levin') \
                        and abs(vf - vref) > tol_free:
                    flag = '  <-- CHECK (training-free mismatch)'
                print(f'{det:12s} {k:8s} {vref:9.3f} {vf:9.3f} '
                      f'{vf - vref:+8.3f}{flag}')
