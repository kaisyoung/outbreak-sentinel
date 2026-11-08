"""
Local evaluation harness.

Replicates example_test.py's cost accounting exactly, but sweeps every sample
episode and several missingness seeds, because a single (episode, seed) pair is
far too noisy to tune against -- the never-quarantine cost alone ranges from
1040 to 2720 across the four sample episodes.

Usage:
    python3 evaluate.py                 # score the current solution.py
    python3 evaluate.py --seeds 20      # more seeds, less noise
"""

from __future__ import division, print_function

import argparse
import importlib

import numpy as np
import pandas as pd

import agent as solution

EPISODES = ['data1.csv', 'data2.csv', 'data3.csv', 'data4.csv']

C_EXPOSURE = 40
C_QUARANTINE = 3
P_MISSING = 0.20      # per-reading chance a cheap sensor is missing at test time
PCR_PER_SHIFT = 3     # how many staff get a PCR result each shift

# Reference points used to normalise cost into a 0-50 score.
TOP, BOTTOM = 640, 2070


def build_observations(shift_rows, on_duty, rng):
    """One shift of observations, with test-time missingness injected."""
    staff_obs = {}
    eligible = [int(r.staff) for _, r in shift_rows.iterrows()
                if int(r.staff) in on_duty]
    pcr_people = set(rng.choice(eligible, min(PCR_PER_SHIFT, len(eligible)),
                                replace=False)) if eligible else set()

    for _, r in shift_rows.iterrows():
        i = int(r.staff)
        if i not in on_duty:
            continue

        def maybe(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            return None if rng.random() < P_MISSING else value

        staff_obs[i] = {
            'team': int(r.team),
            'rooms': r.rooms.split(';') if isinstance(r.rooms, str) and r.rooms else [],
            'rapid': maybe(r.rapid),
            'temp': maybe(r.temp),
            'symptom': maybe(r.symptom),
            'pcr': (r.pcr if i in pcr_people else None),
        }
    return staff_obs


def run_episode(data, seed, overrides=None):
    """
    Play one episode and return (total cost, number quarantined).

    `overrides` sets module-level constants after the reload, so tuning sweeps
    do not have to rewrite solution.py between runs.
    """
    rng = np.random.default_rng(seed)
    importlib.reload(solution)          # matches the grading harness
    for name, value in (overrides or {}).items():
        setattr(solution, name, value)

    n_shifts = int(data['shift'].max()) + 1
    all_staff = sorted(int(s) for s in data['staff'].unique())
    truth = data.pivot(index='staff', columns='shift', values='true_status')

    on_duty = set(all_staff)
    quarantined_at = {}
    total_cost = 0

    for t in range(n_shifts):
        shift_rows = data[data['shift'] == t]
        observations = {'shift': t,
                        'staff': build_observations(shift_rows, on_duty, rng)}

        actions = solution.get_action(observations)

        for i in observations['staff']:
            if i not in actions:
                raise Exception(f'No action for staff {i} on shift {t}')
        for i, a in actions.items():
            if a not in ('clear', 'quarantine'):
                raise Exception(f'Invalid action {a!r} for staff {i}')
            if a == 'quarantine' and i in on_duty:
                on_duty.discard(i)
                quarantined_at[i] = t

        for i in all_staff:
            if i in quarantined_at and quarantined_at[i] <= t:
                total_cost += C_QUARANTINE
            elif truth.loc[i, t] == 'I':
                total_cost += C_EXPOSURE

    return total_cost, len(quarantined_at)


def never_quarantine_cost(data):
    """Cost of the baseline policy that clears everyone, for reference."""
    return int((data['true_status'] == 'I').sum()) * C_EXPOSURE


def evaluate(seeds=10, overrides=None, verbose=True, frames=None):
    """Average cost over every episode x seed, plus the implied mark."""
    if frames is None:
        frames = {fn: pd.read_csv(fn) for fn in EPISODES}
    per_episode = {}

    for fn, data in frames.items():
        costs = [run_episode(data, seed, overrides)[0] for seed in range(seeds)]
        per_episode[fn] = float(np.mean(costs))
        if verbose:
            print(f'  {fn}  cost {np.mean(costs):7.1f} '
                  f'(sd {np.std(costs):5.1f}, min {min(costs)}, max {max(costs)})'
                  f'   never-quarantine {never_quarantine_cost(data)}')

    score = float(np.mean(list(per_episode.values())))
    mark = max(0.0, min(50.0, 50.0 * (BOTTOM - score) / (BOTTOM - TOP)))
    if verbose:
        print(f'  average cost {score:.1f}   ->  {mark:.1f} / 50 marks')
    return score, mark


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()
    evaluate(seeds=args.seeds)
