"""
Offline parameter estimation for the containment-breach DBN.

The training episodes are fully observed -- every sensor column and the ground
truth status are present for every staff member on every shift -- so maximum
likelihood reduces to counting.  The only wrinkles are:

  * PCR is reported with a one-shift delay, so it must be aligned against the
    PREVIOUS shift's status (pcr_delay) rather than the current one.
  * Several conditioning cells are tiny (state R, high contact counts), so the
    raw MLE produces degenerate 0/1 probabilities.  Every categorical is
    smoothed with a symmetric Dirichlet prior over its full value domain.

Run directly to write parameter.json, which solution.py loads at module level:

    python3 learn_parameters.py
"""

from __future__ import division, print_function

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

FILES = ["data1.csv", "data2.csv", "data3.csv", "data4.csv"]

STATES = ["S", "E", "I", "R"]

# Full value domain of every categorical, so that smoothing can reach values
# that happen never to co-occur with a rare state in the sample episodes.
VALUES = {
    "rapid": ["negative", "positive"],
    "temp": ["normal", "fever"],
    "symptom": ["none", "mild", "severe"],
    "pcr": ["negative", "positive"],
    "pcr_delay": ["negative", "positive"],
}

ALPHA = 1.0          # Dirichlet pseudocount per cell
MAX_CONTACTS = 5     # contact-count buckets to accumulate


def norm(counter, domain, alpha=ALPHA):
    """Normalise a Counter into a distribution, Laplace-smoothed over `domain`."""
    total = sum(counter.get(k, 0) for k in domain) + alpha * len(domain)
    return {k: (counter.get(k, 0) + alpha) / total for k in domain}


def room_map(df):
    """shift -> room -> [(staff, true_status)] for one episode."""
    m = defaultdict(lambda: defaultdict(list))
    for _, r in df.iterrows():
        for room in str(r['rooms']).split(";"):
            if room:
                m[int(r['shift'])][room].append((int(r['staff']), r['true_status']))
    return m


def contact_beta(table, contact_counts, fallback=0.08):
    """
    Pool the contact buckets into one per-contact transmission probability.

    Under independent exposure P(infection | k) = 1 - (1-beta)^k, so each bucket
    yields an estimate of beta; we average them weighted by how many susceptible
    person-shifts each bucket was estimated from.  This is both more stable than
    the raw table -- the high-k buckets rest on a handful of samples, and k=5
    comes out as a degenerate 1.0 unsmoothed -- and it extrapolates beyond the
    largest bucket ever observed.
    """
    num = den = 0.0
    for k_str, p in table.items():
        k = int(k_str)
        if k <= 0 or not (0.0 < p < 1.0):
            continue
        weight = contact_counts[k][1]
        num += weight * (1.0 - (1.0 - p) ** (1.0 / k))
        den += weight
    return num / den if den else fallback


def learn(files=FILES, quiet=False):
    """Count sufficient statistics over the given episodes and return params."""
    prior = Counter()
    transition = defaultdict(Counter)
    emissions = {name: defaultdict(Counter)
                 for name in ('rapid', 'temp', 'symptom', 'pcr')}
    pcr_delay = defaultdict(Counter)
    contact_counts = defaultdict(lambda: [0, 0])   # exposed, total

    for fn in files:
        if not Path(fn).exists():
            if not quiet:
                print(f"Skip {fn}")
            continue
        df = pd.read_csv(fn)
        rm = room_map(df)
        prior.update(df[df['shift'] == 0]['true_status'])

        for _, r in df.iterrows():
            status = r['true_status']
            # Guard against missing readings so NaN never becomes a category.
            for name, table in emissions.items():
                if pd.notna(r[name]):
                    table[status][r[name]] += 1

        for _, g in df.groupby("staff"):
            rows = [r for _, r in g.sort_values("shift").iterrows()]

            for row, nxt in zip(rows, rows[1:]):
                transition[row["true_status"]][nxt["true_status"]] += 1

                # PCR reported on the next shift describes THIS shift's status.
                if pd.notna(nxt["pcr"]):
                    pcr_delay[row["true_status"]][nxt["pcr"]] += 1

                # How often does a susceptible person with k infectious
                # room-mates this shift turn up exposed on the next one?
                if row["true_status"] == "S":
                    infectious = set()
                    for room in str(row["rooms"]).split(";"):
                        if not room:
                            continue
                        for other, status in rm[int(row["shift"])][room]:
                            if other != row["staff"] and status == "I":
                                infectious.add(other)

                    k = min(len(infectious), MAX_CONTACTS)
                    contact_counts[k][1] += 1
                    if nxt["true_status"] == "E":
                        contact_counts[k][0] += 1

    if not prior:
        raise SystemExit(
            "No episode data found -- refusing to overwrite parameter.json with "
            "an empty model. Put data1.csv..data4.csv in this directory first.")

    contact_table = {str(k): (v[0] + ALPHA) / (v[1] + 2 * ALPHA)
                     for k, v in sorted(contact_counts.items())}

    params = {
        "prior": norm(prior, STATES),
        "transition": {k: norm(v, STATES) for k, v in transition.items()},
        "pcr_delay": {k: norm(v, VALUES["pcr_delay"]) for k, v in pcr_delay.items()},
        "contact_infection": contact_table,
        "contact_beta": contact_beta(contact_table, contact_counts),
    }
    for name, table in emissions.items():
        params[name] = {k: norm(v, VALUES[name]) for k, v in table.items()}

    return params


def main():
    params = learn()
    with open("parameter.json", "w") as f:
        json.dump(params, f, indent=2)
    print("parameter.json written.")
    print("  contact_beta =", round(params["contact_beta"], 4))


if __name__ == '__main__':
    main()
