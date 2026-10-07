#!/usr/bin/env python3
"""
Generate 24 hourly OD matrices from the Mandl base matrix (synthetic, seeded).

Model:  D_h[i,j] = base[i,j] * s_h * (1 + a_h * (w_j - w_i)) * noise_ij
  s_h   hourly volume profile (peak hour = 1.0 -> 15,570 pax, i.e. the paper's matrix)
  a_h   directional tilt: AM peak pushes trips TOWARD hub stops, PM peak pushes them AWAY
  w_i   hub-ness of stop i in [-1, 1], from the base matrix's row+column totals
  noise lognormal multiplicative noise (small), then rescaled so the hour hits s_h * 15,570

Zero cells of the base matrix stay zero (same OD structure as Mandl, e.g. stop 15 has no demand).
The profile below is an ASSUMED urban two-peak shape, not measured data: replace PROFILE with
local count/ticketing data if you have it, and state this in the report.

Outputs (folder od_matrices/):  od_hour_00.csv ... od_hour_23.csv  and  hourly_summary.csv
"""
import argparse
import os

import numpy as np

from bndfs_q_learning import DEMAND

# Share of the peak-hour volume, hours 00..23 (AM peak 08:00, PM peak 17:00)
PROFILE = np.array([0.03, 0.02, 0.02, 0.02, 0.04, 0.12, 0.35, 0.75, 1.00, 0.70, 0.50, 0.50,
                    0.55, 0.55, 0.55, 0.65, 0.90, 1.00, 0.85, 0.55, 0.40, 0.30, 0.18, 0.08])


def directional_tilt(h, strength=0.5):
    """+ in the AM peak (toward hubs), - in the PM peak (away from hubs), ~0 otherwise."""
    am = np.exp(-((h - 8.0) / 1.5) ** 2)
    pm = np.exp(-((h - 17.5) / 1.5) ** 2)
    return strength * (am - pm)


def generate(seed=7, noise_sigma=0.10, tilt_strength=0.5, base=DEMAND):
    rng = np.random.default_rng(seed)
    activity = base.sum(axis=0) + base.sum(axis=1)             # row + column totals
    w = (activity - activity.mean()) / np.abs(activity - activity.mean()).max()   # [-1, 1]
    mats = []
    for h in range(24):
        a = directional_tilt(h, tilt_strength)
        tilt = np.clip(1.0 + a * (w[None, :] - w[:, None]), 0.05, None)
        noise = rng.lognormal(0.0, noise_sigma, base.shape)
        m = base * tilt * noise
        m *= PROFILE[h] * base.sum() / m.sum()                  # hit the hourly target volume
        m = np.rint(m)
        np.fill_diagonal(m, 0)
        mats.append(m)
    return mats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="od_matrices")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--tilt", type=float, default=0.5, help="AM/PM directional strength (0 = symmetric)")
    p.add_argument("--peak-fleet", type=int, default=99)
    p.add_argument("--min-fleet", type=int, default=15)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    mats = generate(a.seed, a.noise, a.tilt)
    rows = ["hour,total_pax,suggested_fleet,file"]
    for h, m in enumerate(mats):
        f = os.path.join(a.out, f"od_hour_{h:02d}.csv")
        np.savetxt(f, m, fmt="%d", delimiter=",")
        fleet = max(a.min_fleet, int(round(a.peak_fleet * PROFILE[h])))
        rows.append(f"{h},{int(m.sum())},{fleet},{f}")
    with open(os.path.join(a.out, "hourly_summary.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()