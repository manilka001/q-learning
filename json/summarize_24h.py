#!/usr/bin/env python3
"""
Summarise 24 hourly BNDFS runs (JSON files written by bndfs_q_learning.py --json).

    python summarize_24h.py --dir out_nofill --label nofill [--dir2 out_fill --label2 fill]

Writes, into --out (default 'summary'):
  hourly_<label>.csv     one row per hour: demand, tt, per-passenger times, fleet, plan transition terms
  fig_hourly_<label>.png avg travel/wait time per passenger, fleet, direct %
  fig_stability_<label>.png  plan transition between consecutive hours
  fig_linkheat_<label>.png   route-link service frequency (bus/h) for every hour
Transition terms between hour h-1 and h (routes matched by their stop sequence, direction-insensitive):
  route_changes = |R_h  xor R_{h-1}|          (routes added + removed)
  link_changes  = |L_h  xor L_{h-1}|          (road links gaining/losing service)
  freq_change   = sum over links |F_h - F_{h-1}|  (F = bus/h summed over routes using the link)
  jaccard       = |L_h & L_{h-1}| / |L_h | L_{h-1}|   (1 = same links served)
  delta         = w_route*route_changes + w_link*link_changes + w_freq*freq_change
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        h = int("".join(c for c in os.path.basename(f).split(".")[0][-2:] if c.isdigit()))
        runs[h] = j
    return runs


def canon(r):
    r = tuple(r)
    return min(r, r[::-1])


def links(r):
    return {tuple(sorted(p)) for p in zip(r, r[1:])}


def link_freq(j):
    F = {}
    for r, f in zip(j["routes"], j["freq"]):
        for l in links(r):
            F[l] = F.get(l, 0.0) + f
    return F


def summarise(runs, w):
    rows, prev = [], None
    for h in sorted(runs):
        j = runs[h]; m = j["metrics"]; P = j["total_pax"]
        R = {canon(r) for r in j["routes"]}; F = link_freq(j); L = set(F)
        row = dict(hour=h, pax=P, routes=len(j["routes"]), fleet=m["fleet"], tt=m["tt"],
                   ivt=m["ivt"], wt=m["wt"], trt=m["trt"], tt_per_pax=m["tt"] / P,
                   wt_per_pax=m["wt"] / P, ivt_per_pax=m["ivt"] / P,
                   direct_pct=m["direct_pct"], one_pct=m["one_pct"], unmet_pct=m["unmet_pct"],
                   route_changes="", link_changes="", freq_change="", jaccard="", delta="")
        if prev is not None:
            Rp, Fp = prev
            Lp = set(Fp)
            rc, lc = len(R ^ Rp), len(L ^ Lp)
            fc = sum(abs(F.get(l, 0) - Fp.get(l, 0)) for l in L | Lp)
            row.update(route_changes=rc, link_changes=lc, freq_change=round(fc, 2),
                       jaccard=round(len(L & Lp) / len(L | Lp), 3),
                       delta=round(w[0] * rc + w[1] * lc + w[2] * fc, 2))
        prev = (R, F)
        rows.append(row)
    return rows


def plots(rows, runs, label, out):
    H = [r["hour"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(H, [r["tt_per_pax"] for r in rows], "o-"); ax[0, 0].set_title("Avg travel time per passenger (min)")
    ax[0, 1].plot(H, [r["wt_per_pax"] for r in rows], "o-", c="tab:orange"); ax[0, 1].set_title("Avg waiting time per passenger (min)")
    ax[1, 0].bar(H, [r["fleet"] for r in rows], color="tab:green"); ax[1, 0].set_title("Required fleet (buses)")
    ax[1, 1].plot(H, [r["direct_pct"] for r in rows], "o-", c="tab:red", label="direct %")
    ax[1, 1].plot(H, [r["unmet_pct"] for r in rows], "s-", c="k", label="unmet %"); ax[1, 1].legend()
    ax[1, 1].set_title("Direct-trip and unmet demand (%)")
    ax2 = ax[0, 0].twinx(); ax2.bar(H, [r["pax"] for r in rows], alpha=.15, color="grey"); ax2.set_ylabel("pax (grey)")
    for a in ax.ravel(): a.set_xlabel("hour of day"); a.set_xticks(range(0, 24, 2))
    fig.suptitle(f"Hourly service-plan performance ({label})"); fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_hourly_{label}.png"), dpi=150); plt.close(fig)

    t = [r for r in rows if r["delta"] != ""]
    if t:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        x = [r["hour"] for r in t]
        ax[0].bar(x, [r["route_changes"] for r in t], color="tab:purple"); ax[0].set_title("Routes added+removed vs previous hour")
        ax[1].plot(x, [r["jaccard"] for r in t], "o-"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Link-set Jaccard similarity")
        ax[2].bar(x, [r["delta"] for r in t], color="tab:red"); ax[2].set_title("Transition cost Δ (weighted)")
        for a in ax: a.set_xlabel("hour (transition from h-1 to h)")
        fig.suptitle(f"Plan stability under independent per-hour optimisation ({label})"); fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_stability_{label}.png"), dpi=150); plt.close(fig)

    allL = sorted({l for j in runs.values() for l in link_freq(j)})
    M = np.array([[link_freq(runs[h]).get(l, 0.0) for h in H] for l in allL])
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(allL))); ax.set_yticklabels([f"{a}-{b}" for a, b in allL], fontsize=7)
    ax.set_xticks(range(len(H))); ax.set_xticklabels(H); ax.set_xlabel("hour of day"); ax.set_ylabel("road link")
    fig.colorbar(im, label="bus/h on link (sum over routes)"); ax.set_title(f"Link service frequency by hour ({label})")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_linkheat_{label}.png"), dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True); p.add_argument("--label", default="A")
    p.add_argument("--dir2", default=""); p.add_argument("--label2", default="B")
    p.add_argument("--out", default="summary")
    p.add_argument("--w", type=float, nargs=3, default=[1.0, 0.5, 0.1],
                   help="weights for route_changes, link_changes, freq_change (illustrative)")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    for d, lab in [(a.dir, a.label)] + ([(a.dir2, a.label2)] if a.dir2 else []):
        runs = load(d)
        if not runs: print("no JSON in", d); continue
        rows = summarise(runs, a.w)
        with open(os.path.join(a.out, f"hourly_{lab}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        plots(rows, runs, lab, a.out)
        tr = [r for r in rows if r["delta"] != ""]
        print(f"[{lab}] {len(rows)} hours | mean routes {np.mean([r['routes'] for r in rows]):.1f} | "
              f"mean tt/pax {np.mean([r['tt_per_pax'] for r in rows]):.2f} min | "
              f"mean Jaccard {np.mean([r['jaccard'] for r in tr]):.2f} | mean route changes {np.mean([r['route_changes'] for r in tr]):.1f}")
    print("written to", a.out)


if __name__ == "__main__":
    main()
