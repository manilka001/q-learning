#!/usr/bin/env python3
"""
Tabular Q-Learning for Bus Network Design and Frequency Setting (BNDFS)
on the Mandl Swiss network, after Yoo, Lee & Han (2023), Public Transport 15:503-534.

Structure
---------
1. Network data      : Mandl graph (15 nodes, 21 links) and the 15x15 OD demand matrix
2. NetworkEvaluator  : frequency setting + passenger assignment + reward d_sat / tt
3. QLearningAgent    : state/action spaces, epsilon-greedy, retroactive Q-update
4. run_replication   : one independent replication (fresh Q-table)
5. main()            : runs all replications (in parallel) and prints the best design

Usage
-----
    python bndfs_q_learning.py                       # 5 x 10,000 iterations, max 8 stops/route
    python bndfs_q_learning.py --max-stops 15
    python bndfs_q_learning.py --iterations 500 --replications 2 --workers 1   # smoke test

Internally nodes are 0-based; everything printed is 1-based (stops 1..15).
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool

import numpy as np

# =============================================================================
# 1. NETWORK DATA
# =============================================================================
N_NODES = 15

# (stop_a, stop_b): in-vehicle minutes. Links are bidirectional, same time both ways.
LINKS = {
    (1, 2): 8, (2, 3): 2, (2, 4): 3, (2, 5): 6, (3, 6): 3, (4, 5): 4, (4, 6): 4,
    (4, 12): 10, (6, 8): 2, (6, 15): 3, (7, 10): 7, (7, 15): 2, (8, 10): 8,
    (8, 15): 2, (9, 15): 8, (10, 11): 5, (10, 13): 10, (10, 14): 8, (11, 12): 10,
    (11, 13): 5, (13, 14): 2,
}

# Table 1 of the paper (rows = origin 1..15, cols = destination 1..15); total 15,570.
DEMAND = np.array([
    [0, 400, 200, 60, 80, 150, 75, 75, 30, 160, 30, 25, 35, 0, 0],
    [400, 0, 50, 120, 20, 180, 90, 90, 15, 130, 20, 10, 10, 5, 0],
    [200, 50, 0, 40, 60, 180, 90, 90, 15, 45, 20, 10, 10, 5, 0],
    [60, 120, 40, 0, 50, 100, 50, 50, 15, 240, 40, 25, 10, 5, 0],
    [80, 20, 60, 50, 0, 50, 25, 25, 10, 120, 20, 15, 5, 0, 0],
    [150, 180, 180, 100, 50, 0, 100, 100, 30, 880, 60, 15, 15, 10, 0],
    [75, 90, 90, 50, 25, 100, 0, 50, 15, 440, 35, 10, 10, 5, 0],
    [75, 90, 90, 50, 25, 100, 50, 0, 15, 440, 35, 10, 10, 5, 0],
    [30, 15, 15, 15, 10, 30, 15, 15, 0, 140, 20, 5, 0, 0, 0],
    [160, 130, 45, 240, 120, 880, 440, 440, 140, 0, 600, 250, 500, 200, 0],
    [30, 20, 20, 40, 20, 60, 35, 35, 20, 600, 0, 75, 95, 15, 0],
    [25, 10, 10, 25, 15, 15, 10, 10, 5, 250, 75, 0, 70, 0, 0],
    [35, 10, 10, 10, 5, 15, 10, 10, 0, 500, 95, 70, 0, 45, 0],
    [0, 5, 5, 5, 0, 10, 5, 5, 0, 200, 15, 0, 45, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)
assert DEMAND.sum() == 15570, "OD matrix must total 15,570 passengers"

# 0-based adjacency list and link-time lookup
ADJ = [[] for _ in range(N_NODES)]
LINK_TIME = [[math.inf] * N_NODES for _ in range(N_NODES)]
for (_a, _b), _t in LINKS.items():
    _a, _b = _a - 1, _b - 1
    ADJ[_a].append(_b)
    ADJ[_b].append(_a)
    LINK_TIME[_a][_b] = LINK_TIME[_b][_a] = float(_t)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Config:
    # --- operational parameters (Section 6.1 of the paper) ---
    max_fleet: int = 99               # maximum fleet
    seats: int = 40                   # C_k
    load_factor: float = 1.25         # delta_max  -> capacity 50 pax/bus
    min_stops: int = 2
    max_stops: int = 8                # 8 or 15 in the paper
    max_routes: int = 6               # paper allows up to N_NODES (15); capped here to avoid bloat
    forbid_redundant: bool = True     # reject duplicate / sub-path routes (reward 0)
    fill_fleet: bool = True           # allocate the whole fleet (all designs in Table 2 total 99)
    min_freq: float = 1.0             # buses/hour on any active route
    transfer_penalty: float = 5.0     # minutes per transfer
    direct_tol: float = 0.50          # reject direct paths > 50% above best direct IVT
    transfer_tol: float = 0.10        # keep 1-transfer paths within 10% of best IVT
    freq_passes: int = 2              # frequency <-> assignment fixed-point passes
    # --- learning parameters ---
    demand_file: str = ""            # CSV 15x15 OD matrix (default: Mandl Table 1)
    n_iterations: int = 10_000
    n_replications: int = 5
    cache_size: int = 150_000         # per-process evaluation cache (entries)
    # --- NOT in the paper's reward: minutes of penalty per unserved passenger added to tt.
    #     60 keeps the agent from "winning" by serving only a sliver of demand (paper Sec. 4 has
    #     a -alpha*unmet term). Set 0.0 for the literal d_sat/tt reward.
    unmet_penalty: float = 60.0
    # Best-network selection prefers designs that serve all demand (paper: "best network has
    # the shortest total travel time without unmet transit demand").
    require_full_coverage: bool = True


# =============================================================================
# 2. NETWORK EVALUATOR: frequency setting, passenger assignment, reward
# =============================================================================
class NetworkEvaluator:
    """Evaluates a set of bus routes (each a tuple of 0-based stops, >= 2 stops)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        D = np.loadtxt(cfg.demand_file, delimiter=",") if cfg.demand_file else DEMAND
        assert D.shape == (N_NODES, N_NODES), "OD matrix must be 15x15"
        self.od = [(a, b, float(D[a, b]))
                   for a in range(N_NODES) for b in range(N_NODES)
                   if a != b and D[a, b] > 0]
        self.total_demand = float(D.sum())
        self._cache = {}

    # ---- public API ---------------------------------------------------------
    def reward(self, routes):
        """Cached (reward, unmet_fraction) for a route set; order/direction independent."""
        valid = [r for r in routes if len(r) >= 2]
        if not valid:
            return 0.0, 1.0
        key = tuple(sorted(min(r, r[::-1]) for r in valid))
        hit = self._cache.get(key)
        if hit is None:
            m = self.compute(valid)
            hit = (m["reward"], m["unmet_frac"])
            if len(self._cache) >= self.cfg.cache_size:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def compute(self, routes):
        """Full evaluation of a route set. Returns a dict of metrics."""
        cfg = self.cfg
        routes = [tuple(r) for r in routes]
        K = len(routes)
        if self.cfg.forbid_redundant and self._has_redundant(routes):
            return self._result(0, 0, 0, 0, 0, 0, self.total_demand,
                                np.ones(K), np.ones(K), False, K)

        # --- cumulative in-vehicle time along each route; round-trip time T_k
        cum, rp = [], [dict() for _ in range(N_NODES)]   # rp[node] = {route: position}
        for k, r in enumerate(routes):
            c = [0.0]
            for u, v in zip(r, r[1:]):
                c.append(c[-1] + LINK_TIME[u][v])
            cum.append(c)
            for i, n in enumerate(r):
                rp[n][k] = i
        T = np.array([2.0 * c[-1] for c in cum])
        lens = [len(r) for r in routes]
        reach = [set().union(*[routes[k] for k in rp[n]]) if rp[n] else set()
                 for n in range(N_NODES)]

        # --- candidate paths per OD pair (independent of frequencies)
        struct, unmet = [], 0.0
        for a, b, d in self.od:
            common = rp[a].keys() & rp[b].keys()
            if common:                                        # 0-transfer
                cand = [(k, rp[a][k], rp[b][k], abs(cum[k][rp[a][k]] - cum[k][rp[b][k]]))
                        for k in common]
                best = min(c[3] for c in cand)
                cand = [c for c in cand if c[3] <= best * (1 + cfg.direct_tol) + 1e-9]
                struct.append((0, d, cand))
                continue
            paths = []                                        # 1-transfer
            for c in (reach[a] & reach[b]) - {a, b}:
                rc = rp[c]
                for k1, pa in rp[a].items():
                    if k1 not in rc:
                        continue
                    pc1 = rc[k1]
                    iv1 = abs(cum[k1][pa] - cum[k1][pc1])
                    for k2, pb in rp[b].items():
                        if k2 != k1 and k2 in rc:
                            pc2 = rc[k2]
                            iv2 = abs(cum[k2][pc2] - cum[k2][pb])
                            paths.append((iv1 + iv2, k1, pa, pc1, iv1, c, k2, pc2, pb, iv2))
            if not paths:
                unmet += d                                    # no 0/1-transfer path
                continue
            best = min(p[0] for p in paths)
            groups = {}                                       # first leg (k1, c) -> second legs
            for tot, k1, pa, pc1, iv1, c, k2, pc2, pb, iv2 in paths:
                if tot <= best * (1 + cfg.transfer_tol) + 1e-9:
                    g = groups.setdefault((k1, c), [k1, pa, pc1, iv1, []])
                    g[4].append((k2, pc2, pb, iv2))
            struct.append((1, d, list(groups.values())))

        # --- frequency setting <-> assignment fixed point
        f = np.ones(K)
        feasible = True
        for _ in range(cfg.freq_passes):
            res = self._assign(struct, f.tolist(), lens)
            f, feasible = self._frequencies(res["qmax"], T)
            if not feasible:
                break
        if not feasible:
            return self._result(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unmet, f, T, False, K)
        res = self._assign(struct, f.tolist(), lens)
        return self._result(res["ivt"], res["wt"], res["trt"], res["direct"], res["one"],
                            0.0, unmet, f, T, True, K)

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _is_subpath(a, b):
        """True if route a is a contiguous sub-path of route b (either direction)."""
        la = len(a)
        for c in (b, b[::-1]):
            for i in range(len(c) - la + 1):
                if c[i:i + la] == a:
                    return True
        return False

    def _has_redundant(self, routes):
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i != j and len(routes[i]) <= len(routes[j]) \
                        and self._is_subpath(routes[i], routes[j]):
                    return True
        return False

    @staticmethod
    def _add_flow(ff, fb, k, i, j, q):
        if i < j:
            ff[k][i:j] += q
        else:
            fb[k][j:i] += q

    def _assign(self, struct, f, lens):
        """Assign all demand given frequencies f (buses/h); accumulate directional link flows."""
        cfg, half = self.cfg, 30.0              # T/2 with T = 60 min -> 30/f minutes of waiting
        ff = [np.zeros(n - 1) for n in lens]
        fb = [np.zeros(n - 1) for n in lens]
        ivt = wt = trt = direct = one = 0.0
        add = self._add_flow
        for kind, d, data in struct:
            if kind == 0:                                       # direct trip(s)
                tot = sum(f[c[0]] for c in data)
                wt += d * half / tot
                direct += d
                for k, pa, pb, iv in data:
                    q = d * f[k] / tot
                    ivt += q * iv
                    add(ff, fb, k, pa, pb, q)
            else:                                               # one-transfer trip(s)
                groups = data
                tot1 = sum(f[g[0]] for g in groups)             # eq. (12): share ~ frequency
                dist1 = sum(f[k1] for k1 in {g[0] for g in groups})
                wt += d * half / dist1                          # wait at origin
                trt += d * cfg.transfer_penalty
                one += d
                for k1, pa, pc1, iv1, legs in groups:
                    q1 = d * f[k1] / tot1
                    ivt += q1 * iv1
                    add(ff, fb, k1, pa, pc1, q1)
                    tot2 = sum(f[l[0]] for l in legs)           # eq. (14)
                    wt += q1 * half / tot2                      # wait at transfer stop
                    for k2, pc2, pb, iv2 in legs:
                        q2 = q1 * f[k2] / tot2
                        ivt += q2 * iv2
                        add(ff, fb, k2, pc2, pb, q2)
        qmax = np.array([max(ff[k].max(), fb[k].max()) for k in range(len(lens))])
        return dict(ivt=ivt, wt=wt, trt=trt, direct=direct, one=one, qmax=qmax)

    def _frequencies(self, qmax, T):
        """f_k = Qmax/(delta*C) (eq. 9); fleet v_k = f_k*T_k/60 (eq. 10).

        Each active route needs >= min_freq buses/h and >= 1 vehicle (eq. 7).  If sum(v_k) >
        max_fleet frequencies are rescaled; if fill_fleet, they are scaled up to use the whole
        fleet.  Fleets are then rounded to integers (largest remainder) and f_k = v_k*60/T_k.
        """
        cfg = self.cfg
        base = T / 60.0                                     # fleet needed per 1 bus/h
        fmin = np.maximum(cfg.min_freq, 1.0 / base)
        vlow = np.ceil(fmin * base - 1e-9)                  # integer lower bound per route
        if vlow.sum() > cfg.max_fleet:
            return fmin, False                              # infeasible even at minimum service
        f = np.maximum(qmax / (cfg.load_factor * cfg.seats), fmin)
        if (f * base).sum() > cfg.max_fleet + 1e-9:         # rescale down to the fleet limit
            fixed = np.zeros(len(f), dtype=bool)
            while True:
                free = ~fixed
                if not free.any():
                    break
                budget = cfg.max_fleet - (vlow[fixed]).sum()
                scale = budget / (f[free] * base[free]).sum()
                cand = f[free] * scale
                low = cand * base[free] < vlow[free]
                if not low.any():
                    f = f.copy()
                    f[free] = cand
                    break
                fixed[np.where(free)[0][low]] = True
            f[fixed] = vlow[fixed] / base[fixed]
        v = f * base
        target = cfg.max_fleet if cfg.fill_fleet else min(cfg.max_fleet, int(math.ceil(v.sum() - 1e-9)))
        if cfg.fill_fleet:
            v = v * (cfg.max_fleet / v.sum())               # use the entire fleet
        vi = np.maximum(vlow, np.floor(v + 1e-9))
        while vi.sum() < target:                            # hand out remaining buses
            vi[np.argmax(v - vi)] += 1
        while vi.sum() > target:                            # (rare) take back extra buses
            cand = np.where(vi > vlow)[0]
            if len(cand) == 0:
                break
            vi[cand[np.argmin((v - vi)[cand])]] -= 1
        return vi / base, True

    def _result(self, ivt, wt, trt, direct, one, _unused, unmet, f, T, feasible, K):
        tt = ivt + wt + trt + self.cfg.unmet_penalty * unmet
        served = self.total_demand - unmet
        d_sat = served / self.total_demand
        reward = d_sat / tt if (feasible and tt > 0) else 0.0
        fleet_cont = f * T / 60.0
        return dict(
            reward=reward, feasible=feasible, tt=tt, ivt=ivt, wt=wt, trt=trt,
            d_sat=d_sat, unmet_frac=1.0 - d_sat if feasible else 1.0,
            direct_pct=100 * direct / self.total_demand,
            one_pct=100 * one / self.total_demand,
            unmet_pct=100 * unmet / self.total_demand if feasible else 100.0,
            freq=f, route_time=T, fleet_per_route=fleet_cont,
            fleet=int(math.ceil(fleet_cont.sum() - 1e-9)) if feasible else None,
        )


# =============================================================================
# 3. Q-LEARNING AGENT
# =============================================================================
# State  : tuple of routes (tuple of stops); the LAST route is the one being built.
# Actions: ('add', n)  append adjacent stop n to the current route   (priority 0)
#          ('new', n)  close current route, start a new one at n     (priority 1)
#          ('end',)    terminate the design iteration (goal state)   (priority 2)
PRIORITY = {"add": 0, "new": 1, "end": 2}


class QLearningAgent:
    def __init__(self, cfg: Config, evaluator: NetworkEvaluator, rng: random.Random):
        self.cfg, self.ev, self.rng = cfg, evaluator, rng
        self.Q = {}                      # (state, action) -> Q-value ; default 0

    # ---- action space -------------------------------------------------------
    def available_actions(self, state):
        cfg = self.cfg
        if not state:                                       # first stop of first route
            return [("new", n) for n in range(N_NODES)]
        cur = state[-1]
        acts = []
        if len(cur) < cfg.max_stops:                        # extend with adjacent stop
            acts += [("add", n) for n in ADJ[cur[-1]] if n not in cur]
        if len(cur) >= cfg.min_stops:                       # route may be closed
            if len(state) < cfg.max_routes:
                acts += [("new", n) for n in range(N_NODES)]
            acts.append(("end",))
        return acts

    @staticmethod
    def step(state, action):
        if action[0] == "add":
            return state[:-1] + (state[-1] + (action[1],),)
        return state + ((action[1],),)                      # 'new'

    # ---- policy -------------------------------------------------------------
    def greedy(self, state, acts):
        """Max Q; ties broken by priority add > new > end, then randomly."""
        Q = self.Q
        best_key, best = None, []
        for a in acts:
            key = (-Q.get((state, a), 0.0), PRIORITY[a[0]])
            if best_key is None or key < best_key:
                best_key, best = key, [a]
            elif key == best_key:
                best.append(a)
        return self.rng.choice(best)

    # ---- one design iteration (episode) ---------------------------------------
    def run_episode(self, epsilon):
        Q, state, traj, last_r = self.Q, (), [], 0.0
        while True:
            acts = self.available_actions(state)
            a = self.rng.choice(acts) if self.rng.random() < epsilon else self.greedy(state, acts)
            if a[0] == "end":
                Q[(state, a)] = last_r          # Q(s, END) = reward just before END
                return state, last_r
            nxt = self.step(state, a)
            r, _ = self.ev.reward(nxt)          # route sets with < 2 stops are ignored -> r = 0
            traj.append((state, a))
            # Q_new = Q_old + alpha (r - Q_old), alpha = 1 if r > Q_old else 0, applied
            # retroactively to every (s, a) already taken in this iteration.
            for key in traj:
                if r > Q.get(key, 0.0):
                    Q[key] = r
            state, last_r = nxt, r


# =============================================================================
# 4. REPLICATION DRIVER
# =============================================================================
def _score(cfg, reward, unmet):
    """Sort key for 'best network': full coverage first (optional), then reward."""
    covered = 1 if (unmet <= 1e-12 or not cfg.require_full_coverage) else 0
    return (covered, reward)


def run_replication(args):
    cfg, rep, seed = args
    rng = random.Random(seed)
    ev = NetworkEvaluator(cfg)
    agent = QLearningAgent(cfg, ev, rng)
    best_routes, best_score, best_reward = None, (-1, -1.0), 0.0
    t0 = time.time()
    for k in range(cfg.n_iterations):
        eps = (cfg.n_iterations - k) / cfg.n_iterations     # eq. (19): 1.0 -> ~0.0
        terminal, _ = agent.run_episode(eps)
        r, unmet = ev.reward(terminal)
        sc = _score(cfg, r, unmet)
        if terminal and sc > best_score:
            best_score, best_routes, best_reward = sc, terminal, r
        if (k + 1) % max(1, cfg.n_iterations // 10) == 0:
            print(f"  [rep {rep + 1}] iter {k + 1:>6}/{cfg.n_iterations}  eps={eps:.2f}  "
                  f"best reward={best_reward:.4e}  |Q|={len(agent.Q):,}  "
                  f"{time.time() - t0:,.0f}s", flush=True)
    return best_routes, best_score, best_reward


# =============================================================================
# 4b. IMAGE OUTPUT: network graph with the routes + service-plan table + metrics
# =============================================================================
# Approximate stop layout of Mandl's network (Fig. 7 of the paper), 1-based stop -> (x, y)
NODE_XY = {1: (0.0, 5.2), 2: (1.0, 4.4), 3: (2.4, 4.4), 4: (1.0, 3.2), 5: (0.0, 2.7),
           6: (2.4, 3.2), 7: (5.2, 1.8), 8: (3.6, 2.4), 9: (6.4, 4.0), 10: (3.8, 1.2),
           11: (2.4, 0.2), 12: (1.0, 1.1), 13: (4.2, -0.4), 14: (5.2, 0.2), 15: (5.2, 3.2)}


def save_plan_image(routes, m, cfg, path, title=""):
    """Draw routes on the Mandl graph (one colour per route, parallel offset lines on shared
    links) next to a service-plan table and the performance metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.35, 1],
                          left=0.03, right=0.98, top=0.92, bottom=0.04, wspace=0.08, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    # --- background road network with link times
    for (a, b), t in LINKS.items():
        (x1, y1), (x2, y2) = NODE_XY[a], NODE_XY[b]
        ax.plot([x1, x2], [y1, y2], color="#d0d0d0", lw=2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{t}", fontsize=8, color="#888888", ha="center",
                va="center", zorder=4, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    # --- routes: offset parallel lines where several routes share a link
    users = {}
    for k, r in enumerate(routes):
        for u, v in zip(r, r[1:]):
            users.setdefault(tuple(sorted((u + 1, v + 1))), []).append(k)
    gap = 0.075
    for (a, b), ks in users.items():
        (x1, y1), (x2, y2) = NODE_XY[a], NODE_XY[b]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        for i, k in enumerate(ks):
            off = (i - (len(ks) - 1) / 2) * gap
            ax.plot([x1 + nx * off, x2 + nx * off], [y1 + ny * off, y2 + ny * off],
                    color=colors[k % 10], lw=3.2, solid_capstyle="round", zorder=3)
    for n, (x, y) in NODE_XY.items():
        ax.scatter([x], [y], s=620, c="white", edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)
    ax.legend(handles=[Line2D([0], [0], color=colors[k % 10], lw=4,
                              label=f"Route {k + 1}: {fmt_route(r)}") for k, r in enumerate(routes)],
              loc="lower left", fontsize=9, frameon=True, title="Routes (both directions)")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bus network design on Mandl's Swiss network (grey numbers = link minutes)",
                 fontsize=12)

    # --- service-plan table
    ax_t.axis("off")
    rows = [[f"{k + 1}", fmt_route(r), f"{len(r)}", f"{m['route_time'][k] / 2:.0f}",
             f"{m['route_time'][k]:.0f}", f"{m['freq'][k]:.2f}", f"{60 / m['freq'][k]:.1f}",
             f"{m['fleet_per_route'][k]:.0f}"] for k, r in enumerate(routes)]
    rows.append(["", "TOTAL", "", "", "", "", "", f"{m['fleet']}"])
    tbl = ax_t.table(cellText=rows,
                     colLabels=["Route", "Stops (sequence)", "#Stops", "One-way\n(min)",
                                "Round trip\n(min)", "Freq\n(bus/h)", "Headway\n(min)", "Fleet\n(buses)"],
                     loc="upper center", cellLoc="center",
                     colWidths=[0.07, 0.38, 0.07, 0.09, 0.1, 0.09, 0.1, 0.1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)
    for k in range(len(routes)):
        tbl[(k + 1, 0)].set_facecolor(colors[k % 10])
        tbl[(k + 1, 0)].get_text().set_color("white")
    for c in range(8):
        tbl[(0, c)].set_facecolor("#333333")
        tbl[(0, c)].get_text().set_color("white")
        tbl[(len(rows), c)].set_facecolor("#eeeeee")
    ax_t.set_title("Service plan", fontsize=12)

    # --- metrics
    ax_m.axis("off")
    lines = [
        ("Total travel time", f"{m['tt']:,.0f} min"),
        ("  In-vehicle time", f"{m['ivt']:,.0f} min"),
        ("  Waiting time", f"{m['wt']:,.0f} min"),
        ("  Transfer time", f"{m['trt']:,.0f} min"),
        ("Direct trips", f"{m['direct_pct']:.2f} %"),
        ("One-transfer trips", f"{m['one_pct']:.2f} %"),
        ("Unmet demand", f"{m['unmet_pct']:.2f} %"),
        ("Required fleet", f"{m['fleet']} buses (limit {cfg.max_fleet})"),
        ("Reward (d_sat / tt)", f"{m['reward']:.4e}"),
    ]
    for i, (a, b) in enumerate(lines):
        y = 0.93 - i * 0.1
        ax_m.text(0.05, y, a, fontsize=11, transform=ax_m.transAxes, va="top")
        ax_m.text(0.95, y, b, fontsize=11, transform=ax_m.transAxes, va="top", ha="right",
                  fontweight="bold")
    ax_m.set_title("Performance metrics", fontsize=12)
    fig.suptitle(title or "BNDFS Q-learning result", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =============================================================================
# 5. MAIN
# =============================================================================
def fmt_route(r):
    return "-".join(str(n + 1) for n in r)


def main():
    p = argparse.ArgumentParser(description="Tabular Q-learning for BNDFS on Mandl's network")
    p.add_argument("--max-stops", type=int, default=8, help="max stops per route (8 or 15)")
    p.add_argument("--iterations", type=int, default=10_000)
    p.add_argument("--replications", type=int, default=5)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--workers", type=int, default=0, help="processes (0 = min(reps, CPUs))")
    p.add_argument("--image", default="bndfs_best_network.png", help="output PNG ('' to disable)")
    p.add_argument("--demand-file", default="", help="CSV with a 15x15 OD matrix")
    p.add_argument("--max-fleet", type=int, default=99, help="fleet limit (buses)")
    p.add_argument("--max-routes", type=int, default=6, help="cap on number of routes (paper: 15)")
    p.add_argument("--allow-redundant", action="store_true", help="allow duplicate/sub-path routes")
    p.add_argument("--no-fill-fleet", action="store_true", help="do not scale frequencies up to 99 buses")
    p.add_argument("--unmet-penalty", type=float, default=60.0,
                   help="minutes of penalty per unserved passenger added to tt (default: none)")
    args = p.parse_args()

    cfg = replace(Config(), max_stops=args.max_stops, n_iterations=args.iterations,
                  n_replications=args.replications, unmet_penalty=args.unmet_penalty,
                  max_routes=args.max_routes, forbid_redundant=not args.allow_redundant,
                  fill_fleet=not args.no_fill_fleet,
                  demand_file=args.demand_file, max_fleet=args.max_fleet)
    workers = args.workers or min(cfg.n_replications, os.cpu_count() or 1)
    jobs = [(cfg, i, args.seed + 1000 * i) for i in range(cfg.n_replications)]

    print(f"BNDFS Q-learning | max stops/route={cfg.max_stops} | "
          f"{cfg.n_replications} replications x {cfg.n_iterations:,} iterations | workers={workers}")
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(run_replication, jobs)
    else:
        results = [run_replication(j) for j in jobs]

    for i, (routes, sc, r) in enumerate(results):
        print(f"replication {i + 1}: reward={r:.4e}  routes={len(routes) if routes else 0}")
    best_routes = max(results, key=lambda x: x[1])[0]
    if not best_routes:
        print("No valid network found - increase iterations.")
        return

    m = NetworkEvaluator(cfg).compute(list(best_routes))
    print("\n" + "=" * 70)
    print(f"BEST NETWORK  ({len(best_routes)} routes, max {cfg.max_stops} stops/route)")
    print("=" * 70)
    for k, r in enumerate(best_routes):
        print(f"  Route {k + 1}: {fmt_route(r):<36} f={m['freq'][k]:6.2f} bus/h   "
              f"fleet={m['fleet_per_route'][k]:6.2f}   round trip={m['route_time'][k]:.0f} min")
    print("-" * 70)
    print(f"  Total Travel Time (min) : {m['tt']:>12,.0f}")
    print(f"  In-Vehicle Time (min)   : {m['ivt']:>12,.0f}")
    print(f"  Waiting Time (min)      : {m['wt']:>12,.0f}")
    print(f"  Transfer Time (min)     : {m['trt']:>12,.0f}")
    print(f"  Direct Trips (%)        : {m['direct_pct']:>12.2f}")
    print(f"  One-Transfer Trips (%)  : {m['one_pct']:>12.2f}")
    print(f"  Unmet Demand (%)        : {m['unmet_pct']:>12.2f}")
    print(f"  Required Fleet Size     : {m['fleet']!s:>12}")
    print(f"  Reward (d_sat / tt)     : {m['reward']:>12.4e}")
    if args.image:
        label = os.path.basename(args.demand_file) if args.demand_file else "Mandl base demand"
        title = (f"BNDFS Q-learning | {label} | max {cfg.max_stops} stops/route | "
                 f"fleet limit {cfg.max_fleet} | {len(best_routes)} routes")
        print(f"\nImage saved to: {save_plan_image(best_routes, m, cfg, args.image, title)}")
    print(f"\nTotal wall time: {time.time() - t0:,.0f} s")
    print("Paper reference (15 stops): TT=198,273  IVT=168,306  WT=26,767  TR=3,200  "
          "Direct=95.89%  One-transfer=4.11%  Fleet=99")


if __name__ == "__main__":
    main()
