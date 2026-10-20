"""
HOTA and MOTA (CLEAR-MOT) for 2D/3D point-based tracking.  Needs only numpy + scipy.

A *sequence* is a list of frames.  Each frame is a tuple:
    (gt_ids, gt_xy, pred_ids, pred_xy)
gt_ids / pred_ids : int arrays, dense ids starting at 0 *within the sequence*
gt_xy / pred_xy   : float arrays (N, 2) or (N, 3) of box centres (metres)

Matching uses centre distance (nuScenes convention).
  - HOTA similarity  : sim = max(0, 1 - dist / dmax)      (alphas 0.05 ... 0.95)
  - MOTA match rule  : dist < thr                          (nuScenes uses 2 m)

HOTA follows TrackEval's implementation (global-alignment Hungarian matching,
per-alpha DetA/AssA, sequences combined by TP-weighted averaging).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = np.finfo(float).eps
ALPHAS = np.arange(0.05, 0.99, 0.05)


def pairwise_dist(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


# --------------------------------------------------------------------------- HOTA
def hota_sequence(frames, dmax=2.0):
    """Per-sequence HOTA counts. Returns a dict of arrays of length len(ALPHAS)."""
    n_gt = max([int(g.max()) for g, _, _, _ in frames if len(g)] + [-1]) + 1
    n_pr = max([int(p.max()) for _, _, p, _ in frames if len(p)] + [-1]) + 1
    pot = np.zeros((n_gt, n_pr))
    gt_cnt = np.zeros(n_gt)
    pr_cnt = np.zeros(n_pr)
    sims = []

    # pass 1: global alignment score between every (gt id, pred id) pair
    for g_ids, g_xy, p_ids, p_xy in frames:
        sim = np.maximum(0.0, 1.0 - pairwise_dist(g_xy, p_xy) / dmax)
        sims.append(sim)
        if sim.size:
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            m = denom > EPS
            sim_iou[m] = sim[m] / denom[m]
            pot[g_ids[:, None], p_ids[None, :]] += sim_iou
        gt_cnt[g_ids] += 1
        pr_cnt[p_ids] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        glob = pot / (gt_cnt[:, None] + pr_cnt[None, :] - pot)
    glob = np.nan_to_num(glob)

    A = len(ALPHAS)
    TP, FN, FP, loc = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    mc = [np.zeros((n_gt, n_pr)) for _ in range(A)]

    # pass 2: per-frame matching, weighted by global alignment
    for (g_ids, _, p_ids, _), sim in zip(frames, sims):
        if len(g_ids) == 0:
            FP += len(p_ids)
            continue
        if len(p_ids) == 0:
            FN += len(g_ids)
            continue
        score = glob[g_ids[:, None], p_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            TP[a] += len(r)
            FN[a] += len(g_ids) - len(r)
            FP[a] += len(p_ids) - len(r)
            if len(r):
                loc[a] += sim[r, c].sum()
                mc[a][g_ids[r], p_ids[c]] += 1

    ass_a = np.zeros(A)
    for a in range(A):
        ass = mc[a] / np.maximum(1, gt_cnt[:, None] + pr_cnt[None, :] - mc[a])
        ass_a[a] = (mc[a] * ass).sum() / max(1.0, TP[a])
    return {"TP": TP, "FN": FN, "FP": FP, "AssA": ass_a, "LocA": loc / np.maximum(1e-10, TP)}


def hota_combine(seq_results):
    """Combine per-sequence results the way TrackEval does."""
    if not seq_results:
        return None
    TP = sum(r["TP"] for r in seq_results)
    FN = sum(r["FN"] for r in seq_results)
    FP = sum(r["FP"] for r in seq_results)
    w = np.maximum(1e-10, TP)
    ass_a = sum(r["AssA"] * r["TP"] for r in seq_results) / w
    loc_a = sum(r["LocA"] * r["TP"] for r in seq_results) / w
    det_a = TP / np.maximum(1, TP + FN + FP)
    return {"TP": TP, "FN": FN, "FP": FP, "DetA": det_a, "AssA": ass_a,
            "LocA": loc_a, "HOTA": np.sqrt(det_a * ass_a)}


# --------------------------------------------------------------------------- MOTA
def clear_sequence(frames, thr=2.0):
    """CLEAR-MOT counts for one sequence (same matching rules as py-motmetrics,
    which the nuScenes devkit uses): previous matches are carried over while
    dist < thr, the rest is solved with Hungarian."""
    fn = fp = idsw = n_gt = n_match = 0
    dist_sum = 0.0
    last = {}  # gt id -> pred id it was last matched to (persists over gaps)
    for g_ids, g_xy, p_ids, p_xy in frames:
        n_gt += len(g_ids)
        if len(g_ids) == 0 or len(p_ids) == 0:
            fn += len(g_ids)
            fp += len(p_ids)
            continue
        d = pairwise_dist(g_xy, p_xy)
        gi = {int(g): i for i, g in enumerate(g_ids)}
        pj = {int(p): j for j, p in enumerate(p_ids)}
        used_g, used_p, pairs = set(), set(), []
        for g, p in last.items():
            if g in gi and p in pj and d[gi[g], pj[p]] < thr:
                pairs.append((gi[g], pj[p]))
                used_g.add(gi[g])
                used_p.add(pj[p])
        rg = [i for i in range(len(g_ids)) if i not in used_g]
        rp = [j for j in range(len(p_ids)) if j not in used_p]
        if rg and rp:
            sub = d[np.ix_(rg, rp)]
            cost = np.where(sub < thr, sub, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if sub[r, c] < thr:
                    g, p = int(g_ids[rg[r]]), int(p_ids[rp[c]])
                    if g in last and last[g] != p:
                        idsw += 1
                    pairs.append((rg[r], rp[c]))
        for i, j in pairs:
            last[int(g_ids[i])] = int(p_ids[j])
            dist_sum += d[i, j]
        n_match += len(pairs)
        fn += len(g_ids) - len(pairs)
        fp += len(p_ids) - len(pairs)
    return {"gt": n_gt, "fn": fn, "fp": fp, "idsw": idsw, "match": n_match, "dist": dist_sum}


def clear_combine(seq_results):
    keys = ("gt", "fn", "fp", "idsw", "match", "dist")
    t = {k: sum(r[k] for r in seq_results) for k in keys}
    t["MOTA"] = 1.0 - (t["fn"] + t["fp"] + t["idsw"]) / t["gt"] if t["gt"] else float("nan")
    t["MOTP"] = t["dist"] / t["match"] if t["match"] else float("nan")
    return t