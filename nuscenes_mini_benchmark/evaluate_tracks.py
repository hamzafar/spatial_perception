#!/usr/bin/env python3
"""
HOTA / MOTA for AB3DMOT tracks on nuScenes (mini), per class and class-averaged.

Differences from a naive evaluation (and why they matter):
  * GT uses the official nuScenes *tracking* classes only (wheelchair / stroller /
    personal_mobility are NOT pedestrians), and drops GT boxes with 0 lidar+radar
    points and boxes beyond the official per-class range (car 50 m, pedestrian 40 m ...).
    Predictions get the same range filter.
  * MOTA follows CLEAR-MOT / py-motmetrics (what the nuScenes devkit uses): matches
    from earlier frames are kept while the pair is still < threshold, so ID switches are
    not inflated by re-solving the assignment from scratch every frame.
  * HOTA follows TrackEval (DetA = TP/(TP+FN+FP), global-alignment matching) and is
    averaged over 19 thresholds alpha = 0.05..0.95, with
        similarity = max(0, 1 - BEV centre distance / dmax)     (default dmax = 2 m).
    This is still a distance-based HOTA, not the IoU-based one used for 2D MOT.

Input tracks JSON (your ab3dmot_tracks_global format):
    {"results": {sample_token: [{"scene_name", "class_id", "track_id",
                                 "translation": [x, y, z], ("score": float)}, ...]}}
Prediction class ids follow BEVFusion's 10-class order.
"""
import argparse
import json
import os

import numpy as np

from track_metrics import (ALPHAS, clear_combine, clear_sequence, hota_combine,
                           hota_sequence)

# Official nuScenes tracking class mapping and per-class ranges, inlined so the script does not
# import nuscenes.eval.tracking (which requires the motmetrics package).
CATEGORY_TO_TRACKING = {
    "vehicle.bicycle": "bicycle", "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "vehicle.car": "car", "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.trailer": "trailer", "vehicle.truck": "truck",
}
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50,
               "pedestrian": 40, "motorcycle": 40, "bicycle": 40}


def category_to_tracking_name(category_name):
    return CATEGORY_TO_TRACKING.get(category_name)


PRED_CLASS = {0: "car", 1: "truck", 3: "bus", 4: "trailer",
              6: "motorcycle", 7: "bicycle", 8: "pedestrian"}
TRACK_CLASSES = ["car", "truck", "bus", "trailer", "motorcycle", "bicycle", "pedestrian"]


def scene_samples(nusc, scene):
    tok, out = scene["first_sample_token"], []
    while tok:
        out.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return out


def ego_xy(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return np.array(nusc.get("ego_pose", sd["ego_pose_token"])["translation"][:2])


def build_frames(nusc, scene, results, class_range, category_to_name,
                 score_thr=0.0, filter_gt=True):
    """-> {class_name: [(gt_ids, gt_xy, pr_ids, pr_xy) per frame]} with dense int ids."""
    per_cls = {c: [] for c in TRACK_CLASSES}
    id_maps = {c: ({}, {}) for c in TRACK_CLASSES}
    for tok in scene_samples(nusc, scene):
        sample = nusc.get("sample", tok)
        ego = ego_xy(nusc, sample)
        gt = {c: ([], []) for c in TRACK_CLASSES}
        pr = {c: ([], []) for c in TRACK_CLASSES}
        for at in sample["anns"]:
            a = nusc.get("sample_annotation", at)
            name = category_to_name(a["category_name"])
            if name is None:
                continue
            if filter_gt:
                if a["num_lidar_pts"] + a["num_radar_pts"] <= 0:
                    continue
                if np.linalg.norm(np.array(a["translation"][:2]) - ego) > class_range[name]:
                    continue
            gt[name][0].append(a["instance_token"])
            gt[name][1].append(a["translation"][:2])
        for r in results.get(tok, []):
            if r.get("scene_name", scene["name"]) != scene["name"]:
                continue
            name = PRED_CLASS.get(int(r["class_id"]))
            if name is None or float(r.get("score", 1.0)) < score_thr:
                continue
            xy = np.array(r["translation"][:2])
            if filter_gt and np.linalg.norm(xy - ego) > class_range[name]:
                continue
            pr[name][0].append(int(r["track_id"]))
            pr[name][1].append(xy)
        for c in TRACK_CLASSES:
            gm, pm = id_maps[c]
            g_ids = np.array([gm.setdefault(i, len(gm)) for i in gt[c][0]], dtype=int)
            p_ids = np.array([pm.setdefault(i, len(pm)) for i in pr[c][0]], dtype=int)
            g_xy = np.array(gt[c][1], dtype=float).reshape(-1, 2)
            p_xy = np.array(pr[c][1], dtype=float).reshape(-1, 2)
            per_cls[c].append((g_ids, g_xy, p_ids, p_xy))
    return per_cls


def evaluate(nusc, results, scene_names, class_range, category_to_name,
             dmax=2.0, thr=2.0, score_thr=0.0, filter_gt=True):
    scenes = [s for s in nusc.scene if s["name"] in scene_names]
    missing = set(scene_names) - {s["name"] for s in scenes}
    if missing:
        raise RuntimeError(f"Scenes not in dataset: {sorted(missing)}")
    hota_seq = {c: [] for c in TRACK_CLASSES}
    clear_seq = {c: [] for c in TRACK_CLASSES}
    for sc in scenes:
        frames = build_frames(nusc, sc, results, class_range, category_to_name,
                              score_thr, filter_gt)
        for c, fr in frames.items():
            if sum(len(f[0]) for f in fr) == 0 and sum(len(f[2]) for f in fr) == 0:
                continue
            hota_seq[c].append(hota_sequence(fr, dmax))
            clear_seq[c].append(clear_sequence(fr, thr))
    out = {}
    for c in TRACK_CLASSES:
        if not hota_seq[c]:
            continue
        h, m = hota_combine(hota_seq[c]), clear_combine(clear_seq[c])
        if m["gt"] == 0:
            continue  # class has no GT: excluded from averages
        out[c] = {"HOTA": float(h["HOTA"].mean()), "DetA": float(h["DetA"].mean()),
                  "AssA": float(h["AssA"].mean()), "LocA": float(h["LocA"].mean()),
                  "MOTA": float(m["MOTA"]), "MOTP_m": float(m["MOTP"]),
                  "GT": int(m["gt"]), "FN": int(m["fn"]), "FP": int(m["fp"]),
                  "IDSW": int(m["idsw"]), "pred": int(m["match"] + m["fp"])}
    keys = ["HOTA", "DetA", "AssA", "LocA", "MOTA"]
    out["MEAN"] = {k: float(np.mean([out[c][k] for c in out])) for k in keys}
    out["MEAN"]["classes"] = [c for c in out if c != "MEAN"]
    return out


def main():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="*", help="scene names; default = every scene in --tracks")
    ap.add_argument("--split", choices=["mini_train", "mini_val", "mini_all"],
                    help="use official mini split(s) instead of --scenes")
    ap.add_argument("--dmax", type=float, default=2.0, help="HOTA similarity range (m)")
    ap.add_argument("--thr", type=float, default=2.0, help="MOTA match threshold (m)")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--no-filter", action="store_true",
                    help="skip official GT/pred filtering (points, range)")
    ap.add_argument("--out", default="hota_mota.json")
    a = ap.parse_args()

    with open(a.tracks) as f:
        results = json.load(f)["results"]
    splits = create_splits_scenes()
    if a.split == "mini_all":
        names = splits["mini_train"] + splits["mini_val"]
    elif a.split:
        names = splits[a.split]
    elif a.scenes:
        names = a.scenes
    else:
        names = sorted({r["scene_name"] for rows in results.values() for r in rows if "scene_name" in r})
    have = {r.get("scene_name") for rows in results.values() for r in rows}
    for n in names:
        if n not in have:
            print(f"WARNING: no predictions for {n}; all its GT counts as missed")

    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    res = evaluate(nusc, results, set(names), CLASS_RANGE, category_to_tracking_name,
                   a.dmax, a.thr, a.score_thr, not a.no_filter)

    print(f"scenes: {names}")
    print(f"{'class':<11}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>8}{'GT':>7}{'FN':>7}{'FP':>6}{'IDSW':>6}")
    for c, v in res.items():
        if c == "MEAN":
            continue
        print(f"{c:<11}{v['HOTA']:7.3f}{v['DetA']:7.3f}{v['AssA']:7.3f}{v['MOTA']:8.3f}"
              f"{v['GT']:7d}{v['FN']:7d}{v['FP']:6d}{v['IDSW']:6d}")
    m = res["MEAN"]
    print(f"{'MEAN':<11}{m['HOTA']:7.3f}{m['DetA']:7.3f}{m['AssA']:7.3f}{m['MOTA']:8.3f}"
          f"   (class-mean over {len(m['classes'])} classes with GT)")
    res["settings"] = {"scenes": names, "dmax": a.dmax, "mota_thr": a.thr,
                       "score_thr": a.score_thr, "official_filtering": not a.no_filter,
                       "alphas": [round(x, 2) for x in ALPHAS]}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()