#!/usr/bin/env python3
"""Baseline HOTA/MOTA for AB3DMOT tracks on one nuScenes scene.

Matching: same class + global 3D center distance <= --distance-threshold (default 2 m).
HOTA is the distance-threshold variant at that fixed threshold, not official nuScenes tracking evaluation.
"""
import argparse, json, os, csv
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes

# nuScenes category prefix -> detection class used by BEVFusion predictions
CATEGORY_TO_CLASS = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.bus.bendy': 3,
    'vehicle.bus.rigid': 3,
    'vehicle.trailer': 4,
    'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7,
}
SUPPORTED = {0, 1, 3, 4, 6, 7, 8}
CLASS_NAMES = {0:'car', 1:'truck', 3:'bus', 4:'trailer', 6:'motorcycle', 7:'bicycle', 8:'pedestrian'}

def gt_class(category_name):
    if category_name.startswith('human.pedestrian.'):
        return 8
    return CATEGORY_TO_CLASS.get(category_name)

def load_gt(nusc, scene_name, allowed_tokens):
    scenes = [s for s in nusc.scene if s['name'] == scene_name]
    if not scenes:
        raise RuntimeError(f'Scene not found: {scene_name}')
    scene = scenes[0]
    frames = {}
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        if token in allowed_tokens:
            rows=[]
            for ann_token in sample['anns']:
                ann=nusc.get('sample_annotation', ann_token)
                cid=gt_class(ann['category_name'])
                if cid not in SUPPORTED:
                    continue
                rows.append({'id': ann['instance_token'], 'class_id': cid,
                             'xyz': np.asarray(ann['translation'][:3], dtype=float)})
            frames[token]=rows
        token=sample['next']
    return frames

def match_frame(gt_rows, pred_rows, threshold):
    matches=[]
    for cid in sorted(SUPPORTED):
        g=[x for x in gt_rows if x['class_id']==cid]
        p=[x for x in pred_rows if x['class_id']==cid]
        if not g or not p:
            continue
        dist=np.linalg.norm(np.stack([x['xyz'] for x in g])[:,None,:]-np.stack([x['xyz'] for x in p])[None,:,:],axis=2)
        cost=dist.copy()
        cost[cost > threshold] = 1e6
        gi,pi=linear_sum_assignment(cost)
        for i,j in zip(gi,pi):
            if dist[i,j] <= threshold:
                matches.append((g[i]['id'], p[j]['id'], cid, float(dist[i,j])))
    return matches

def evaluate(gt_frames, pred_frames, threshold):
    tp=fp=fn=0
    gt_counts=defaultdict(int); pred_counts=defaultdict(int); pair_counts=defaultdict(int)
    id_switches=0; last_match={}
    class_stats=defaultdict(lambda: {'tp':0,'fp':0,'fn':0,'idsw':0})
    matched_rows=[]
    for token in sorted(set(gt_frames)|set(pred_frames), key=lambda t: (pred_frames.get(t,{}).get('frame_index',10**9),t)):
        gs=gt_frames.get(token,[]); ps=pred_frames.get(token,[])
        for g in gs: gt_counts[(g['class_id'],g['id'])]+=1
        for p in ps: pred_counts[(p['class_id'],p['id'])]+=1
        ms=match_frame(gs,ps,threshold)
        tpf=len(ms); fpf=len(ps)-tpf; fnf=len(gs)-tpf
        tp+=tpf; fp+=fpf; fn+=fnf
        for g in gs: class_stats[g['class_id']]['fn'] += 1
        for p in ps: class_stats[p['class_id']]['fp'] += 1
        for gid,pid,cid,d in ms:
            class_stats[cid]['tp']+=1; class_stats[cid]['fn']-=1; class_stats[cid]['fp']-=1
            pair_counts[(cid,gid,pid)]+=1
            old=last_match.get((cid,gid))
            if old is not None and old != pid:
                id_switches+=1; class_stats[cid]['idsw']+=1
            last_match[(cid,gid)]=pid
            matched_rows.append((cid,gid,pid))
    mota=1.0-(fn+fp+id_switches)/max(1,sum(len(v) for v in gt_frames.values()))
    deta=tp/max(1,tp+0.5*fp+0.5*fn)
    ass_sum=0.0
    for cid,gid,pid in matched_rows:
        pair=pair_counts[(cid,gid,pid)]
        gt_n=gt_counts[(cid,gid)]
        pred_n=pred_counts[(cid,pid)]
        ass_sum += pair/max(1, gt_n+pred_n-pair)
    assa=ass_sum/max(1,tp)
    hota=float(np.sqrt(deta*assa))
    return {'HOTA_distance_variant':hota,'DetA':deta,'AssA':assa,'MOTA':mota,
            'TP':tp,'FP':fp,'FN':fn,'ID_switches':id_switches,
            'GT_detections':sum(len(v) for v in gt_frames.values()),
            'pred_detections':sum(len(v) for v in pred_frames.values()),
            'class_stats':{str(c):dict(v, name=CLASS_NAMES[c], MOTA=(1-(v['fn']+v['fp']+v['idsw'])/max(1,sum(1 for fr in gt_frames.values() for x in fr if x['class_id']==c)))) for c,v in sorted(class_stats.items())}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True, help='nuScenes-mini dataset root containing v1.0-mini')
    ap.add_argument('--tracks', required=True, help='ab3dmot_tracks_global JSON')
    ap.add_argument('--scene', default='scene-0061')
    ap.add_argument('--distance-threshold', type=float, default=2.0)
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--out', default='baseline_hota_mota.json')
    args=ap.parse_args()
    with open(args.tracks) as f: data=json.load(f)
    pred_frames={tok:rows for tok,rows in data['results'].items() if any(r.get('scene_name')==args.scene for r in rows)}
    if not pred_frames: raise RuntimeError(f'No predictions found for {args.scene}')
    allowed=set(pred_frames)
    nusc=NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    gt_frames=load_gt(nusc,args.scene,allowed)
    # Normalize prediction rows for matching; tracker IDs can overlap between classes, so retain class in identity.
    pred_norm={}
    for tok,rows in pred_frames.items():
        pred_norm[tok]=[]
        for r in rows:
            cid=int(r['class_id'])
            if cid not in SUPPORTED: continue
            pred_norm[tok].append({'id':int(r['track_id']),'class_id':cid,'xyz':np.asarray(r['translation'][:3],dtype=float)})
    # Include every sample in this scene, even frames with no predicted detections.
    scene=next(s for s in nusc.scene if s['name']==args.scene)
    tok=scene['first_sample_token']; scene_tokens=[]
    while tok:
        scene_tokens.append(tok); tok=nusc.get('sample',tok)['next']
    for tok in scene_tokens:
        gt_frames.setdefault(tok,[]); pred_norm.setdefault(tok,[])
    # Tracker IDs are class-local; make their identities unique across classes.
    for rows in pred_norm.values():
        for r in rows: r['id']=f"{r['class_id']}:{r['id']}"
    result=evaluate(gt_frames,pred_norm,args.distance_threshold)
    result.update({'dataset':'nuScenes-mini','scene':args.scene,'frames':len(scene_tokens),
                   'matching':'same-class global 3D center distance',
                   'distance_threshold_m':args.distance_threshold,
                   'note':'HOTA_distance_variant uses one fixed center-distance threshold; it is not TrackEval standard IoU-based HOTA or official nuScenes AMOTA/AMOTP.'})
    with open(args.out,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    print(f'\nSaved: {os.path.abspath(args.out)}')
if __name__=='__main__': main()
