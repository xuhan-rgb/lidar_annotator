"""Quality-ranked rotated BEV NMS of map review candidates; original labels untouched."""
import argparse,json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def corners(box):
    b=np.asarray(box);q=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*b[3:5]/2
    R=np.array([[np.cos(b[6]),-np.sin(b[6])],[np.sin(b[6]),np.cos(b[6])]])
    return q@R.T+b[:2]


def bev_iou(a,b):
    p,q=Polygon(corners(a)),Polygon(corners(b));intersection=p.intersection(q).area
    return float(intersection/(p.area+q.area-intersection))


def deduplicate(records,threshold=.6):
    ranked=sorted(records,key=lambda r:(-r['anchor_quality']['score'],r['track_id']))
    kept=[];suppressed=[]
    for r in ranked:
        match=next((k for k in kept if bev_iou(r['candidate_box_map'],k['candidate_box_map'])>=threshold),None)
        if match is None:kept.append(r)
        else:suppressed.append(dict(track_id=r['track_id'],kept_track_id=match['track_id'],bev_iou=bev_iou(r['candidate_box_map'],match['candidate_box_map']),reason='high_rotated_bev_overlap_lower_anchor_quality'))
    return kept,suppressed


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();c=json.loads(args.config.read_text());root=Path(c['temporal_output']);out=root/'map_review/history_fit/bev_dedup';out.mkdir(exist_ok=True)
    report=json.loads((root/'map_review/history_fit/report.json').read_text());records=[r for r in report['results'] if 'candidate_box_map'in r];kept,suppressed=deduplicate(records)
    np.savez_compressed(out/'vehicle_map_boxes.npz',box_3d_map=np.array([r['candidate_box_map'] for r in kept]),track_ids=np.array([r['track_id'] for r in kept]))
    with np.load(root/'map_review/map_points.npz') as z:bg=z['background_xyz_map']
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(18,14),dpi=140);ax.scatter(bg[:,1],bg[:,0],s=.15,c='#87939c',alpha=.3,linewidths=0)
    selected={r['track_id'] for r in kept}
    for r in records:
        q=corners(r['candidate_box_map']);q=np.r_[q,q[:1]];yes=r['track_id'] in selected;ax.plot(q[:,1],q[:,0],'-' if yes else '--',c='#22e690' if yes else '#ff719b',lw=1.6 if yes else 1.,alpha=1 if yes else .7)
        b=r['candidate_box_map'];ax.annotate(f'T{r["track_id"]}'+('' if yes else ' suppressed'),(b[1],b[0]),xytext=(3,3 if yes else -12),textcoords='offset points',color='white' if yes else '#ff719b',fontsize=8)
    ax.plot([],[],c='#22e690',label='Retained map boxes');ax.plot([],[],'--',c='#ff719b',label='Suppressed duplicates (diagnostic)');ax.legend();ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m), +X up');ax.grid(alpha=.15);ax.set_title(f'Vehicle map BEV deduplication · {len(records)} -> {len(kept)} candidates\nRotated IoU >= 0.60; higher anchor quality retained; original labels unchanged');fig.tight_layout();fig.savefig(out/'whole_map_dedup.png');plt.close(fig)
    pairs=[dict(track_a=a['track_id'],track_b=b['track_id'],bev_iou=bev_iou(a['candidate_box_map'],b['candidate_box_map'])) for i,a in enumerate(records) for b in records[i+1:] if bev_iou(a['candidate_box_map'],b['candidate_box_map'])>.01]
    result=dict(status='complete',input_map_candidates=len(records),retained_map_candidates=len(kept),rotated_bev_iou_threshold=.6,ranking='existing anchor_quality.score descending; track ID tie-break',retained_track_ids=[r['track_id'] for r in kept],suppressed=suppressed,overlapping_pairs=pairs,labels_modified=False,identity_tracks_merged=False,limits='Geometric duplicate suppression on map candidates only; not proven physical identity merging; invalid localization can overlap neighbouring vehicles. No per-frame labels deleted.')
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
