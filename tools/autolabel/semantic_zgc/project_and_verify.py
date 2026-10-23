import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA
ROOT=DATA
from centerpoint_weak_semantics import weak_targets
from fresh_projection import project_labels
from common import atomic_json, sha, SOURCE, NAMES, read_source_image, render_overlay
from label_mapping import training_image_labels
from bev_visualization import render_review


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--incremental',action='store_true')
    args=parser.parse_args()
    out=ROOT/('smoke_labels' if args.smoke else 'labels')
    counts=np.zeros(7,np.int64)
    all_rows=[]
    for kind in ('zgc',):
        source=json.loads((SOURCE/kind/'index.json').read_text())
        metas=sorted((out/kind/'metadata').glob('*.json'))
        if not args.smoke and not args.incremental:
            assert len(metas)==len(source['rows']), (kind,len(metas),len(source['rows']))
        rows=[]
        for ordinal,meta_path in enumerate(metas,1):
            meta=json.loads(meta_path.read_text())
            row=meta['source']
            dest=out/kind
            for key in ('mask','confidence','overlay'):
                assert sha(dest/meta[key])==meta[key+'_sha256']
            with np.load(SOURCE/kind/row['path'],allow_pickle=False) as a:
                arrays={k:a[k] for k in a.files}
            assert sha(SOURCE/kind/row['path'])==row['sha256']
            mask=np.array(Image.open(dest/meta['mask']))
            with np.load(dest/meta['confidence']) as a:
                confidence=a['confidence'].astype(np.float32)
            assert mask.shape==confidence.shape
            assert set(np.unique(mask))<=set(range(10))|{255}
            assert np.isfinite(confidence).all() and (confidence>=0).all() and (confidence<=1).all()
            assert (confidence[mask==255]==0).all()
            labels,scores=project_labels(arrays,training_image_labels(mask),confidence)
            arrays['point_labels']=labels
            arrays['point_confidence']=scores
            arrays['supervision_policy']=np.array('static_drivable_occupancy_v1')
            arrays.pop('bev_drivable_teacher_labels',None)
            targets=weak_targets(arrays,schema='zgc_image_teacher_seven_v2')
            occ_bits=targets['weak_occupancy_bits']
            occupancy=np.full((320,320),255,np.uint8)
            teacher=np.full((320,320),255,np.uint8)
            mapping=[0,1,2,5,4,6,3]
            for label,occ in enumerate(mapping):
                selected=occ_bits==(1<<occ)
                occupancy[selected]=occ
                teacher[selected]=label
            drive=np.full((320,320),255,np.uint8)
            drive[targets['weak_drivable_bits']==1]=0
            drive[targets['weak_drivable_bits']==2]=1
            arrays['bev_labels']=teacher.T.copy()
            arrays['occupancy_labels']=occupancy
            arrays['occupancy_confidence']=targets['weak_confidence']
            arrays['observed']=np.asarray(arrays.get('observed',arrays.get('bev_observed')),bool)
            arrays.update(targets)
            # Recompute through the exact training reader after archive construction.
            check=weak_targets(arrays,schema='zgc_image_teacher_seven_v2')
            for key in targets:
                np.testing.assert_array_equal(check[key],targets[key])
            assert (scores[labels==255]==0).all()
            frame_path=dest/row['path']
            frame_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=frame_path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**arrays)
            temporary.replace(frame_path)
            with np.load(frame_path,allow_pickle=False) as a:
                for key,value in check.items():
                    np.testing.assert_array_equal(weak_targets(a,schema='zgc_image_teacher_seven_v2')[key],value)
            stem=frame_path.stem
            class_counts=np.bincount(labels[labels<7],minlength=7)
            updated={**row,'sha256':sha(frame_path),'source_label_path':str(SOURCE/kind/row['path']),
                     'source_label_sha256':row['sha256'],'class_point_counts':class_counts.tolist(),
                     'ignored_points':int((labels==255).sum()),'supervision_policy':'static_drivable_occupancy_v1',
                     'mask_path':str(dest/meta['mask'])}
            rows.append(updated)
            if row['split']=='train':
                counts+=class_counts
            original=read_source_image(row)
            assert original.size==(mask.shape[1],mask.shape[0])
            pair=render_overlay(original,mask,kind,stem,.55)
            raw_point_labels,_=project_labels(arrays,mask,confidence)
            canvas=render_review(pair,original,arrays,raw_point_labels,occupancy)
            temporary_overlay=(dest/meta['overlay']).with_suffix('.tmp.jpg')
            canvas.save(temporary_overlay,quality=70)
            temporary_overlay.replace(dest/meta['overlay'])
            meta['overlay_sha256']=sha(dest/meta['overlay'])
            meta['overlay_layout']='2x2: original | image mask / LiDAR image projection | BEV training mask'
            meta['image_to_training_labels']=[0,1,2,3,4,5,6,6,6,4]
            meta['bev_priority_low_to_high']=['ground','open obstacle','building','fence','other obstacle','vehicle','person']
            meta.update(output_sha256=updated['sha256'],class_point_counts=class_counts.tolist(),
                        bev_drivable_cells={'non_drivable':int((drive==0).sum()),'drivable':int((drive==1).sum()),'ignored':int((drive==255).sum())})
            atomic_json(meta_path,meta)
            all_rows.append((kind,stem,row,meta))
            print(f'project_verify {kind} {ordinal}/{len(metas)} {row["frame_id"]}',flush=True)
        atomic_json(out/kind/'index.json',{**source,'supervision_policy':'static_drivable_occupancy_v1','rows':rows})
    if not args.smoke and not args.incremental:
        assert len(all_rows)==971
    atomic_json(out/'verification.json',{'state':'partial' if args.incremental else 'complete','frames':len(all_rows),
                'full_data_integrity_checks':True,'train_point_counts':counts.tolist(),
                'train_frames':sum(r['split']=='train' for _,_,r,_ in all_rows),
                'val_frames':sum(r['split']=='val' for _,_,r,_ in all_rows),
                'class_names':['ground','person','vehicle','building','foliage','fence','other obstacle'], 'image_class_names':NAMES, 'raw_pole_label':7, 'training_pole_label':6, 'occupancy_pole_label':3, 'raw_ground_object_label':8, 'training_ground_object_label':6, 'occupancy_ground_object_label':3, 'raw_roof_label':9, 'training_roof_label':4, 'occupancy_roof_label':4,'manual_ground_truth':False})
    page=['<!doctype html><meta charset="utf-8"><title>SAM3 图像叠加图</title><style>body{font:18px sans-serif;margin:24px;background:#eee;line-height:1.6}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}article{margin:36px 0}</style><h1>SAM3 图像叠加图</h1><p>2×2：左上原图，右上图像mask，左下点云投影，右下BEV训练mask。柱子/灯杆/电线在图像mask中为玫红色，训练BEV中归为其他障碍物（红）。井盖/小碎屑=地面；屋顶=紫（镂空障碍物）；地面物体=红（BEV其他障碍物）；点云与BEV显示XY±20m；地面=绿；行人=亮青；车辆=黄；建筑=蓝；树叶=紫；围栏/灌木/绿篱=橙色；树干=红。路沿不作为围栏。标签是伪标签，没有人工真值。可视化仅保留图像叠加图；原始类别掩码、置信度、点云及BEV数组保留用于训练。</p>']
    for kind,stem,row,meta in all_rows:
        prefix=f'{kind}/'
        page.append(f'<article><h2>{kind} {row["frame_id"]}</h2><details><summary>查看图像叠加图</summary>')
        for title,key in [('图像叠加图','overlay')]:
            path=prefix+meta[key]
            page.append(f'<p>{title}</p><a href="{path}"><img loading="lazy" src="{path}"></a>')
        page.append('</details><p>原数据：<code>'+str(row.get('image_path',row.get('original_shard','')))+' :: '+str(row.get('original_lmdb_key',''))+'</code></p></article>')
    (out/'index.html').write_text('\n'.join(page))
    from visualization_gallery import build_gallery
    build_gallery(out)
    print('PROJECT_AND_VERIFY_COMPLETE',flush=True)

if __name__=='__main__':
    main()
