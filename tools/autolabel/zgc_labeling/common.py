import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import configured

import numpy as np
from PIL import Image, ImageDraw

SOURCE=configured('base_semantics')
COLORS=np.array([[55,190,65],[0,255,255],[255,220,0],[65,135,240],
                 [160,105,225],[255,115,15],[245,65,60],[255,80,160],[245,65,60],[160,105,225]],np.uint8)
NAMES=['ground','person','vehicle','building','foliage','fence','tree trunk','pole / wire','ground object','roof / eaves']

def render_overlay(image,labels,kind,stem,mask_alpha):
    known=labels!=255
    rgb=np.array(image)
    rgb[known]=(rgb[known]*(1-mask_alpha)+COLORS[labels[known]]*mask_alpha).astype(np.uint8)
    canvas=Image.new('RGB',(image.width*2,image.height+78),'#171717')
    canvas.paste(image,(0,78))
    canvas.paste(Image.fromarray(rgb),(image.width,78))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,7),f'Original image | {kind} {stem}',fill='white')
    draw.text((image.width+12,7),f'SAM3 refined masks | building minus (roof UNION agreed canopy) | mask alpha {mask_alpha:.2f}',fill='white')
    for i,name in enumerate(NAMES):
        x=image.width+12+170*i
        draw.rectangle((x,30,x+16,45),fill=tuple(COLORS[i]))
        draw.text((x+22,30),name,fill='white')
    draw.text((image.width+12,55),'roof -> open obstacle; covers/debris -> ground; ground object -> other obstacle; foliage -> open obstacle; bush/hedge -> fence; grass -> drivable ground',fill='white')
    return canvas

def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read_source_image(row):
    import io
    import lmdb
    if 'image_path' in row:
        return Image.open(row['image_path']).convert('RGB')
    with lmdb.open(row['original_shard'],readonly=True,lock=False) as env:
        with env.begin() as txn:
            payload=txn.get(row['original_lmdb_key'].encode())
    assert hashlib.sha256(payload).hexdigest()==row['original_payload_sha256']
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        return Image.open(io.BytesIO(data['image_jpeg_bytes'].tobytes())).convert('RGB')
