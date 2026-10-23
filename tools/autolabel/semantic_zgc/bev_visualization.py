import numpy as np
from PIL import Image, ImageDraw
from common import COLORS as IMAGE_COLORS

COLORS=np.array([[55,190,65],[0,255,255],[255,220,0],[245,65,60],
                 [160,105,225],[255,115,15],[65,135,240]],np.uint8)
# Network occupancy: ground, person, vehicle, other, open, building, fence.
COLORS=COLORS[[0,1,2,3,4,6,5]]
NAMES=['ground','person','vehicle','other obstacle','open obstacle','building','fence']


def render_review(pair,original,arrays,raw_point_labels,occupancy):
    w,h=original.size
    panel_h=h+78
    canvas=Image.new('RGB',(2*w,2*panel_h),'#171717')
    canvas.paste(pair,(0,0))
    projection=np.array(original)
    uv=np.asarray(arrays['pixels_uv'])
    finite=np.isfinite(uv).all(1)
    pixels=np.zeros(uv.shape,np.int64)
    pixels[finite]=np.floor(uv[finite]).astype(np.int64)
    valid=np.asarray(arrays['visible'],bool)&finite&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    xy=np.asarray(arrays['xyz'])[:,:2]
    valid &= np.isfinite(xy).all(1)&((xy>=-20)&(xy<20)).all(1)
    ids=np.flatnonzero(valid)
    coords=pixels[ids]
    labels=raw_point_labels[ids]
    colors=np.full((len(ids),3),160,np.uint8)
    known=labels<10
    colors[known]=IMAGE_COLORS[labels[known]]
    # Small dark outlines keep points readable against bright backgrounds.
    for radius,color in ((3,np.full_like(colors,25)),(2,colors)):
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                if dx*dx+dy*dy>radius*radius:
                    continue
                x=coords[:,0]+dx;y=coords[:,1]+dy
                keep=(x>=0)&(x<w)&(y>=0)&(y<h)
                projection[y[keep],x[keep]]=color[keep]
    canvas.paste(Image.fromarray(projection),(0,panel_h+78))
    # Network [Y,X] -> display +X up and +Y left.
    # Stored 320x320 grid spans +/-40m at 0.25m/cell. Display only +/-20m.
    grid=occupancy[80:240,80:240].T[::-1,::-1]
    rgb=np.full((*grid.shape,3),[32,42,48],np.uint8)
    known=grid<7
    rgb[known]=COLORS[grid[known]]
    side=h-48
    x0=w+(w-side)//2;y0=panel_h+78
    canvas.paste(Image.fromarray(rgb).resize((side,side),Image.Resampling.NEAREST),(x0,y0))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,panel_h+7),'LiDAR projected onto original image | XY [-20,20)m | raw image labels | gray = ignore',fill='white')
    draw.text((12,panel_h+30),'Same-frame stored calibrated pixels_uv + visible mask; pole/wire is rose pink; person is cyan',fill='white')
    draw.text((w+12,panel_h+7),'Projected BEV training labels | +X up, +Y left | 0.25m/cell | dark = ignore',fill='white')
    for i,name in enumerate(NAMES):
        x=w+12+175*i
        draw.rectangle((x,panel_h+30,x+14,panel_h+44),fill=tuple(COLORS[i]))
        draw.text((x+20,panel_h+30),name,fill='white')
    draw.text((w+12,panel_h+55),'Ground < open obstacle < other classes; trunk/pole/wire -> other obstacle',fill='white')
    draw.text((x0,y0+side+10),'Range X,Y +/-20m; no labels filled into unobserved cells',fill='white')
    return canvas
