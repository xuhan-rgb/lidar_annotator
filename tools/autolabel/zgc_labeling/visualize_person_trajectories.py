"""Map review of existing visual-confirmed pedestrian tracks and robust XY smoothing."""
import argparse,colorsys,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def smooth_track(times,xy):
    """Data-supported positions only; weak acceleration regularity, no extrapolation."""
    times=np.asarray(times);xy=np.asarray(xy,float);n=len(xy)
    if n<3:return xy.copy()
    dt=np.diff(times)
    if np.any(dt<=0) or np.any(dt>.50001):raise ValueError('Track must contain ordered, continuous observations')
    def residual(flat):
        p=flat.reshape(n,2);obs=(p-xy)/.2
        # Robust observation loss, quadratic weak acceleration term.
        obs=np.sign(obs)*np.sqrt(2*(np.sqrt(1+obs**2)-1))
        velocity=np.diff(p,axis=0)/dt[:,None]
        acceleration=np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
        return np.r_[obs.ravel(),(acceleration/3.).ravel()]
    pattern=lil_matrix((2*n+2*(n-2),2*n),dtype=int)
    for i in range(n):pattern[2*i:2*i+2,2*i:2*i+2]=1
    for i in range(n-2):pattern[2*n+2*i:2*n+2*i+2,2*i:2*i+6]=1
    fit=least_squares(residual,xy.ravel(),jac_sparsity=pattern.tocsr(),max_nfev=100,ftol=1e-7)
    if not fit.success:raise RuntimeError(f'Trajectory optimization failed: {fit.message}')
    return fit.x.reshape(n,2)


def draw_vehicles(ax,records):
    for r in records:
        if 'candidate_box_map' not in r:continue
        b=np.array(r['candidate_box_map']);q=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*b[3:5]/2
        R=np.array([[np.cos(b[6]),-np.sin(b[6])],[np.sin(b[6]),np.cos(b[6])]]);q=q@R.T+b[:2]
        ax.plot(q[:,1],q[:,0],c='#32cf9e',lw=1.2,alpha=.8);ax.text(b[1],b[0],f'V{r["track_id"]}',color='#aee4d3',fontsize=8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=json.loads(args.config.read_text());root=Path(c['temporal_output']);out=root/'map_review'/'person_trajectories';out.mkdir(exist_ok=True)
    groups=defaultdict(list);ignored=[];statuses=Counter()
    for f in sorted((root/'metadata').glob('*.json')):
        for o in json.loads(f.read_text())['observations']:
            if o['kind']!='person':continue
            statuses[o['status']]+=1
            if 'track_id'in o and 'world'in o:groups[o['track_id']].append(o)
            else:ignored.append(o)
    with np.load(root/'map_review/map_points.npz') as z:bg=z['background_xyz_map']
    vehicles=json.loads((root/'map_review/history_fit/report.json').read_text())['results']
    summaries=[];allraw=[];allopt=[];allframes=[];alltimes=[];allids=[];allflags=[]
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(18,14),dpi=140)
    ax.scatter(bg[:,1],bg[:,0],s=.15,c='#87939c',alpha=.3,linewidths=0);draw_vehicles(ax,vehicles)
    # Interactive map includes all trajectories; background only is a seeded random sample.
    rng=np.random.default_rng(20261015);b=bg[rng.choice(len(bg),min(100000,len(bg)),replace=False)]
    interactive=go.Figure(go.Scattergl(x=b[:,1],y=b[:,0],mode='markers',name='Map background',marker=dict(color='#87939c',size=1,opacity=.3),hoverinfo='skip'))
    for tid,obs in sorted(groups.items()):
        obs.sort(key=lambda o:o['time']);raw=np.array([o['world'][:2] for o in obs]);times=np.array([o['time'] for o in obs]);frames=np.array([o['frame'] for o in obs]);flags=np.array([o['status']=='accepted_temporal' for o in obs]);optimized=raw.copy()
        # Never smooth across ignored observations or time gaps.
        runs=[];current=[]
        for i in range(len(obs)):
            if not flags[i]:
                if current:runs.append(current);current=[]
                continue
            if current and times[i]-times[current[-1]]>.50001:runs.append(current);current=[]
            current.append(i)
        if current:runs.append(current)
        for run in runs:optimized[run]=smooth_track(times[run],raw[run])
        rgb=colorsys.hsv_to_rgb(tid*.61803398875%1,.7,1.);col='rgb(%d,%d,%d)'%tuple(round(v*255) for v in rgb)
        for run in runs:
            p=optimized[run];ax.plot(p[:,1],p[:,0],color=rgb,lw=1.1,alpha=.85);ax.scatter(raw[run,1],raw[run,0],color=rgb,s=3,alpha=.3,linewidths=0)
        if np.any(~flags):ax.scatter(raw[~flags,1],raw[~flags,0],c='#999',marker='x',s=8,alpha=.7)
        if flags.sum()>=10:
            start=optimized[np.flatnonzero(flags)[0]];ax.annotate(f'P{tid}',(start[1],start[0]),xytext=(4,4),textcoords='offset points',fontsize=7,color=rgb)
        custom=np.c_[frames,flags.astype(int)]
        interactive.add_trace(go.Scattergl(x=raw[:,1],y=raw[:,0],mode='markers',name=f'P{tid} raw',legendgroup=str(tid),marker=dict(size=4,color=col,opacity=.35),customdata=custom,hovertemplate='frame %{customdata[0]}<br>accepted %{customdata[1]}<br>X %{y:.2f}, Y %{x:.2f}<extra></extra>'))
        x=[];y=[];hover=[]
        for run in runs:
            x.extend(optimized[run,1].tolist()+[None]);y.extend(optimized[run,0].tolist()+[None]);hover.extend(frames[run].tolist()+[None])
        interactive.add_trace(go.Scattergl(x=x,y=y,mode='lines+markers',name=f'P{tid} optimized',legendgroup=str(tid),line=dict(color=col,width=2),marker=dict(size=3),customdata=hover,hovertemplate='frame %{customdata}<br>X %{y:.2f}, Y %{x:.2f}<extra></extra>'))
        summaries.append(dict(track_id=int(tid),observations=len(obs),accepted_observations=int(flags.sum()),first_frame=int(frames[0]),last_frame=int(frames[-1]),max_position_adjustment_m=float(np.linalg.norm(optimized-raw,axis=1).max())))
        allraw.append(raw);allopt.append(optimized);allframes.append(frames);alltimes.append(times);allids.append(np.full(len(obs),tid));allflags.append(flags)
    diagnostic=np.array([o['world'][:2] for o in ignored if 'world'in o]).reshape(-1,2)
    if len(diagnostic):ax.scatter(diagnostic[:,1],diagnostic[:,0],c='#aaa',marker='x',s=12,label='Rejected unassociated positions')
    ax.plot([],[],c='#00ffff',lw=2,label='Colored lines: optimized accepted XY tracks');ax.scatter([],[],c='#aaa',s=6,label='Faint dots: original XY / gray crosses: ignored')
    ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m), +X up');ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=9)
    ax.set_title('ZGC · all existing pedestrian tracks in recorded-odom map\nRobust trajectory review; no extrapolation, no ID merging, saved labels unchanged');fig.tight_layout();fig.savefig(out/'whole_map_person_tracks.png');plt.close(fig)
    interactive.update_layout(template='plotly_dark',title='All pedestrian tracks · original positions and optimized review trajectories',xaxis=dict(title='Map Y (m), +Y left',autorange='reversed'),yaxis=dict(title='Map X (m)',scaleanchor='x',scaleratio=1),height=1000)
    interactive.write_html(out/'interactive_tracks.html',include_plotlyjs=True)
    np.savez_compressed(out/'trajectories.npz',original_xy_map=np.concatenate(allraw),optimized_xy_map=np.concatenate(allopt),frame_ids=np.concatenate(allframes),times=np.concatenate(alltimes),track_ids=np.concatenate(allids),accepted_observation=np.concatenate(allflags))
    report=dict(status='complete',labels_modified=False,poses_modified=False,track_ids=len(groups),tracked_observations=sum(len(v) for v in groups.values()),accepted_observations=int(np.concatenate(allflags).sum()),person_status=dict(statuses),visual_only_or_unassociated_records=len(ignored),parameters=dict(observation_scale_m=.2,acceleration_scale_m_s2=3.,max_gap_s=.5),objective='soft-L1 observed XY residual / .2m + quadratic acceleration / 3m/s2; optimized per continuous accepted run',limits='Existing visual-confirmed instance associations only. Track IDs may fragment. No metric positions for visual-only detections without reliable depth; no interpolation or future prediction. Fits have no independent ground-truth accuracy assessment.',statistics_method='full population, no trajectory/observation subsampling; interactive background random seed20261015 at most100000',tracks=summaries)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>行人轨迹</title><style>body{background:#15191d;color:white;font:18px sans-serif;margin:30px}a{color:cyan}img{max-width:100%}</style><h1>全部行人轨迹：原始位置与优化候选</h1><p>彩色线：优化后的有效轨迹；淡色点：原始位置；灰色叉：忽略的定位。未修改标签，未对无可靠深度的视觉观测编造位置。已有 ID 可能包含同一行人的不同轨迹片段。</p><p><a href="interactive_tracks.html">交互式轨迹图：图例选择 P 编号，悬停查看帧号</a> · <a href="trajectories.npz">保存的轨迹 NPZ</a> · <a href="report.json">完整记录</a></p><img src="whole_map_person_tracks.png">')
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'},indent=2))


if __name__=='__main__':main()
