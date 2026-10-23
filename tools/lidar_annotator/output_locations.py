"""Shared visualization paths for CLI announcements and Qt links."""
from pathlib import Path
from .cli import REPO


def visualization_location(context,stage):
    out=Path(context['output'])
    backend=context.get('backend',{})
    temporal=Path(backend.get('temporal_output',out/'temporal'))
    if not temporal.is_absolute():temporal=REPO/temporal
    if stage=='semantic':
        source=Path(backend.get('source_index',out/'data/labels/zgc/index.json')).parent
        if not source.is_absolute():source=REPO/source
        return source/'overlay','图像语义叠加图，随帧处理逐张保存。'
    if stage=='optimize':return temporal/'map_review','地图与轨迹优化可视化。'
    if stage=='validate':return temporal/'map_review/sam3_full_video/interior_half_review/consistent_tracks','轨迹校验图。'
    if stage in ('render','catalog'):return Path(context['review_output'])/'visualization','行人、车辆与当前帧点云联合可视化。'
    return None,'本阶段保存标签、mask 或关联数据；联合图将在“逐帧联合可视化”阶段生成。'

