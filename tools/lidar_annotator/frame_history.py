"""Persistent frame-test history and an editable dropdown for exact result runs."""
import json,os,time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import Signal,Qt
from PySide6.QtWidgets import QComboBox,QSizePolicy
from .frame_selection import parse_frames
from .storage import directory


def history_path():
    return Path(os.environ.get('XDG_STATE_HOME',Path.home()/'.local/state'))/'lidar_annotator/test_history.json'


def frame_text(frames):
    if frames is None:return ''
    ids=sorted(set(map(int,frames)))
    if not ids:return ''
    groups=[];start=previous=ids[0]
    for value in ids[1:]:
        if value!=previous+1:
            groups.append(str(start) if start==previous else f'{start}-{previous}');start=value
        previous=value
    groups.append(str(start) if start==previous else f'{start}-{previous}')
    return ' '.join(groups)


def read_records():
    try:return json.loads(history_path().read_text())['entries']
    except (OSError,ValueError,KeyError):return []


def write_records(entries):
    path=history_path();path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('history_'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps({'entries':entries},ensure_ascii=False,indent=2));temporary.replace(path)


def record_test(source,frames,context,status='running'):
    source=str(Path(source).expanduser().resolve());entries=read_records()
    entries=[e for e in entries if not (e['source']==source and e['context']['output']==context['output'])]
    entries.append(dict(source=source,frames_text=frame_text(frames),context=context,started=time.time(),status=status))
    write_records(entries)


def finish_test(context,status):
    entries=read_records();changed=False
    for entry in entries:
        if entry['context']['output']==context['output']:entry['status']=status;changed=True
    if changed:write_records(entries)


def test_history(source):
    source=str(Path(source).expanduser().resolve())
    entries={e['context']['output']:e for e in read_records() if e['source']==source}
    debug=directory('jobs')/'debug'
    owners=list(debug.glob('*/job.json'))+list(debug.glob('*_runs/*/job.json'))+list(debug.glob('*/image_*/job.json'))+list(debug.glob('*/image_*_runs/*/job.json'))
    for owner in owners:
        try:
            job=json.loads(owner.read_text())
            if job.get('schema')!='lidar_fresh_job_v1' or str(Path(job['source']).resolve())!=source:continue
            out=owner.parent;key=str(out)
            report=json.loads((out/'run_report.json').read_text()) if (out/'run_report.json').is_file() else {}
            if key in entries:
                if report.get('status'):entries[key]['status']=report['status']
                continue
            entries[key]=dict(source=source,frames_text=frame_text(job['frame_ids']),started=report.get('started_unix',owner.stat().st_mtime),status=report.get('status','unknown'),context=dict(output=key,review_output=str(out/'review'),input_format=job.get('input_format','lmdb'),temporary_debug=True,**({'camera':job['camera']} if 'camera' in job else {})))
        except (OSError,ValueError,KeyError,TypeError):continue
    return sorted(entries.values(),key=lambda e:e['started'],reverse=True)


class FrameHistoryBox(QComboBox):
    entry_selected=Signal(object)
    history_requested=Signal()

    def __init__(self,parent=None):
        super().__init__(parent);self.setEditable(True);self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon);self.setMinimumContentsLength(6)
        self.setMinimumWidth(0);self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.activated.connect(self.select_entry)

    @property
    def textChanged(self):return self.editTextChanged
    @property
    def editingFinished(self):return self.lineEdit().editingFinished
    def text(self):return self.currentText()
    def setText(self,text):self.setEditText(text)
    def clear(self):self.lineEdit().clear()
    def setPlaceholderText(self,text):self.lineEdit().setPlaceholderText(text)
    def placeholderText(self):return self.lineEdit().placeholderText()

    def set_history(self,entries):
        text=self.text();blocked=self.blockSignals(True);line_blocked=self.lineEdit().blockSignals(True)
        super().clear()
        names={'running':'运行中','complete':'已完成','failed':'失败','cancelled':'已停止','unknown':'已有结果'}
        for entry in entries:
            label=(entry['frames_text'] or '全部帧')+' · '+datetime.fromtimestamp(entry['started']).strftime('%m/%d %H:%M:%S')+' · '+names.get(entry['status'],entry['status'])
            self.addItem(label,entry);self.setItemData(self.count()-1,entry['context']['output'],Qt.ItemDataRole.ToolTipRole)
        self.setCurrentIndex(-1);self.setEditText(text)
        self.lineEdit().blockSignals(line_blocked);self.blockSignals(blocked)

    def select_entry(self,index):
        entry=self.itemData(index)
        if entry is not None:self.setEditText(entry['frames_text']);self.entry_selected.emit(entry)

    def showPopup(self):
        self.history_requested.emit();super().showPopup()
