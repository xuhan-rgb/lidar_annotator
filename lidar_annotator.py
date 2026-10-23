#!/usr/bin/env python3
"""Portable source-tree launcher for CLI and Qt desktop entry points."""
import sys

if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='demo':
            from tools.lidar_annotator.storage import demo
            demo(sys.argv[2:]);raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1] in ('annotate','annotate-stage'):
            from tools.lidar_annotator.fresh_pipeline import main
            main(sys.argv[2:],worker=sys.argv[1]=='annotate-stage');raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='lmdb':
            from tools.lidar_annotator.lmdb_input import main
            main(sys.argv[2:]);raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='export-labels':
            from tools.lidar_annotator.dataset_labels import main
            main(sys.argv[2:]);raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='export-kitti':
            from tools.lidar_annotator.kitti_input import main
            main(sys.argv[2:]);raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='self-test':
            from scripts.verify_bundle import main
            main(sys.argv[2:]);raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='install-desktop':
            from scripts.install_desktop import main
            raise SystemExit(main(sys.argv[2:]))
        if len(sys.argv)>1 and sys.argv[1]=='doctor':
            from tools.lidar_annotator.doctor import main
            raise SystemExit(main())
        if len(sys.argv)>1 and sys.argv[1]=='gui':
            from tools.lidar_annotator.qt_app import main
            raise SystemExit(main(sys.argv[2:]))
        from tools.lidar_annotator.cli import main
        raise SystemExit(main())
    except KeyboardInterrupt:
        if len(sys.argv)<2 or sys.argv[1]!='annotate-stage':
            print('已停止，已完成的结果保留。',file=sys.stderr)
        raise SystemExit(130)
