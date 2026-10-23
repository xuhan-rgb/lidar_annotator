"""Install a per-user desktop launcher for LiDAR Annotator."""
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / 'desktop' / 'lidar-annotator.desktop.in'


def quote_exec_arg(value):
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
    return f'"{escaped}"'


def escape_entry_value(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t').replace(' ', '\\s')


def install(applications_dir=None, project_root=PROJECT_ROOT):
    root = Path(project_root).expanduser().resolve()
    destination = Path(applications_dir).expanduser() if applications_dir is not None else Path.home() / '.local/share/applications'
    destination.mkdir(parents=True, exist_ok=True)
    content = TEMPLATE.read_text(encoding='utf-8')
    content = content.replace('@EXEC_RUN_SH@', quote_exec_arg(root / 'run.sh'))
    content = content.replace('@ICON_PATH@', escape_entry_value(root / 'assets/icons/lidar-annotator.svg'))
    target = destination / 'lidar-annotator.desktop'
    target.write_text(content, encoding='utf-8')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--applications-dir', type=Path, help='applications directory (default: ~/.local/share/applications)')
    args = parser.parse_args(argv)
    print(install(args.applications_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
