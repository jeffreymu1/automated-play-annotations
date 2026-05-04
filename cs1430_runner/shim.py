from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_code_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    s = str(repo_root)
    if sys.path[0] != s:
        sys.path.insert(0, s)


def annotate_main() -> None:
    _ensure_project_code_on_path()
    from code.cli import main

    main()


def export_yolo_main() -> None:
    _ensure_project_code_on_path()
    from code.export_yolo import main

    main()


def train_yolo_main() -> None:
    _ensure_project_code_on_path()
    from code.train_yolo import main

    main()
