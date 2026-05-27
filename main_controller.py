from __future__ import annotations

import sys
from pathlib import Path

from modules.gui import build_and_run_gui
from modules.utils import load_settings


def main() -> None:
    project_root = Path(__file__).resolve().parent
    settings_path = project_root / "config" / "settings.yaml"

    settings = load_settings(str(settings_path))
    build_and_run_gui(settings=settings, project_root=str(project_root))


if __name__ == "__main__":
    # Windows-friendly argv handling
    sys.exit(main())

