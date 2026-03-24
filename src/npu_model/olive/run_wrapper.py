from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Olive with a JSON config dict.")
    parser.add_argument("--config", required=True, help="Path to Olive config JSON file")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from olive import run

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

