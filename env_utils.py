from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv(
    start: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> Path | None:
    base = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for directory in (base, *base.parents):
        env_path = directory / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            if verbose:
                print(f"[load_env] Loaded {env_path}")
            return env_path
    if verbose:
        print(f"[load_env] No .env found starting from {base}")
    return None
