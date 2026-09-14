import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if getattr(
    sys,
    "frozen",
    False,
):
    RUNTIME_DIR = Path(
        sys._MEIPASS
    )
else:
    RUNTIME_DIR = ROOT_DIR


runtime_path = str(
    RUNTIME_DIR
)

if runtime_path not in sys.path:
    sys.path.insert(
        0,
        runtime_path,
    )


os.chdir(
    RUNTIME_DIR
)


env_file = os.getenv(
    "TRUSTNET_ENV_FILE",
)

if env_file:
    load_dotenv(
        dotenv_path=env_file,
        override=False,
    )
else:
    load_dotenv(
        override=False,
    )


import uvicorn

from backend.app.main import app


def main() -> None:
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()