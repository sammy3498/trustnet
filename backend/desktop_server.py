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

root_path = str(
    ROOT_DIR
)

if root_path not in sys.path:
    sys.path.insert(
        0,
        root_path,
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