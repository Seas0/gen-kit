"""swissroll shortcut; accepts the same --set overrides as scripts/main.py."""

from pathlib import Path

from diffusion.training import main


if __name__ == "__main__":
    main(default_config=str(Path(__file__).resolve().parents[1] / "config/swissroll.yaml"))
