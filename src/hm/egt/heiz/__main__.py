"""Command-line entry point for hm.egt.heiz."""

from __future__ import annotations

from hm.egt.heiz import __version__


def main() -> None:
    print(f"hm.egt.heiz v{__version__}")


if __name__ == "__main__":
    main()
