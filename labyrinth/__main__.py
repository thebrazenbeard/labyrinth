"""Allow ``python -m labyrinth`` without package installation."""

from .cli import main

raise SystemExit(main())
