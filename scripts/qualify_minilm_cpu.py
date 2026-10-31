"""Development-only compatibility entry point for the packaged MiniLM probe.

The implementation is package-private so an installed wheel does not need to
publish the generic top-level ``scripts`` package.
"""

import importlib
from pathlib import Path
import sys

# This wrapper also supports legacy source tools that execute it as an
# independent file from outside the checkout, without an installed package.
# The sibling package code is still selected from this same checkout.
_checkout_root = str(Path(__file__).resolve().parents[1])
if _checkout_root not in sys.path:
    sys.path.insert(0, _checkout_root)

if __name__ == "__main__":
    from labyrinth.minilm_cpu_experiment import main

    raise SystemExit(main())
else:
    _module = importlib.import_module("labyrinth.minilm_cpu_experiment")
    # Some legacy tests/tools use spec_from_file_location and retain their
    # own module object instead of consulting sys.modules after exec_module.
    globals().update({key: value for key, value in vars(_module).items()
                      if not (key.startswith("__") and key.endswith("__"))})
    sys.modules[__name__] = _module
