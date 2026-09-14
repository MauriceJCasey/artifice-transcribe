# SPDX-FileCopyrightText: 2026 Maurice Casey
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point for the frozen artifice-transcribe executable.

Imports ``main.cli()`` through the normal Python import system so that
relative imports (``from .web.window import ...``) inside ``main.py`` resolve
correctly.  Running ``main.py`` as a bare script loses the package context.

``cli()`` is the same entry point the installed ``artifice-transcribe``
console script binds to (``artifice_transcribe.main:cli``), so the frozen
binary behaves identically to ``uv run artifice-transcribe``.
"""

from artifice_transcribe.main import cli

if __name__ == "__main__":
    cli()
