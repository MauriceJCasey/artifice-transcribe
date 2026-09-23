# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic Playwright browser baseline for the Transcribe web UI.

Stage 0 of the research-instrument redesign (``UI_REDESIGN_PLAN.md``). These
tests serve the real Jinja templates and static assets from the real FastAPI
application, then intercept every ``/api/**`` request with synthetic fixtures.
No real user data, model loads, downloads or external services are involved.
"""
