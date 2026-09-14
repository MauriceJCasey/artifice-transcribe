# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared token redaction helper for both the download and transcription services.

Every ``str(exc)`` boundary that can carry a credential — Hugging Face bearer
tokens in 401 messages from ``huggingface_hub``, or OpenAI-style ``sk-`` keys
from the inference config — must pass through :func:`redact_token` before it
is logged, stored, or served.
"""

from __future__ import annotations

import re

# Match ``hf_`` followed by 20+ alphanumeric / dash / underscore chars,
# or ``sk-`` / ``sk-ant-`` style OpenAI keys (the prefix separator is a
# hyphen, not an underscore).
_TOKEN_RE = re.compile(r"(?:hf_|sk-)[A-Za-z0-9_\-]{20,}")


def redact_token(text: str) -> str:
    """Replace any credential-bearing token in *text* with ``[REDACTED]``.

    Covers Hugging Face tokens (``hf_…``) and OpenAI-style keys
    (``sk-…``, ``sk-ant-…``).

    Safe to call on text that does not contain a token — returns *text*
    unchanged.
    """
    return _TOKEN_RE.sub("[REDACTED]", text)


_URL_USERINFO_RE = re.compile(r"://[^/\s@]+@")


def strip_url_userinfo(text: str) -> str:
    """Strip ``user:pass@`` from any URL embedded in *text*.

    A misconfigured base URL can carry credentials
    (``http://user:pass@host``); if the connection then fails, the raw
    exception text can include the full URL. Safe to call on text with no
    embedded URL — returns *text* unchanged.
    """
    return _URL_USERINFO_RE.sub("://", text)
