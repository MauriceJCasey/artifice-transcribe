# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stage 0 Transcribe browser baseline journeys.

Every test drives the real FastAPI app (Jinja templates + static assets) with
synthetic, deterministic API fixtures.  No real user data, model loads,
downloads or external services are involved.  Each journey ends by asserting
there were no unexpected API requests, browser page errors, console errors or
same-origin error responses.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs

import pytest
from playwright.sync_api import expect

from . import _fixtures as fx

pytestmark = pytest.mark.transcribe_ui


def _select_file(page, tmp_path, name: str = "interview_mrs_hamilton.wav") -> None:
    path = tmp_path / name
    path.write_bytes(fx.silent_wav())
    page.set_input_files("#file-input", path)


def test_empty_state_shell_loads(ui):
    """The shell renders with an empty Transcribe intake and empty library."""
    u = ui(populated=False, byom_configured=True)

    expect(u.page.locator(".app-shell")).to_be_visible()
    expect(u.page.locator(".shell-brand")).to_contain_text("Transcribe")
    expect(u.page.locator("#panel-transcribe")).to_be_visible()
    expect(u.page.locator("#dropzone")).to_be_visible()
    expect(u.page.locator("#btn-start")).to_be_disabled()
    expect(u.page.locator("#active-body")).to_contain_text("No active recordings yet.")

    u.tab("library")
    expect(u.page.locator("#library-body")).to_contain_text("No transcriptions yet.")

    u.tab("transcribe")
    u.screenshot("empty")
    u.assert_clean()


def test_research_shell_delivers_local_font_and_variant_styles(ui):
    """The opt-in shell exposes its variant and loads the offline font face."""
    u = ui(populated=False, byom_configured=True)

    state = u.page.evaluate(
        """async () => {
          await document.fonts.ready;
          const body = getComputedStyle(document.body);
          return {
            variant: document.documentElement.dataset.shellVariant,
            bodyFont: body.fontFamily,
            fontLoaded: document.fonts.check('16px "Libre Baskerville"'),
            researchCss: [...document.styleSheets].some((sheet) =>
              sheet.href && sheet.href.includes('/shared/research.css')),
          };
        }"""
    )
    assert state == {
        "variant": "research",
        "bodyFont": '"Libre Baskerville", Georgia, serif',
        "fontLoaded": True,
        "researchCss": True,
    }
    u.assert_clean()


@pytest.mark.parametrize("width", [1024, 1440])
def test_open_transcript_fits_its_card(ui, width):
    """The transcript column wraps instead of stretching the card sideways.

    A bare ``2fr`` track grew to the no-wrap edit toolbar's width, so text ran
    off the card and the metadata form spilled over the transcript.
    """
    u = ui(byom_configured=True)
    u.page.set_viewport_size({"width": width, "height": 900})
    u.tab("library")
    u.page.locator("#library-body [data-row-job]").first.click()
    expect(u.page.locator("#transcript-card")).to_be_visible()

    layout = u.page.evaluate(
        """() => {
          const box = (sel) => document.querySelector(sel).getBoundingClientRect();
          const panel = document.getElementById('panel-library');
          const card = box('#transcript-card'), pane = box('.audio-pane');
          const segs = [...document.querySelectorAll('#segments .seg-text')];
          return {
            panelScrollsSideways: panel.scrollWidth > panel.clientWidth + 1,
            segmentsPastCard: segs.some((s) => s.getBoundingClientRect().right > card.right + 1),
            metadataPastPane: box('.metadata-grid').right > pane.right + 1,
          };
        }"""
    )
    assert layout == {
        "panelScrollsSideways": False,
        "segmentsPastCard": False,
        "metadataPastPane": False,
    }
    u.assert_clean()


def test_research_navigation_preserves_deep_links_history_and_focus(ui):
    """Direct views and browser history remain authoritative after the rail move."""
    u = ui(populated=False, byom_configured=True)
    u.goto("/?view=settings")
    u.wait_ready()
    expect(u.page.locator("#panel-settings")).to_be_visible()
    legacy_tabs = u.page.locator(".tabs .tab").evaluate_all(
        "tabs => tabs.map(tab => ({tabIndex: tab.tabIndex, "
        "hidden: tab.getAttribute('aria-hidden')}))"
    )
    assert legacy_tabs and all(
        tab["tabIndex"] == -1 and tab["hidden"] == "true" for tab in legacy_tabs
    )

    u.page.go_back()
    expect(u.page.locator("#panel-transcribe")).to_be_visible()
    u.page.go_forward()
    expect(u.page.locator("#panel-settings")).to_be_visible()
    u.assert_clean()


def test_upload_review_edit_save_export_journey(ui, tmp_path):
    """The full journey: upload -> review -> edit/save -> rename -> export.

    Asserts the upload query payload, the segment-edit payload and the exported
    content, so a synthetic edit must survive all the way to the export body.
    """
    u = ui(byom_configured=True)

    # ── Upload with vocabulary and speaker limits ────────────────────────────
    u.page.locator("#opt-vocabulary").fill("Harland, Wolff, Belfast")
    u.page.locator("#opt-min-speakers").fill("2")
    u.page.locator("#opt-max-speakers").fill("2")
    _select_file(u.page, tmp_path)
    expect(u.page.locator("#btn-start")).to_be_enabled()
    u.page.locator("#btn-start").click()

    assert u.wait_until(lambda: bool(u.uploads)), "upload POST was not observed"
    assert len(u.uploads) == 1
    params = parse_qs(u.uploads[0]["query"])
    assert params.get("custom_vocabulary") == ["Harland, Wolff, Belfast"]
    assert params.get("min_speakers") == ["2"]
    assert params.get("max_speakers") == ["2"]
    expect(u.page.locator("#active-body")).to_contain_text("interview_mrs_hamilton.wav")

    # ── Review: the queued job completes and appears in the library ──────────
    u.tab("library")
    expect(u.page.locator("#library-body [data-row-job]")).to_have_count(1)
    u.page.locator("#library-body [data-row-job]").first.click()
    expect(u.page.locator("#transcript-card")).to_be_visible()
    expect(u.page.locator("#segments [data-seg-text]")).to_have_count(3)
    expect(u.page.locator("#segments")).to_contain_text("Mrs. Hamilton")

    u.screenshot("populated")

    # ── Edit a segment and save ──────────────────────────────────────────────
    seg = u.page.locator('#segments [data-seg-text="1"]')
    seg.fill(fx.EDITED_TEXT)
    seg.blur()
    expect(u.page.locator("#btn-save-edits")).to_be_enabled()
    u.page.locator("#btn-save-edits").click()
    expect(u.page.locator("#btn-save-edits")).to_be_disabled()

    assert len(u.segment_patches) == 1
    assert u.segment_patches[0] == {"segment_id": fx.SEGMENT_IDS[1], "text": fx.EDITED_TEXT}

    # ── Rename a speaker and save ────────────────────────────────────────────
    speaker_input = u.page.locator('[data-speaker-label="SPEAKER_01"]')
    speaker_input.fill("Mrs. M. Hamilton")
    u.page.locator("#btn-save-speakers").click()
    assert u.wait_until(lambda: bool(u.speaker_patches)), "speaker rename was not observed"
    assert {
        "speaker_label": "SPEAKER_01",
        "custom_name": "Mrs. M. Hamilton",
    } in u.speaker_patches[0]["speakers"]

    # ── Export and assert the edit reached the exported content ──────────────
    with u.context.expect_page() as popup_info:
        u.page.locator('[data-export="txt"]').click()
    popup = popup_info.value
    popup.wait_for_load_state("load")
    popup.close()

    assert u.wait_until(lambda: bool(u.exports)), "export request was not observed"
    txt_exports = [e for e in u.exports if e["format"] == "txt"]
    assert txt_exports, f"no txt export recorded: {u.exports}"
    body = txt_exports[0]["body"]
    assert fx.EDITED_TEXT in body, f"edited text missing from export: {body!r}"
    assert "Mrs. M. Hamilton" in body, f"renamed speaker missing from export: {body!r}"

    u.assert_clean()


def test_missing_speech_disables_intake(ui):
    """Required speech (ASR) missing: intake disables, library stays usable."""
    u = ui(asr_available=False, byom_configured=True)

    expect(u.page.locator("#asr-notice-transcribe")).to_be_visible()
    expect(u.page.locator("#asr-notice-transcribe-cmd")).to_have_text("uv sync --extra asr")
    expect(u.page.locator("#file-input")).to_be_disabled()
    expect(u.page.locator("#dropzone")).to_have_class(re.compile("dropzone-disabled"))
    expect(u.page.locator("#btn-start")).to_be_disabled()

    # Pure database work must stay enabled regardless of ASR availability.
    u.tab("library")
    expect(u.page.locator("#library-body")).to_be_visible()

    u.screenshot("missing-speech")
    u.assert_clean()


def test_optional_text_model_setup_is_inline_until_requested(ui):
    """An unconfigured optional model does not block intake or the library."""
    u = ui(byom_configured=False)
    expect(u.page.locator(".byom-overlay")).to_have_count(0)
    expect(u.page.locator('[data-shell-action="model"]')).to_have_attribute(
        "data-state", "unconfigured"
    )
    expect(u.page.locator("#dropzone")).to_be_visible()
    u.page.locator("#opt-vocabulary").fill("Harland, Wolff")
    u.page.locator('[data-shell-action="model"]').click()
    expect(u.page.locator(".byom-overlay")).to_be_visible()
    u.page.locator(".byom-close").click()
    expect(u.page.locator(".byom-overlay")).to_have_count(0)
    expect(u.page.locator("#opt-vocabulary")).to_have_value("Harland, Wolff")
    u.assert_clean()


def test_optional_text_model_setup_skips_when_configured(ui):
    """A configured optional text model does not auto-open the onboarding."""
    u = ui(byom_configured=True)
    expect(u.page.locator(".byom-overlay")).to_have_count(0)
    expect(u.page.locator('[data-shell-action="model"]')).to_have_attribute(
        "data-state", "configured"
    )
    u.assert_clean()


def test_settings_view_populated(ui):
    """Settings renders the read-only active model configuration."""
    u = ui(byom_configured=True)
    u.tab("settings")
    expect(u.page.locator("#setting-model-size")).to_have_value("base")
    expect(u.page.locator("#setting-device")).to_have_value("auto")
    u.screenshot("settings")
    u.assert_clean()


def test_model_download_consent_flow(ui):
    """The ASR model download dialog exercises consent and download fixtures."""
    u = ui(byom_configured=True)
    u.tab("settings")
    u.page.locator("#btn-manage-models").click()

    expect(u.page.locator("#download-modal-overlay")).to_be_visible()
    expect(u.page.locator("#dlg-model-list")).to_contain_text("openai/whisper-large-v3")
    expect(u.page.locator("#dlg-total-size")).not_to_be_empty()

    download_btn = u.page.locator("#dlg-btn-download")
    expect(download_btn).to_have_text("Download")
    expect(download_btn).to_be_enabled()
    download_btn.click()

    # The heartbeat-only synthetic SSE ends, driving the dialog to a terminal
    # state; by then the consent + download POSTs have been dispatched.
    expect(u.page.locator("#dlg-btn-download")).to_have_text("Retry", timeout=10_000)
    # By reference, not a "key": "<literal>" pair, which gitleaks reads as an API key.
    whisper = fx.ASR_MODEL_KEYS[0]
    assert u.consents == [{"key": whisper, "consent": True}]
    assert u.downloads == [whisper]

    u.page.locator("#dlg-close").click()
    expect(u.page.locator("#download-modal-overlay")).to_be_hidden()
    u.screenshot("model-download")
    u.assert_clean()


def test_model_download_token_gated(ui):
    """A gated model requires an HF token before Download is enabled."""
    u = ui(model_keys=["pyannote-speaker-diarization"], byom_configured=True)
    u.tab("settings")
    u.page.locator("#btn-manage-models").click()

    expect(u.page.locator("#download-modal-overlay")).to_be_visible()
    expect(u.page.locator("#dlg-model-list")).to_contain_text("pyannote/speaker-diarization-3.0")
    expect(u.page.locator("#dlg-token-section")).to_be_visible()
    expect(u.page.locator("#dlg-btn-download")).to_have_text("Continue")
    expect(u.page.locator("#dlg-btn-download")).to_be_disabled()

    u.page.locator("#dlg-token-input").fill("hf_synthetic")
    expect(u.page.locator("#dlg-btn-download")).to_be_enabled()

    u.assert_clean()
