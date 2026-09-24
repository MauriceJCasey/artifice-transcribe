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


_BUTTON_STYLE_JS = """() => [...document.querySelectorAll('button.btn')]
  .filter((b) => b.getBoundingClientRect().width > 0)
  .map((b) => {
    const cs = getComputedStyle(b);
    return {id: b.id || b.textContent.trim().slice(0, 24), shadow: cs.boxShadow,
            transform: cs.textTransform, border: cs.borderTopWidth};
  })
  .filter((s) => s.shadow !== 'none' || s.transform !== 'none' || s.border !== '1px')"""


def test_buttons_match_ocr_flat_sentence_case_hairline(ui):
    """Buttons had hard offset shadows, uppercase tracked labels and a heavy
    ink border; Artifice OCR's are flat, sentence case, with a hairline."""
    u = ui(byom_configured=True)
    assert u.page.evaluate(_BUTTON_STYLE_JS) == []

    u.tab("library")
    u.page.locator("#library-body [data-row-job]").first.click()
    expect(u.page.locator("#transcript-card")).to_be_visible()
    assert u.page.evaluate(_BUTTON_STYLE_JS) == []

    # Hover and press must not bring a shadow back either.
    save = u.page.locator("#btn-save-speakers")
    save.hover()
    u.page.wait_for_timeout(350)
    assert save.evaluate("el => getComputedStyle(el).boxShadow") == "none"
    u.page.mouse.down()
    assert save.evaluate("el => getComputedStyle(el).boxShadow") == "none"
    u.page.mouse.up()
    u.assert_clean()


def test_pane_before_a_transcript_is_open_does_not_deny_there_are_any(ui):
    """It said "No transcripts yet." directly beneath a library that listed one."""
    u = ui(byom_configured=True)
    u.tab("library")
    expect(u.page.locator("#library-body [data-row-job]")).to_have_count(1)
    expect(u.page.locator("#transcript-empty")).to_contain_text("No transcript open")
    expect(u.page.locator("#transcript-empty")).not_to_contain_text("No transcripts yet")
    u.assert_clean()


def test_wrapping_speaker_buttons_have_room_between_lines(ui):
    """ "Enroll selected as known speaker" wraps in the audio pane; at the
    inherited line-height of 1 its two lines overlapped."""
    u = ui(byom_configured=True)
    u.tab("library")
    u.page.locator("#library-body [data-row-job]").first.click()
    expect(u.page.locator("#transcript-card")).to_be_visible()
    ratio = u.page.locator("#btn-enroll-from-job").evaluate(
        """el => { const cs = getComputedStyle(el);
          return parseFloat(cs.lineHeight) / parseFloat(cs.fontSize); }"""
    )
    assert ratio >= 1.2, f"line-height is {ratio:.2f}x the font size"


def test_diarisation_guide_opens_and_closes_from_a_real_button(ui):
    """The help beside Model is a keyboard-reachable button that works.

    It was a click-only <span> containing U+24D8, which the bundled fonts lack
    (so it drew as a missing-glyph box), and nothing was listening to it.
    """
    u = ui(byom_configured=True)
    toggle = u.page.locator("#model-guide-toggle")
    panel = u.page.locator("#model-guide-panel")

    assert toggle.evaluate("el => el.tagName") == "BUTTON"
    assert toggle.get_attribute("aria-label") == "About speaker diarisation"
    assert "ⓘ" not in toggle.inner_text()
    expect(panel).to_be_hidden()

    toggle.focus()
    u.page.keyboard.press("Enter")
    expect(panel).to_be_visible()
    expect(toggle).to_have_attribute("aria-expanded", "true")

    u.page.keyboard.press("Escape")
    expect(panel).to_be_hidden()
    expect(toggle).to_have_attribute("aria-expanded", "false")

    toggle.click()
    expect(panel).to_be_visible()
    u.page.locator("#model-guide-close").click()
    expect(panel).to_be_hidden()
    expect(toggle).to_be_focused()
    u.assert_clean()


@pytest.mark.parametrize("width", [1024, 1200])
def test_every_tab_is_visible(ui, width):
    """The connection label collapses to its dot so all four tabs fit.

    With the full label, "People & dictionary" and "Settings" sat under the
    utility cluster at 1024px (and Settings was clipped at 1200px),
    reachable only by scrolling the tab rail.
    """
    u = ui(byom_configured=False)
    u.page.set_viewport_size({"width": width, "height": 800})
    state = u.page.evaluate(
        """() => {
          const nav = document.querySelector('.shell-titlebar-nav');
          const btn = document.querySelector('[data-shell-action="model"]');
          const label = btn.querySelector('[data-model-label]');
          return {
            railScrolls: nav.scrollWidth > nav.clientWidth + 1,
            labelShown: label.getBoundingClientRect().width > 2,
            name: btn.getAttribute('aria-label'),
            title: btn.getAttribute('title'),
          };
        }"""
    )
    assert state == {
        "railScrolls": False,
        "labelShown": False,
        "name": "Set up connection",
        "title": "Set up connection",
    }
    # Wide windows keep the full label.
    u.page.set_viewport_size({"width": 1440, "height": 800})
    assert u.page.locator("[data-model-label]").bounding_box()["width"] > 2
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
