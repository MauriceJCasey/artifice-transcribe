// SPDX-FileCopyrightText: 2026 Maurice Casey
// SPDX-License-Identifier: AGPL-3.0-or-later
(function () {
  "use strict";
  const sizeKey = "artifice.transcribe.review-text-size";
  const allowed = new Set(["16px", "18px", "20px", "24px"]);
  let size = "18px";
  try { const stored = localStorage.getItem(sizeKey); if (allowed.has(stored)) size = stored; } catch (_) {}
  const audio = document.getElementById("audio-player");
  const speed = document.getElementById("playback-speed");
  if (speed && audio) speed.addEventListener("change", () => { audio.playbackRate = Number(speed.value); });
  function apply(next) {
    if (!allowed.has(next)) return;
    size = next;
    document.documentElement.style.setProperty("--transcript-text-size", size);
    document.querySelectorAll("[data-transcribe-reading-size]").forEach((select) => { select.value = size; });
    try { localStorage.setItem(sizeKey, size); } catch (_) {}
  }
  document.addEventListener("change", (event) => {
    if (event.target.matches("[data-transcribe-reading-size]")) apply(event.target.value);
  });
  apply(size);
})();
