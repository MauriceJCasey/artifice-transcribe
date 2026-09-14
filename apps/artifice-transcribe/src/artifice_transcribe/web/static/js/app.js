// SPDX-FileCopyrightText: 2026 Maurice Casey
//
// SPDX-License-Identifier: AGPL-3.0-or-later

(() => {
  'use strict';

  const API = '/api/v1';

  const $ = (id) => document.getElementById(id);
  const escapeHtml = (s) => (s ?? '').toString().replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));

  // ------------------------------------------------------------------ toast

  async function api(path, options = {}) {
    const resp = await fetch(`${API}${path}`, options);
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        detail = body.detail || detail;
      } catch (_) { /* no json body */ }
      throw new Error(detail);
    }
    if (resp.status === 204) return null;
    if (resp.headers.get('content-type')?.includes('application/json')) {
      return resp.json();
    }
    return resp;
  }

  // -------------------------------------------------------- ASR availability

  // GET /api/v1/capabilities never imports torch to answer (base install is
  // ~207 MB with no PyTorch/CUDA stack). asr.available === false means
  // Whisper transcription and pyannote diarisation are not installed.
  // Everything that is pure database work — Library, every export button,
  // transcript editing, the Dictionary tab, and the BYOM summarise/cleanup
  // calls — does not depend on this and must stay enabled regardless.
  // See .briefs/ui-ux-asr-unavailable-state.md.
  let asrAvailable = true;

  // `withCommand` only applies to the Transcribe tab's notice: it is the one
  // place the install command appears, copyable, so the same string is not
  // repeated three times on one page. The other two notices are same-panel
  // local copies of the reason text only, required because aria-describedby
  // pointing across a hidden ("display:none") tab panel announces nothing —
  // Settings and the Library tab's Speakers form each need their own.
  function renderAsrNotice(el, asr, { withCommand = false } = {}) {
    if (!el) return;
    const reason = (asr && asr.reason) || 'The transcription stack is not installed.';
    const reasonEl = el.querySelector('[id$="-reason"]');
    if (withCommand) {
      const hint = (asr && asr.install_hint) || '';
      if (reasonEl) reasonEl.textContent = reason;
      const cmdEl = el.querySelector('[id$="-cmd"]');
      if (cmdEl) cmdEl.textContent = hint;
      const copyBtn = el.querySelector('.asr-unavailable-copy');
      if (copyBtn) {
        copyBtn.dataset.copy = hint;
        copyBtn.closest('.byom-code-row')?.classList.toggle('hidden', !hint);
      }
    } else if (reasonEl) {
      reasonEl.textContent = `${reason} See the Transcribe tab for the install command.`;
    }
    el.classList.remove('hidden');
  }

  function applyCapabilities(cap) {
    const asr = (cap && cap.asr) || { available: true };
    asrAvailable = asr.available !== false;

    if (!asrAvailable) {
      renderAsrNotice($('asr-notice-transcribe'), asr, { withCommand: true });
      renderAsrNotice($('asr-notice-settings'), asr);
      renderAsrNotice($('asr-notice-speakers'), asr);
      renderAsrNotice($('asr-notice-health'), asr);
      renderAsrNotice($('asr-notice-enroll'), asr);
    }

    [
      'file-input', 'batch-file-input', 'btn-batch-upload',
      'btn-match-speakers', 'btn-enroll-from-job', 'setting-model-size',
      // btn-apply-model: PATCH /config always includes whisper_model, which
      // always triggers _reload_engine_with_new_model server-side — on a
      // base install that is a guaranteed 503, so the control has no
      // non-error outcome (routes.py:475-497).
      'btn-apply-model',
      // btn-load-models: POST /health/preload calls _get_engine() directly
      // (routes.py:747-755) — same category as above.
      'btn-load-models',
      // btn-enroll-speaker: POST /speakers/enroll extracts a pyannote
      // embedding via _get_engine() unconditionally (routes.py:1373-1414) —
      // same category. Not in the original brief's list; found while
      // auditing for other ASR-only controls per the coordinator's request.
      'btn-enroll-speaker',
    ].forEach((id) => {
      const el = $(id);
      if (el) el.disabled = !asrAvailable;
    });
    // btn-start / btn-clear-file are left to initUploadForm's setFile() guard
    // below: they start `disabled` in the markup and only ever become
    // enabled once a file is picked, which the guard now refuses to do.

    const dropzone = $('dropzone');
    if (dropzone) {
      dropzone.classList.toggle('dropzone-disabled', !asrAvailable);
      // Only asserted when true, never `aria-disabled="false"` — the full
      // install must not gain an attribute it never had (point 2 of the
      // brief's deliverable: byte-for-byte unchanged when available).
      if (!asrAvailable) dropzone.setAttribute('aria-disabled', 'true');
      else dropzone.removeAttribute('aria-disabled');
    }
  }

  function initAsrCopyButtons() {
    // The button's accessible name is the fixed aria-label "Copy install
    // command" (set in the template) so it doesn't change when the visible
    // "Copy"/"Copied" span does — matching shared-ui/byom.js's .byom-copy.
    // That means the state change needs its own announcement, hence
    // #asr-copy-status: a visually-hidden aria-live region, not the toast
    // (the toast container carries no aria-live and is not itself
    // announced — a pre-existing gap in this app, out of scope here).
    const srStatus = $('asr-copy-status');
    document.querySelectorAll('.asr-unavailable-copy').forEach((btn) => {
      btn.addEventListener('click', () => {
        const text = btn.dataset.copy || '';
        if (!text) return;
        const label = btn.querySelector('span');
        navigator.clipboard.writeText(text).then(() => {
          window.ArtificeToast.success('Copied to clipboard', {duration: 3600});
          if (srStatus) srStatus.textContent = 'Command copied to clipboard.';
          btn.setAttribute('data-copied', 'true');
          if (label) label.textContent = 'Copied';
          setTimeout(() => {
            btn.removeAttribute('data-copied');
            if (label) label.textContent = 'Copy';
          }, 1500);
        }).catch(() => {
          if (label) label.textContent = 'Copy failed';
          if (srStatus) srStatus.textContent = 'Could not copy the command — copy it manually.';
          window.ArtificeToast.warning('Could not copy — copy it manually', {duration: 3600});
          setTimeout(() => { if (label) label.textContent = 'Copy'; }, 1500);
        });
      });
    });
  }

  async function loadCapabilities() {
    try {
      const cap = await api('/capabilities');
      applyCapabilities(cap);
    } catch (_) {
      // The endpoint is designed to answer without importing torch, so this
      // should not fail in practice. Fail open (leave controls as shipped)
      // rather than disabling working functionality on a transient error.
    }
  }

  // ----------------------------------------------------------- keyboard shortcuts

  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl+S or Ctrl+Enter: save edits
      if ((e.ctrlKey && e.key === 's') || (e.ctrlKey && e.key === 'Enter')) {
        e.preventDefault();
        saveEdits();
      }
      // Ctrl+Z: revert edits
      if (e.ctrlKey && e.key === 'z') {
        if (!e.target.closest('[contenteditable]')) {
          e.preventDefault();
          revertEdits();
        }
      }
      // /: focus global search (when not in an input)
      if (e.key === '/' && !e.target.closest('input, textarea, [contenteditable]')) {
        e.preventDefault();
        const searchInput = $('global-search-input');
        if (searchInput) { searchInput.focus(); searchInput.select(); }
      }
      // Escape: close modals
      if (e.key === 'Escape') {
        closeSearchModal();
        $('history-panel')?.classList.add('hidden');
      }
      // Alt+S: split segment at cursor
      if (e.altKey && e.key === 's') {
        e.preventDefault();
        splitSegment();
      }
      // Alt+M: merge segment with next
      if (e.altKey && e.key === 'm') {
        e.preventDefault();
        mergeSegment();
      }
      // Tab/Shift+Tab: navigate between editable segments
      if (e.key === 'Tab' && e.target.closest('[data-seg-text]')) {
        e.preventDefault();
        const segs = [...document.querySelectorAll('[data-seg-text]')];
        const idx = segs.indexOf(e.target);
        if (idx >= 0) {
          const next = e.shiftKey ? segs[Math.max(0, idx - 1)] : segs[Math.min(segs.length - 1, idx + 1)];
          next.focus();
          // Move cursor to end
          const range = document.createRange();
          range.selectNodeContents(next);
          range.collapse(false);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        }
      }
      // Arrow Up/Down: navigate between segments from the focused contenteditable
      if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && !e.ctrlKey && !e.altKey && !e.metaKey) {
        const focused = e.target.closest('[data-seg-text]');
        if (!focused) return;
        const all = [...document.querySelectorAll('[data-seg-text]')];
        const idx = all.indexOf(focused);
        if (idx < 0) return;
        e.preventDefault();
        const next = e.key === 'ArrowDown' ? all[Math.min(all.length - 1, idx + 1)] : all[Math.max(0, idx - 1)];
        next.focus();
        const range = document.createRange();
        range.selectNodeContents(next);
        range.collapse(e.key === 'ArrowUp');
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        next.closest('[data-seg-index]')?.scrollIntoView({ block: 'nearest' });
      }
    });
  }

  // ------------------------------------------------------------------ tabs

  function initTabs() {
    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        tab.classList.add('active');
        $(`panel-${tab.dataset.tab}`).classList.add('active');
        if (tab.dataset.tab === 'library') loadLibrary();
      });
    });

    const requested = new URLSearchParams(window.location.search).get('view');
    const requestedTab = document.querySelector(`.tab[data-tab="${requested}"]`);
    if (requestedTab) requestedTab.click();
  }

  // ----------------------------------------------------------------- theme

  function initTheme() {
    const stored = localStorage.getItem('pt-theme');
    const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
    const initial = stored || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', initial);
    const btn = document.getElementById('themeToggle');
    if (btn) {
      btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('pt-theme', next);
      });
    }
  }

  // ------------------------------------------------------------ formatting

  function formatClock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${r.toString().padStart(2, '0')}`;
  }

  function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString();
  }

  function statusPill(status) {
    return `<span class="status-pill ${status}">${status}</span>`;
  }

  // ------------------------------------------------------------- dropzone

  let selectedFile = null;

  function initUploadForm() {
    const dropzone = $('dropzone');
    const fileInput = $('file-input');
    const dropzoneText = $('dropzone-text');
    const btnStart = $('btn-start');
    const btnClear = $('btn-clear-file');

    const setFile = (file) => {
      // Guards every entry point (click-to-browse, the change event, and a
      // direct drag-drop onto the dropzone div, which bypasses file-input's
      // own `disabled` attribute since it never touches that element).
      if (!asrAvailable) return;
      selectedFile = file;
      if (file) {
        dropzoneText.innerHTML = `<span class="picked">${escapeHtml(file.name)}</span> selected`;
        btnStart.disabled = false;
        btnClear.disabled = false;
      } else {
        dropzoneText.textContent = 'Drop an audio file here, or click to browse';
        btnStart.disabled = true;
        btnClear.disabled = true;
      }
    };

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => setFile(fileInput.files[0] || null));

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); if (asrAvailable) dropzone.classList.add('drag'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag');
      const file = e.dataTransfer.files[0];
      if (file) setFile(file);
    });

    btnClear.addEventListener('click', () => { fileInput.value = ''; setFile(null); });
    btnStart.addEventListener('click', startTranscription);

    // Batch upload
    $('btn-batch-upload').addEventListener('click', () => {
      $('batch-file-input').click();
    });
    $('batch-file-input').addEventListener('change', async (e) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      const btn = $('btn-batch-upload');
      btn.disabled = true;
      btn.textContent = 'Uploading...';
      let queued = 0;
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        try {
          const resp = await fetch(`${API}/transcribe`, { method: 'POST', body: formData });
          if (resp.ok) {
            const created = await resp.json();
            trackActiveJob(created.job_id, file.name);
            queued++;
          }
        } catch (_) { /* skip failed */ }
      }
      btn.disabled = false;
      btn.textContent = 'Batch Upload';
      window.ArtificeToast.success(`Queued ${queued} file(s)`, {duration: 3600});
      e.target.value = '';
    });
  }

  async function startTranscription() {
    if (!selectedFile) return;
    const btnStart = $('btn-start');
    const statusEl = $('upload-status');
    btnStart.disabled = true;
    statusEl.textContent = 'Uploading…';

    const params = new URLSearchParams();
    const language = $('opt-language').value.trim();
    const minSpeakers = $('opt-min-speakers').value;
    const maxSpeakers = $('opt-max-speakers').value;
    const vocabulary = $('opt-vocabulary').value.trim();
    if (language) params.set('language', language);
    if (minSpeakers) params.set('min_speakers', minSpeakers);
    if (maxSpeakers) params.set('max_speakers', maxSpeakers);
    if (vocabulary) params.set('custom_vocabulary', vocabulary);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const resp = await fetch(`${API}/transcribe?${params.toString()}`, {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error(`Upload failed (${resp.status})`);
      const created = await resp.json();
      window.ArtificeToast.success(`Queued: ${selectedFile.name}`, {duration: 3600});
      trackActiveJob(created.job_id, selectedFile.name);
      $('file-input').value = '';
      selectedFile = null;
      $('dropzone-text').textContent = 'Drop an audio file here, or click to browse';
      $('btn-clear-file').disabled = true;
    } catch (err) {
      window.ArtificeToast.error(err.message);
    } finally {
      statusEl.textContent = '';
      btnStart.disabled = !selectedFile;
    }
  }

  // -------------------------------------------------------------- active jobs

  const activeJobs = new Map();
  const pollTimers = new Map();

  function trackActiveJob(jobId, filename) {
    activeJobs.set(jobId, {
      id: jobId, filename, status: 'queued', progress_percentage: 0, created_at: new Date().toISOString(),
    });
    renderActiveJobs();
    pollJob(jobId);
  }

  function pollJob(jobId) {
    if (pollTimers.has(jobId)) return;
    const tick = async () => {
      try {
        const job = await api(`/jobs/${jobId}`);
        activeJobs.set(jobId, job);
        renderActiveJobs();
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(pollTimers.get(jobId));
          pollTimers.delete(jobId);
          if (job.status === 'completed') {
            window.ArtificeToast.success(`Finished: ${job.filename}`, {duration: 3600});
          } else {
            window.ArtificeToast.error(`Failed: ${job.filename} — ${job.error_message || 'unknown error'}`);
          }
          if ($('panel-library').classList.contains('active')) loadLibrary();
        }
      } catch (err) {
        clearInterval(pollTimers.get(jobId));
        pollTimers.delete(jobId);
      }
    };
    pollTimers.set(jobId, setInterval(tick, 2000));
    tick();
  }

  function renderActiveJobs() {
    const body = $('active-body');
    const jobs = Array.from(activeJobs.values());
    const running = jobs.filter((j) => j.status === 'queued' || j.status === 'processing');
    $('active-count').textContent = `${running.length} running`;

    if (jobs.length === 0) {
      body.innerHTML = '<tr><td colspan="5" class="table-empty-cell"><span class="panel-empty-title">No active jobs yet.</span><span class="panel-empty-desc">Upload an audio file above to begin.</span></td></tr>';
      return;
    }

    body.innerHTML = jobs.slice().reverse().map((j) => `
      <tr>
        <td>${escapeHtml(j.filename)}</td>
        <td class="c">${statusPill(j.status)}</td>
        <td class="c">
          <span class="mini-progress"><div style="width:${j.progress_percentage || 0}%"></div></span>
          <span class="dim">${Math.round(j.progress_percentage || 0)}%</span>
        </td>
        <td>${formatDate(j.created_at)}</td>
        <td class="c"><button class="btn" data-view-job="${j.id}" ${j.status === 'completed' ? '' : 'disabled'}>View</button></td>
      </tr>
    `).join('');

    body.querySelectorAll('[data-view-job]').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelector('.tab[data-tab="library"]').click();
        selectJob(btn.dataset.viewJob);
      });
    });
  }

  // ------------------------------------------------------------------ library

  let libraryJobs = [];

  async function loadLibrary() {
    const body = $('library-body');
    try {
      libraryJobs = await api('/jobs');
    } catch (err) {
      body.innerHTML = `<tr><td colspan="5" class="dim" style="text-align:center;">${escapeHtml(err.message)}</td></tr>`;
      return;
    }
    renderLibrary();
  }

  function renderLibrary() {
    const body = $('library-body');
    if (libraryJobs.length === 0) {
      body.innerHTML = '<tr><td colspan="5" class="table-empty-cell"><span class="panel-empty-title">No transcriptions yet.</span><span class="panel-empty-desc">Completed jobs will appear here.</span></td></tr>';
      return;
    }
    body.innerHTML = libraryJobs.map((j) => {
      const errorHtml = j.error_message
        ? `<div class="job-error-msg">${escapeHtml(j.error_message)}</div>`
        : '';
      const interviewee = j.interviewee || '';
      return `
      <tr class="clickable ${j.id === currentJobId ? 'selected' : ''}" data-row-job="${j.id}">
        <td>${escapeHtml(j.filename)}${errorHtml}</td>
        <td class="c">${statusPill(j.status)}</td>
        <td>${formatDate(j.created_at)}</td>
        <td class="meta-cell">${escapeHtml(interviewee)}</td>
        <td class="c"><button class="btn danger" data-delete-job="${j.id}">Delete</button></td>
      </tr>`;
    }).join('');

    body.querySelectorAll('[data-row-job]').forEach((row) => {
      row.addEventListener('click', (e) => {
        if (e.target.closest('[data-delete-job]')) return;
        const job = libraryJobs.find((j) => j.id === row.dataset.rowJob);
        if (job && job.status !== 'completed') {
          window.ArtificeToast.warning(`Job is ${job.status}, not completed yet.`, {duration: 3600});
          return;
        }
        selectJob(row.dataset.rowJob);
      });
    });
    body.querySelectorAll('[data-delete-job]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteJob(btn.dataset.deleteJob);
      });
    });
  }

  // --------------------------------------------------------------- transcript

  let currentJobId = null;
  let currentSegments = [];
  let segmentEdits = new Map();
  let diffMode = false;
  let inTranscriptSearchTerm = '';

  async function selectJob(jobId) {
    if (segmentEdits.size > 0) {
      if (!confirm('You have unsaved edits. Discard changes?')) return;
    }
    try {
      const [job, transcript, speakers] = await Promise.all([
        api(`/jobs/${jobId}`),
        api(`/jobs/${jobId}/transcript`),
        api(`/jobs/${jobId}/speakers`),
      ]);
      currentJobId = jobId;
      currentSegments = transcript.segments;
      segmentEdits.clear();
      diffMode = false;
      inTranscriptSearchTerm = '';
      renderLibrary();
      renderTranscript(job, transcript, speakers);
      fillMetadataForm(job);
      loadWaveform();
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  function fillMetadataForm(job) {
    $('meta-interviewee').value = job.interviewee || '';
    $('meta-interviewer').value = job.interviewer || '';
    $('meta-date').value = job.interview_date || '';
    $('meta-location').value = job.location || '';
    $('meta-project').value = job.project_name || '';
    $('meta-collection').value = job.collection_id || '';
    $('meta-access').value = job.access_restrictions || '';
    $('meta-vocabulary').value = job.custom_vocabulary || '';
  }

  function renderTranscript(job, transcript, speakers) {
    $('transcript-empty').classList.add('hidden');
    $('transcript-card').classList.remove('hidden');

    $('transcript-title').textContent = job.filename;
    const pill = $('transcript-status-pill');
    pill.className = `status-pill ${job.status}`;
    pill.textContent = job.status;

    const audio = $('audio-player');
    audio.src = `${API}/jobs/${job.id}/audio`;

    // Speaker rename rows
    const rowsEl = $('speaker-rows');
    if (speakers.speakers.length === 0) {
      rowsEl.innerHTML = '<p class="dim">No speakers detected.</p>';
    } else {
      rowsEl.innerHTML = speakers.speakers.map((s, i) => `
        <div class="speaker-row">
          <span class="speaker-swatch" style="background:${swatchColor(i)}"></span>
          <input type="text" data-speaker-label="${escapeHtml(s.speaker_label)}" value="${escapeHtml(s.custom_name)}">
        </div>
      `).join('');
    }

    // Map speaker labels to swatch colors
    const speakerColorMap = {};
    speakers.speakers.forEach((s, i) => {
      speakerColorMap[s.custom_name] = swatchColor(i);
    });

    // Metadata badge
    const badgeParts = [job.interviewee, job.interview_date, job.project_name].filter(Boolean);
    $('meta-badge').textContent = badgeParts.length ? badgeParts.join(' \u00b7 ') : '';

    // Segments
    const segEl = $('segments');
    if (transcript.segments.length === 0) {
      segEl.innerHTML = '<p class="dim">No segments.</p>';
    } else {
      segEl.innerHTML = transcript.segments.map((s, i) => {
        const edit = segmentEdits.get(i);
        const displayText = edit ? edit.editedText : s.text;
        const editedClass = edit ? ' edited' : '';
        const tags = s.tags || [];
        const spColor = speakerColorMap[s.speaker_label] || 'transparent';
        return `
        <div class="segment speaker-border${editedClass}" data-seg-index="${i}" data-seg-id="${escapeHtml(s.id)}" data-start="${s.start_time}" style="--speaker-color:${spColor}">
          <span class="seg-time">${formatClock(s.start_time)}</span>
          <div class="seg-body">
            <div class="seg-speaker" style="color:${spColor}">${escapeHtml(s.speaker_label)}</div>
            <div class="seg-text" contenteditable="true" data-seg-text="${i}">${escapeHtml(displayText)}</div>
            <div class="seg-tags" data-tags-for="${i}">
              ${tags.map((t) => `<span class="seg-tag" data-tag="${escapeHtml(t)}">${escapeHtml(t)}</span>`).join('')}
              <span class="seg-tag-add" data-add-tag="${i}">+ tag</span>
            </div>
          </div>
          ${edit ? '<span class="seg-edited-badge" title="Edited">edited</span>' : ''}
        </div>`;
      }).join('');

      // Click to seek
      segEl.querySelectorAll('[data-seg-index]').forEach((el) => {
        el.addEventListener('click', (e) => {
          if (e.target.closest('[contenteditable], .seg-tag, .seg-tag-add')) return;
          audio.currentTime = parseFloat(el.dataset.start);
          audio.play().catch(() => {});
        });
      });

      // Track edits on blur
      segEl.querySelectorAll('[data-seg-text]').forEach((el) => {
        el.addEventListener('blur', () => {
          const idx = parseInt(el.dataset.segText, 10);
          const seg = currentSegments[idx];
          if (!seg) return;
          const original = seg.text;
          const edited = el.textContent.trim();
          if (edited !== original) {
            segmentEdits.set(idx, { originalText: original, editedText: edited });
          } else {
            segmentEdits.delete(idx);
          }
          updateEditToolbar();
        });
        el.addEventListener('click', (e) => e.stopPropagation());
        el.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
        });
      });

      // Tag add
      segEl.querySelectorAll('[data-add-tag]').forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const idx = parseInt(btn.dataset.addTag, 10);
          const seg = currentSegments[idx];
          if (!seg) return;
          const tag = prompt('Add tag (e.g. anecdote, reflection):');
          if (tag && tag.trim()) {
            const tags = seg.tags || [];
            if (!tags.includes(tag.trim())) {
              tags.push(tag.trim());
              updateSegmentTags(idx, tags);
            }
          }
        });
      });

      // Tag remove
      segEl.querySelectorAll('[data-tag]').forEach((el) => {
        el.addEventListener('click', (e) => {
          e.stopPropagation();
          const tagEl = e.target;
          const segEl2 = tagEl.closest('[data-seg-index]');
          if (!segEl2) return;
          const idx = parseInt(segEl2.dataset.segIndex, 10);
          const seg = currentSegments[idx];
          if (!seg) return;
          const tag = tagEl.dataset.tag;
          const tags = (seg.tags || []).filter((t) => t !== tag);
          updateSegmentTags(idx, tags);
        });
      });
    }

    // Stats footer
    const segments = transcript.segments;
    const totalWords = segments.reduce((sum, s) => sum + (s.text ? s.text.split(/\s+/).filter(Boolean).length : 0), 0);
    const totalDuration = segments.length ? (segments[segments.length - 1].end_time - segments[0].start_time) : 0;
    let speakerChanges = 0;
    for (let i = 1; i < segments.length; i++) {
      if (segments[i].speaker_label !== segments[i - 1].speaker_label) speakerChanges++;
    }
    $('transcript-stats').textContent =
      `${segments.length} segment${segments.length !== 1 ? 's' : ''} \u00b7 ` +
      `${totalWords.toLocaleString()} word${totalWords !== 1 ? 's' : ''} \u00b7 ` +
      `${formatClock(totalDuration)} duration \u00b7 ` +
      `${speakerChanges} speaker change${speakerChanges !== 1 ? 's' : ''}`;

    // Export buttons
    document.querySelectorAll('[data-export]').forEach((btn) => {
      btn.onclick = () => window.open(`${API}/jobs/${job.id}/export?format=${btn.dataset.export}`, '_blank');
    });
    $('btn-delete-job').onclick = () => deleteJob(job.id);
    $('btn-save-speakers').onclick = () => saveSpeakers(job.id);
    $('btn-save-metadata').onclick = () => saveMetadata(job.id);
    const canAI = job.status === 'completed';
    $('btn-ai-summarize').disabled = !canAI;
    $('btn-ai-cleanup').disabled = !canAI;
    $('btn-ai-summarize').onclick = () => summarizeTranscript(job.id);
    $('btn-ai-cleanup').onclick = () => cleanupTranscript(job.id);
    initEditToolbar();
    updateEditToolbar();

    // In-transcript search
    $('library-search-input').oninput = (e) => {
      inTranscriptSearchTerm = e.target.value.trim().toLowerCase();
      highlightInTranscript(inTranscriptSearchTerm);
    };
    $('library-search-input').value = '';
    $('search-hit-info').textContent = '';
  }

  async function updateSegmentTags(idx, tags) {
    const seg = currentSegments[idx];
    if (!seg) return;
    try {
      await api(`/jobs/${currentJobId}/segments/${seg.id}/tags`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      });
      seg.tags = tags;
      // Re-render tags
      const container = document.querySelector(`[data-tags-for="${idx}"]`);
      if (container) {
        container.innerHTML = tags.map((t) =>
          `<span class="seg-tag" data-tag="${escapeHtml(t)}">${escapeHtml(t)}</span>`
        ).join('') +
        `<span class="seg-tag-add" data-add-tag="${idx}">+ tag</span>`;
        // Re-bind
        container.querySelectorAll('[data-add-tag]').forEach((btn) => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const i = parseInt(btn.dataset.addTag, 10);
            const s = currentSegments[i];
            if (!s) return;
            const t = prompt('Add tag:');
            if (t && t.trim()) {
              const ts = s.tags || [];
              if (!ts.includes(t.trim())) { ts.push(t.trim()); updateSegmentTags(i, ts); }
            }
          });
        });
        container.querySelectorAll('[data-tag]').forEach((el) => {
          el.addEventListener('click', (e) => {
            e.stopPropagation();
            const segEl = el.closest('[data-seg-index]');
            if (!segEl) return;
            const i = parseInt(segEl.dataset.segIndex, 10);
            const s = currentSegments[i];
            if (!s) return;
            updateSegmentTags(i, (s.tags || []).filter((t) => t !== el.dataset.tag));
          });
        });
      }
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  function highlightInTranscript(term) {
    const textEls = document.querySelectorAll('[data-seg-text]');
    let hitCount = 0;
    textEls.forEach((el) => {
      const idx = parseInt(el.dataset.segText, 10);
      const seg = currentSegments[idx];
      if (!seg) return;
      const edit = segmentEdits.get(idx);
      const baseText = edit ? edit.editedText : seg.text;
      if (!term) {
        el.textContent = baseText;
        el.closest('.segment')?.classList.remove('search-active');
        return;
      }
      const lower = baseText.toLowerCase();
      const pos = lower.indexOf(term);
      if (pos >= 0) {
        hitCount++;
        const before = escapeHtml(baseText.slice(0, pos));
        const match = escapeHtml(baseText.slice(pos, pos + term.length));
        const after = escapeHtml(baseText.slice(pos + term.length));
        el.innerHTML = `${before}<span class="search-hl">${match}</span>${after}`;
        el.closest('.segment')?.classList.add('search-active');
      } else {
        el.textContent = baseText;
        el.closest('.segment')?.classList.remove('search-active');
      }
    });
    $('search-hit-info').textContent = term ? `${hitCount} hit(s)` : '';
  }

  const SWATCHES = ['#2f7d45', '#bf9b30', '#9a3324', '#3d5a80', '#7a4fa0', '#c06d2f'];
  function swatchColor(i) { return SWATCHES[i % SWATCHES.length]; }

  // --------------------------------------------------------- metadata save

  async function saveMetadata(jobId) {
    const data = {
      interviewee: $('meta-interviewee').value.trim() || null,
      interviewer: $('meta-interviewer').value.trim() || null,
      interview_date: $('meta-date').value.trim() || null,
      location: $('meta-location').value.trim() || null,
      project_name: $('meta-project').value.trim() || null,
      collection_id: $('meta-collection').value.trim() || null,
      access_restrictions: $('meta-access').value.trim() || null,
      custom_vocabulary: $('meta-vocabulary').value.trim() || null,
    };
    try {
      await api(`/jobs/${jobId}/metadata`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      window.ArtificeToast.success('Metadata saved', {duration: 3600});
      if ($('panel-library').classList.contains('active')) loadLibrary();
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  // --------------------------------------------------------- editing / diff

  function initEditToolbar() {
    $('btn-save-edits').onclick = saveEdits;
    $('btn-revert-edits').onclick = revertEdits;
    $('btn-diff-toggle').onclick = toggleDiff;
    $('btn-history-toggle').onclick = toggleHistory;
    $('btn-history-close').onclick = () => $('history-panel')?.classList.add('hidden');
  }

  function updateEditToolbar() {
    const hasEdits = segmentEdits.size > 0;
    $('btn-save-edits').disabled = !hasEdits;
    $('btn-revert-edits').disabled = !hasEdits;
  }

  async function saveEdits() {
    if (!currentJobId || segmentEdits.size === 0) return;
    const updates = [];
    for (const [idx, edit] of segmentEdits) {
      const segId = currentSegments[idx]?.id;
      if (segId) {
        updates.push({ segment_id: segId, text: edit.editedText });
      }
    }
    if (updates.length === 0) return;
    try {
      await api(`/jobs/${currentJobId}/segments`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ updates }),
      });
      window.ArtificeToast.success(`Saved ${updates.length} edit(s)`, {duration: 3600});
      for (const [idx, edit] of segmentEdits) {
        currentSegments[idx].text = edit.editedText;
      }
      segmentEdits.clear();
      updateEditToolbar();
      refreshSegmentsDisplay();
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  function revertEdits() {
    if (segmentEdits.size === 0) return;
    segmentEdits.clear();
    updateEditToolbar();
    refreshSegmentsDisplay();
    window.ArtificeToast.success('Reverted all edits', {duration: 3600});
  }

  function refreshSegmentsDisplay() {
    if (!currentJobId || currentSegments.length === 0) return;
    const segEl = $('segments');
    segEl.querySelectorAll('[data-seg-text]').forEach((el) => {
      const idx = parseInt(el.dataset.segText, 10);
      const edit = segmentEdits.get(idx);
      el.textContent = edit ? edit.editedText : currentSegments[idx].text;
      const seg = el.closest('.segment');
      if (seg) {
        seg.classList.toggle('edited', !!edit);
        const badge = seg.querySelector('.seg-edited-badge');
        if (edit && !badge) {
          const b = document.createElement('span');
          b.className = 'seg-edited-badge';
          b.title = 'Edited';
          b.textContent = 'edited';
          seg.appendChild(b);
        } else if (!edit && badge) {
          badge.remove();
        }
      }
    });
  }

  // segment split/merge

  async function splitSegment() {
    const activeSeg = document.querySelector('.segment.active');
    if (!activeSeg) return;
    const sel = window.getSelection();
    if (!sel.rangeCount) return;
    const textEl = activeSeg.querySelector('[data-seg-text]');
    if (!textEl || !sel.containsNode(textEl, true)) return;
    const range = sel.getRangeAt(0);
    const preCaret = range.cloneRange();
    preCaret.selectNodeContents(textEl);
    preCaret.setEnd(range.startContainer, range.startOffset);
    const pos = preCaret.toString().length;
    if (pos <= 0 || pos >= textEl.textContent.length) {
      window.ArtificeToast.warning('Move cursor inside the text to split', {duration: 3600});
      return;
    }
    const segId = activeSeg.dataset.segId;
    try {
      const result = await api(`/jobs/${currentJobId}/segments/${segId}/split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ split_position: pos }),
      });
      // Reload transcript data
      const [job, transcript, speakers] = await Promise.all([
        api(`/jobs/${currentJobId}`),
        api(`/jobs/${currentJobId}/transcript`),
        api(`/jobs/${currentJobId}/speakers`),
      ]);
      currentSegments = transcript.segments;
      segmentEdits.clear();
      diffMode = false;
      renderTranscript(job, transcript, speakers);
      fillMetadataForm(job);
      loadWaveform();
      window.ArtificeToast.success('Segment split', {duration: 3600});
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  async function mergeSegment() {
    const activeSeg = document.querySelector('.segment.active');
    if (!activeSeg) return;
    const segId = activeSeg.dataset.segId;
    const idx = parseInt(activeSeg.dataset.segIndex, 10);
    if (idx >= currentSegments.length - 1) {
      window.ArtificeToast.warning('No next segment to merge with', {duration: 3600});
      return;
    }
    try {
      const result = await api(`/jobs/${currentJobId}/segments/${segId}/merge`, {
        method: 'POST',
      });
      // Reload transcript data
      const [job, transcript, speakers] = await Promise.all([
        api(`/jobs/${currentJobId}`),
        api(`/jobs/${currentJobId}/transcript`),
        api(`/jobs/${currentJobId}/speakers`),
      ]);
      currentSegments = transcript.segments;
      segmentEdits.clear();
      diffMode = false;
      renderTranscript(job, transcript, speakers);
      fillMetadataForm(job);
      loadWaveform();
      window.ArtificeToast.success('Segments merged', {duration: 3600});
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  function toggleDiff() {
    diffMode = !diffMode;
    const btn = $('btn-diff-toggle');
    btn.classList.toggle('active', diffMode);
    const segEl = $('segments');
    const textEls = segEl.querySelectorAll('[data-seg-text]');
    textEls.forEach((el) => {
      const idx = parseInt(el.dataset.segText, 10);
      const edit = segmentEdits.get(idx);
      const seg = el.closest('.segment');
      if (diffMode && edit) {
        el.innerHTML = computeDiffHtml(edit.originalText, edit.editedText);
        if (seg) seg.classList.add('diff-active');
      } else {
        el.textContent = edit ? edit.editedText : (currentSegments[idx]?.text || '');
        if (seg) seg.classList.remove('diff-active');
      }
    });
  }

  async function toggleHistory() {
    const panel = $('history-panel');
    if (!panel.classList.contains('hidden')) {
      panel.classList.add('hidden');
      return;
    }
    // Check if a segment is selected
    const activeSeg = document.querySelector('.segment.active');
    if (!activeSeg) {
      window.ArtificeToast.warning('Click a segment first, then open History', {duration: 3600});
      return;
    }
    const segId = activeSeg.dataset.segId;
    if (!segId) return;
    try {
      const data = await api(`/jobs/${currentJobId}/segments/${segId}/history`);
      const list = $('history-list');
      if (data.versions.length === 0) {
        list.innerHTML = '<p class="dim">No edit history for this segment.</p>';
      } else {
        list.innerHTML = data.versions.map((v) => `
          <div class="history-version">
            <div class="hist-time">${formatDate(v.edited_at)}</div>
            <div class="hist-before">${escapeHtml(v.text_before)}</div>
            <div class="hist-after">${escapeHtml(v.text_after)}</div>
          </div>
        `).join('');
      }
      panel.classList.remove('hidden');
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  function computeDiffHtml(original, edited) {
    const origWords = original.split(/(\s+)/);
    const editWords = edited.split(/(\s+)/);
    const lcs = buildLCS(origWords, editWords);
    let oi = 0, ei = 0, li = 0;
    let html = '';
    while (oi < origWords.length || ei < editWords.length) {
      if (li < lcs.length && oi < origWords.length && origWords[oi] === lcs[li]
          && ei < editWords.length && editWords[ei] === lcs[li]) {
        html += escapeHtml(lcs[li]);
        oi++; ei++; li++;
      } else if (oi < origWords.length && (li >= lcs.length || origWords[oi] !== lcs[li])) {
        html += `<span class="diff-del">${escapeHtml(origWords[oi])}</span>`;
        oi++;
      } else if (ei < editWords.length && (li >= lcs.length || editWords[ei] !== lcs[li])) {
        html += `<span class="diff-add">${escapeHtml(editWords[ei])}</span>`;
        ei++;
      } else { break; }
    }
    return html;
  }

  function buildLCS(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const result = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
      if (a[i - 1] === b[j - 1]) { result.unshift(a[i - 1]); i--; j--; }
      else if (dp[i - 1][j] > dp[i][j - 1]) i--;
      else j--;
    }
    return result;
  }

  async function saveSpeakers(jobId) {
    const inputs = document.querySelectorAll('[data-speaker-label]');
    const payload = {
      speakers: Array.from(inputs).map((input) => ({
        speaker_label: input.dataset.speakerLabel,
        custom_name: input.value.trim() || input.dataset.speakerLabel,
      })),
    };
    try {
      await api(`/jobs/${jobId}/speakers`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      window.ArtificeToast.success('Speaker names saved', {duration: 3600});
      selectJob(jobId);
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  async function deleteJob(jobId) {
    if (!confirm('Delete this job and its audio file? This cannot be undone.')) return;
    try {
      await api(`/jobs/${jobId}`, { method: 'DELETE' });
      activeJobs.delete(jobId);
      window.ArtificeToast.success('Job deleted', {duration: 3600});
      if (currentJobId === jobId) {
        currentJobId = null;
        $('transcript-card').classList.add('hidden');
        $('transcript-empty').classList.remove('hidden');
      }
      renderActiveJobs();
      loadLibrary();
    } catch (err) {
      window.ArtificeToast.error(err.message);
    }
  }

  // --------------------------------------------------------- audio highlight

  function initAudioHighlight() {
    const audio = $('audio-player');
    audio.addEventListener('timeupdate', () => {
      if (!currentSegments.length) return;
      const t = audio.currentTime;
      let activeIndex = -1;
      for (let i = 0; i < currentSegments.length; i++) {
        if (t >= currentSegments[i].start_time && t < currentSegments[i].end_time) { activeIndex = i; break; }
      }
      document.querySelectorAll('.segment').forEach((el) => el.classList.remove('active'));
      if (activeIndex >= 0) {
        const el = document.querySelector(`[data-seg-index="${activeIndex}"]`);
        if (el) el.classList.add('active');
      }
      updateWaveformCursor(t);
    });
  }

  // ------------------------------------------------------------ waveform

  let waveformPeaks = null;
  let waveformDuration = 0;

  async function loadWaveform() {
    const canvas = $('waveform-canvas');
    if (!canvas) return;
    if (!currentJobId) { canvas.style.display = 'none'; return; }
    canvas.style.display = 'block';

    // Get audio duration from the player
    const audio = $('audio-player');
    if (!audio.src) return;

    try {
      const resp = await fetch(audio.src);
      const blob = await resp.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const ctx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 44100, 44100);
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      waveformDuration = audioBuffer.duration;
      const raw = audioBuffer.getChannelData(0);
      // Downsample to canvas width
      const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
      canvas.width = w;
      canvas.height = 80;
      const step = Math.floor(raw.length / w);
      waveformPeaks = [];
      for (let i = 0; i < w; i++) {
        let max = 0;
        for (let j = 0; j < step; j++) {
          const val = Math.abs(raw[i * step + j] || 0);
          if (val > max) max = val;
        }
        waveformPeaks.push(max);
      }
      drawWaveform(0);

      // Click to seek
      canvas.onclick = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = x / canvas.width;
        audio.currentTime = pct * waveformDuration;
      };
    } catch (_) {
      canvas.style.display = 'none';
    }
  }

  function drawWaveform(currentTime) {
    const canvas = $('waveform-canvas');
    if (!canvas || !waveformPeaks) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
      || matchMedia('(prefers-color-scheme: dark)').matches;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = isDark ? '#1f1b16' : '#efebdf';
    ctx.fillRect(0, 0, w, h);

    // Draw segments as region highlights
    if (currentSegments.length > 0 && waveformDuration > 0) {
      currentSegments.forEach((seg) => {
        const sx = (seg.start_time / waveformDuration) * w;
        const ex = (seg.end_time / waveformDuration) * w;
        ctx.fillStyle = 'rgba(47, 125, 69, 0.08)';
        ctx.fillRect(sx, 0, ex - sx, h);
      });
    }

    // Draw waveform
    const center = h / 2;
    ctx.strokeStyle = isDark ? '#4aa066' : '#2f7d45';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < waveformPeaks.length; i++) {
      const x = (i / waveformPeaks.length) * w;
      const amp = waveformPeaks[i] * (center - 4);
      if (i === 0) ctx.moveTo(x, center - amp);
      ctx.lineTo(x, center - amp);
    }
    ctx.stroke();

    // Mirror bottom half
    ctx.beginPath();
    for (let i = 0; i < waveformPeaks.length; i++) {
      const x = (i / waveformPeaks.length) * w;
      const amp = waveformPeaks[i] * (center - 4);
      if (i === 0) ctx.moveTo(x, center + amp);
      ctx.lineTo(x, center + amp);
    }
    ctx.stroke();

    // Playback cursor
    if (currentTime > 0 && waveformDuration > 0) {
      const cx = (currentTime / waveformDuration) * w;
      ctx.strokeStyle = '#bf9b30';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, h);
      ctx.stroke();
    }
  }

  let waveformAnimFrame = null;

  function updateWaveformCursor(t) {
    if (waveformAnimFrame) cancelAnimationFrame(waveformAnimFrame);
    waveformAnimFrame = requestAnimationFrame(() => drawWaveform(t));
  }

  // ------------------------------------------------------------ global search

  function initGlobalSearch() {
    const input = $('global-search-input');
    let debounceTimer = null;
    input.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        const q = input.value.trim();
        if (q.length >= 2) {
          performSearch(q);
        }
      }, 350);
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = input.value.trim();
        if (q.length >= 2) performSearch(q);
      }
    });
  }

  async function performSearch(q) {
    try {
      const data = await api(`/search?q=${encodeURIComponent(q)}`);
      showSearchResults(data, q);
    } catch (err) {
      window.ArtificeToast.error(`Search failed: ${err.message}`);
    }
  }

  function showSearchResults(data, query) {
    const modal = $('search-modal');
    const list = $('search-results-list');
    const count = $('search-result-count');
    count.textContent = `${data.total} result(s)`;

    if (data.results.length === 0) {
      list.innerHTML = '<p class="dim">No results found.</p>';
    } else {
      list.innerHTML = data.results.map((r) => {
        const meta = [r.interviewee, r.interview_date, r.project_name].filter(Boolean).join(' — ');
        const highlighted = r.text.replace(
          new RegExp(escapeRegex(query), 'gi'),
          (m) => `<span class="hl">${escapeHtml(m)}</span>`
        );
        return `
        <div class="search-result-item" data-search-job="${escapeHtml(r.job_id)}" data-search-seg="${escapeHtml(r.segment_id)}" data-search-time="${r.start_time}">
          <div class="search-result-meta">
            <span><strong>${escapeHtml(r.filename)}</strong></span>
            <span>${escapeHtml(r.speaker_label)}</span>
            <span>${formatClock(r.start_time)}</span>
            ${meta ? `<span>${escapeHtml(meta)}</span>` : ''}
          </div>
          <div class="search-result-text">${highlighted}</div>
        </div>`;
      }).join('');

      list.querySelectorAll('.search-result-item').forEach((el) => {
        el.addEventListener('click', () => {
          const jobId = el.dataset.searchJob;
          const segId = el.dataset.searchSeg;
          const time = parseFloat(el.dataset.searchTime);
          closeSearchModal();
          // Switch to library tab and select job
          document.querySelector('.tab[data-tab="library"]').click();
          selectJob(jobId);
          // Seek to time after render
          setTimeout(() => {
            const audio = $('audio-player');
            audio.currentTime = time;
            audio.play().catch(() => {});
            // Highlight segment
            document.querySelectorAll('[data-seg-id]').forEach((s) => {
              if (s.dataset.segId === segId) {
                s.classList.add('active');
                s.scrollIntoView({ behavior: 'smooth', block: 'center' });
              }
            });
          }, 300);
        });
      });
    }

    modal.classList.remove('hidden');
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function closeSearchModal() {
    $('search-modal')?.classList.add('hidden');
  }

  // ------------------------------------------------------------ config load

  async function loadConfig() {
    try {
      const cfg = await api('/config');
      const badge = $('opt-model-size');
      badge.textContent = `${cfg.whisper_model} (${cfg.device})`;
    } catch (_) { /* config endpoint unavailable */ }
  }

  // ------------------------------------------------------------ health check

  function setDot(id, status) {
    const el = $(id);
    if (el) { el.className = 'health-dot ' + status; }
  }

  function setStatusDot(id, status) {
    const el = $(id);
    if (el) { el.className = 'health-status ' + status; }
  }

  function showHealthError(msg) {
    const box = $('health-error-box');
    $('health-error-text').textContent = msg;
    box.classList.remove('hidden');
  }

  function hideHealthError() {
    $('health-error-box').classList.add('hidden');
  }

  function mapModelState(state) {
    if (state === 'loaded') return 'ok';
    if (state === 'failed') return 'error';
    return 'warn';
  }

  async function loadHealth() {
    try {
      const h = await api('/health/detailed');
      const e = h.engine;

      setDot('health-db', h.database.status === 'ok' ? 'ok' : 'error');
      setDot('health-whisper', mapModelState(e.whisper_model.state));
      setDot('health-diarize', mapModelState(e.diarization_model.state));
      setDot('health-gpu', e.gpu ? 'ok' : 'warn');

      const whisperLabel = e.whisper_model.state === 'loaded'
        ? `${e.whisper_model.name} (loaded)`
        : e.whisper_model.state === 'failed'
          ? `${e.whisper_model.name} (failed)`
          : `${e.whisper_model.name} (not loaded)`;
      $('health-whisper-name').textContent = whisperLabel;
      setStatusDot('health-whisper-status', mapModelState(e.whisper_model.state));

      const diarizeLabel = e.diarization_model.state === 'loaded'
        ? 'Loaded' : e.diarization_model.state === 'failed' ? 'Failed' : 'Not loaded';
      $('health-diarize-status').textContent = diarizeLabel;
      setStatusDot('health-diarize-dot', mapModelState(e.diarization_model.state));

      const langs = e.alignment_models.loaded_languages;
      const alignLabel = e.alignment_models.state === 'loaded'
        ? `${langs.join(', ')} (loaded)`
        : e.alignment_models.state === 'failed' ? 'Failed' : 'Not loaded';
      $('health-align-langs').textContent = alignLabel;
      setStatusDot('health-align-dot', mapModelState(e.alignment_models.state));

      $('health-device').textContent = e.device.toUpperCase();
      setStatusDot('health-device-dot', 'ok');

      if (e.gpu) {
        $('health-gpu-name').textContent = e.gpu.name;
        $('health-vram').textContent = `${e.gpu.vram_used_mb} / ${e.gpu.vram_total_mb} MB`;
        setStatusDot('health-gpu-dot', 'ok');
        const vramPct = e.gpu.vram_used_mb / e.gpu.vram_total_mb;
        setStatusDot('health-vram-dot', vramPct > 0.9 ? 'error' : vramPct > 0.7 ? 'warn' : 'ok');
      } else {
        $('health-gpu-name').textContent = 'N/A (CPU mode)';
        $('health-vram').textContent = 'N/A';
        setStatusDot('health-gpu-dot', 'warn');
        setStatusDot('health-vram-dot', 'warn');
      }

      $('health-hf-token').textContent = e.hf_token_configured ? 'Configured' : 'Missing';
      setStatusDot('health-hf-dot', e.hf_token_configured ? 'ok' : 'error');

      $('health-db-status').textContent = h.database.status === 'ok' ? 'Connected' : 'Error';
      setStatusDot('health-db-dot', h.database.status === 'ok' ? 'ok' : 'error');

      if (e.last_error) { showHealthError(e.last_error); } else { hideHealthError(); }
    } catch (_) {
      ['health-db', 'health-whisper', 'health-diarize', 'health-gpu'].forEach((id) => setDot(id, 'error'));
    }
  }

  async function preloadModels() {
    const btn = $('btn-load-models');
    btn.classList.add('loading');
    btn.textContent = 'Loading…';
    hideHealthError();
    try {
      const result = await api('/health/preload', { method: 'POST' });
      if (result.ok) { window.ArtificeToast.success('Models loaded successfully', {duration: 3600}); }
      else { window.ArtificeToast.error(`Model load failed: ${result.error}`); showHealthError(result.error); }
      window.ArtificeToast.error(`Preload failed: ${err.message}`);
      showHealthError(err.message);
    } finally {
      btn.classList.remove('loading');
      btn.textContent = 'Load Models';
      loadHealth();
    }
  }

  function initHealthPanel() {
    $('btn-health-details').addEventListener('click', () => {
      const panel = $('health-panel');
      const wasHidden = panel.classList.contains('hidden');
      panel.classList.toggle('hidden');
      if (wasHidden) loadHealth();
    });
    $('btn-health-close').addEventListener('click', () => { $('health-panel').classList.add('hidden'); });
    $('btn-load-models').addEventListener('click', preloadModels);
  }

  // ------------------------------------------------ persistent dictionary

  function saveDictionary() {
    const words = $('dict-textarea').value.trim() || null;
    const status = $('dict-save-status');
    status.textContent = 'Saving...';
    try {
      api('/dictionary', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words }),
      });
      status.textContent = 'Saved';
      setTimeout(() => { status.textContent = ''; }, 2000);
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    }
  }

  async function loadDictionary() {
    try {
      const data = await api('/dictionary');
      $('dict-textarea').value = (data && data.words) || '';
    } catch (_) { /* first load - table may be empty */ }
  }

  // ------------------------------------------------- speaker enrollment

  async function loadKnownSpeakers() {
    const container = $('known-speakers-list');
    try {
      const data = await api('/speakers/known');
      if (!data.speakers || data.speakers.length === 0) {
        container.innerHTML = '<p class="panel-empty-title">No enrolled speakers.</p><p class="panel-empty-desc">Use the form below to enroll a voice.</p>';
        return;
      }
      container.innerHTML = data.speakers.map((s) => `
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0;border-bottom:1px solid var(--rule);">
          <span style="flex:1;font-weight:600;">${escapeHtml(s.name)}</span>
          <span class="dim">${s.dimension}d (${s.model_name})</span>
          <span class="dim">${formatDate(s.created_at)}</span>
          <button class="btn tiny danger" data-delete-known="${escapeHtml(s.id)}">&times;</button>
        </div>
      `).join('');
      container.querySelectorAll('[data-delete-known]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          if (!confirm('Remove this speaker enrollment?')) return;
          try {
            await api(`/speakers/known/${btn.dataset.deleteKnown}`, { method: 'DELETE' });
            loadKnownSpeakers();
          } catch (err) { window.ArtificeToast.error(err.message); }
        });
      });
    } catch (_) { /* table may be empty */ }
  }

  async function enrollSpeaker() {
    const name = $('enroll-name').value.trim();
    const fileInput = $('enroll-file');
    const file = fileInput.files[0];
    const status = $('enroll-status');
    if (!name || !file) { status.textContent = 'Provide a name and audio file.'; return; }
    status.textContent = 'Enrolling...';
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', file);
    try {
      const resp = await fetch(`${API}/speakers/enroll`, { method: 'POST', body: formData });
      if (!resp.ok) { const b = await resp.json(); throw new Error(b.detail || resp.statusText); }
      const result = await resp.json();
      status.textContent = `Enrolled "${result.name}"`;
      $('enroll-name').value = '';
      fileInput.value = '';
      loadKnownSpeakers();
    } catch (err) { status.textContent = `Error: ${err.message}`; }
  }

  async function enrollFromJob() {
    if (!currentJobId) { window.ArtificeToast.warning('No job selected.', {duration: 3600}); return; }
    const inputs = document.querySelectorAll('[data-speaker-label]');
    const selected = Array.from(inputs).find((inp) => inp.dataset.enrolling) || inputs[0];
    if (!selected) { window.ArtificeToast.warning('No speakers in this job.', {duration: 3600}); return; }
    const speakerLabel = selected.dataset.speakerLabel;
    const name = prompt(`Enter a name for speaker "${speakerLabel}":`);
    if (!name || !name.trim()) return;
    try {
      await api('/speakers/enroll-from-job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: currentJobId, speaker_label: speakerLabel, name: name.trim() }),
      });
      window.ArtificeToast.success(`Enrolled "${name.trim()}"`, {duration: 3600});
      loadKnownSpeakers();
    } catch (err) { window.ArtificeToast.error(err.message); }
  }

  async function matchSpeakers() {
    if (!currentJobId) { window.ArtificeToast.warning('No job selected.', {duration: 3600}); return; }
    const status = $('match-status');
    status.textContent = 'Matching...';
    try {
      const result = await api(`/jobs/${currentJobId}/match-speakers`, { method: 'POST' });
      const matched = result.matches.filter((m) => m.matched_name);
      if (matched.length > 0) {
        window.ArtificeToast.success(`Matched ${matched.length} speaker(s)`, {duration: 3600});
      } else {
        window.ArtificeToast.success('No matches found', {duration: 3600});
      }
      status.textContent = '';
      // Reload transcript to reflect new names
      selectJob(currentJobId);
    } catch (err) { status.textContent = `Error: ${err.message}`; }
  }

  // ------------------------------------------------------------------ settings / models

async function loadModelConfig() {
  try {
    const cfg = await api('/config');
    if (cfg) {
      const modelSize = $('setting-model-size');
      if (modelSize) modelSize.value = cfg.whisper_model || 'base';
      const device = $('setting-device');
      if (device) device.value = cfg.device || 'auto';
      const hfToken = $('setting-hf-token');
      if (hfToken) hfToken.value = cfg.hf_token || '';
      const diarizationProvider = $('setting-diarization-provider');
      if (diarizationProvider) diarizationProvider.value = cfg.diarization_provider || 'pyannote';
      const diarizationModel = $('setting-diarization-model');
      // Empty means "let WhisperX choose"; do not substitute a concrete model
      // name here, or an unset value round-trips back to the server as a hard
      // pin on whatever this fallback happens to say.
      if (diarizationModel) diarizationModel.value = cfg.diarization_model || '';
      const alignmentCache = $('setting-alignment-cache');
      if (alignmentCache) alignmentCache.checked = cfg.enable_alignment_model_cache !== false;
      const asrBackend = $('setting-asr-backend');
      if (asrBackend) asrBackend.value = cfg.asr_backend || 'whisperx';
      const initialPrompt = $('setting-initial-prompt');
      if (initialPrompt) initialPrompt.value = cfg.whisper_initial_prompt || '';
      updateAsrBackendFieldStates();
    }
  } catch (err) {
    console.warn('Failed to load model config:', err);
  }
}

function updateAsrBackendFieldStates() {
  const backend = $('setting-asr-backend')?.value || 'whisperx';
  const isParakeet = backend === 'parakeet';

  const modelSize = $('setting-model-size');
  if (modelSize) modelSize.disabled = isParakeet;
  const noteModel = $('note-model-parakeet');
  if (noteModel) noteModel.style.display = isParakeet ? 'block' : 'none';

  const device = $('setting-device');
  if (device) device.disabled = isParakeet;
  const noteDevice = $('note-device-parakeet');
  if (noteDevice) noteDevice.style.display = isParakeet ? 'block' : 'none';

  const initialPrompt = $('setting-initial-prompt');
  if (initialPrompt) initialPrompt.disabled = isParakeet;
  const notePrompt = $('note-initial-prompt-parakeet');
  if (notePrompt) notePrompt.style.display = isParakeet ? 'block' : 'none';
}

async function saveModelConfig() {
  const body = {
    whisper_model: $('setting-model-size')?.value || 'base',
    device: $('setting-device')?.value || 'auto',
    hf_token: $('setting-hf-token')?.value || '',
    diarization_provider: $('setting-diarization-provider')?.value || 'pyannote',
    diarization_model: $('setting-diarization-model')?.value?.trim() || '',
    enable_alignment_model_cache: $('setting-alignment-cache')?.checked !== false,
    asr_backend: $('setting-asr-backend')?.value || 'whisperx',
    whisper_initial_prompt: $('setting-initial-prompt')?.value?.trim() || '',
  };

  const status = $('model-applied-status');
  const text = $('model-applied-text');

  try {
    await api('/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (status) status.style.display = 'block';
    if (text) text.textContent = 'Model configuration applied successfully.';
    window.ArtificeToast.success('Model settings saved', {duration: 3600});
  } catch (err) {
    if (status) status.style.display = 'block';
    if (text) text.textContent = `Failed to apply: ${err.message}`;
    window.ArtificeToast.error(`Failed to save model settings: ${err.message}`);
  }
}

async function saveInferenceConfig() {
  const base_url = $('setting-base-url').value.trim();
  const api_key = $('setting-api-key').value.trim() || 'not-needed';
  const model_name = $('setting-model-select').value;
  const vision_enabled = $('setting-vision').checked;

  const body = { base_url, api_key, model_name, vision_enabled };
  try {
    await api('/inference/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    // Persist only the non-secret fields needed to restore UI state.
    // The api_key is already stored server-side via POST /inference/config;
    // keeping it in localStorage exposes the plaintext key to any JS running
    // at localhost (a documented attack class against loopback servers).
    localStorage.setItem('pt-inference-config', JSON.stringify({
      base_url, model_name, vision_enabled,
    }));
    window.ArtificeToast.success('Inference settings saved successfully', {duration: 3600});
  } catch (err) {
    window.ArtificeToast.error(`Failed to save settings: ${err.message}`);
  }
}

function loadInferenceConfig() {
  try {
    const raw = localStorage.getItem('pt-inference-config');
    if (!raw) return;
    const cfg = JSON.parse(raw);
    const baseUrl = $('setting-base-url');
    if (baseUrl) baseUrl.value = cfg.base_url || '';
    // api_key is NOT restored from localStorage — it is only stored
    // server-side (POST /inference/config).  If the UI needs to show
    // whether a key is configured, that should come from the server.
    const modelSelect = $('setting-model-select');
    if (modelSelect && cfg.model_name) modelSelect.value = cfg.model_name;
    const vision = $('setting-vision');
    if (vision) vision.checked = !!cfg.vision_enabled;
  } catch (err) {
    console.warn('Failed to load cached inference config:', err);
  }
}

async function fetchInferenceModels() {
  const base_url = $('setting-base-url').value.trim();
  const api_key = $('setting-api-key').value.trim() || 'not-needed';
  const select = $('setting-model-select');
  select.innerHTML = '<option value="">Loading models...</option>';
  try {
    const res = await api('/inference/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url, api_key }),
    });
    select.innerHTML = '';
    if (res.models && res.models.length > 0) {
      res.models.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        select.appendChild(opt);
      });
      window.ArtificeToast.success(`Loaded ${res.models.length} model(s)`, {duration: 3600});
    } else {
      select.innerHTML = '<option value="">No models found</option>';
      window.ArtificeToast.warning('No models returned by server', {duration: 3600});
    }
  } catch (err) {
    select.innerHTML = '<option value="">Failed to fetch models</option>';
    window.ArtificeToast.error(`Error fetching models: ${err.message}`);
  }
}

async function testInferenceConnection() {
  const base_url = $('setting-base-url').value.trim();
  const api_key = $('setting-api-key').value.trim() || 'not-needed';
  const dot = $('connection-status-dot');
  const statusText = $('connection-status-text');
  dot.style.background = 'var(--rule)';
  statusText.textContent = 'Testing connection...';
  try {
    const res = await api('/inference/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url, api_key }),
    });
    if (res.success) {
      dot.style.background = '#22c55e';
      statusText.textContent = res.message;
      window.ArtificeToast.success('Connection successful!', {duration: 3600});
    } else {
      dot.style.background = '#ef4444';
      statusText.textContent = res.message;
      window.ArtificeToast.error(res.message);
    }
  } catch (err) {
    dot.style.background = '#ef4444';
    statusText.textContent = `Error: ${err.message}`;
    window.ArtificeToast.error(`Connection test failed: ${err.message}`);
  }
}

function initSettingsPanel() {
  $('preset-ollama').addEventListener('click', () => {
    $('setting-base-url').value = 'http://localhost:11434/v1';
  });
  $('preset-lmstudio').addEventListener('click', () => {
    $('setting-base-url').value = 'http://localhost:1234/v1';
  });
  $('preset-vllm').addEventListener('click', () => {
    $('setting-base-url').value = 'http://localhost:8080/v1';
  });
  $('preset-openai').addEventListener('click', () => {
    $('setting-base-url').value = 'https://api.openai.com/v1';
  });

  $('btn-fetch-models').addEventListener('click', fetchInferenceModels);
  $('btn-test-connection').addEventListener('click', testInferenceConnection);
  $('btn-save-settings').addEventListener('click', saveInferenceConfig);
  $('btn-apply-model').addEventListener('click', saveModelConfig);
  $('setting-asr-backend').addEventListener('change', updateAsrBackendFieldStates);

  loadModelConfig();
  loadInferenceConfig();
}

function openAIModal(title) {
  $('ai-modal').classList.remove('hidden');
  $('ai-modal-title').textContent = title;
  $('ai-modal-status').textContent = '';
  $('ai-modal-content').textContent = '';
}

function closeAIModal() {
  $('ai-modal').classList.add('hidden');
}

function setAIStatus(text, type) {
  const el = $('ai-status');
  if (el) el.textContent = text;
  const modalStatus = $('ai-modal-status');
  if (modalStatus) {
    modalStatus.textContent = text;
    modalStatus.style.color = type === 'error' ? 'var(--error)' : type === 'ok' ? 'var(--success)' : '';
  }
}

async function streamAIJob(jobId, endpoint) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`${API}/jobs/${jobId}/${endpoint}`);
    let fullText = '';
    es.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'chunk') {
          fullText += data.text;
          $('ai-modal-content').textContent = fullText;
        } else if (data.type === 'error') {
          setAIStatus(`Error: ${data.text}`, 'error');
          es.close();
          reject(new Error(data.text));
        } else if (data.type === 'done') {
          setAIStatus('Done', 'ok');
          es.close();
          resolve(fullText);
        }
      } catch (_) {
        // ignore parse errors
      }
    });
    es.onerror = () => {
      setAIStatus('Connection error', 'error');
      es.close();
      reject(new Error('SSE connection failed'));
    };
  });
}

async function summarizeTranscript(jobId) {
  if (!jobId) return;
  openAIModal('Transcript Summary');
  setAIStatus('Generating summary...', '');
  $('btn-ai-summarize').disabled = true;
  $('btn-ai-cleanup').disabled = true;
  try {
    const result = await streamAIJob(jobId, 'summarize');
    setAIStatus(`Summary generated (${result.length} chars)`, 'ok');
  } catch (err) {
    setAIStatus(`Failed: ${err.message}`, 'error');
  } finally {
    $('btn-ai-summarize').disabled = false;
    $('btn-ai-cleanup').disabled = false;
  }
}

async function cleanupTranscript(jobId) {
  if (!jobId) return;
  openAIModal('Transcript Cleanup');
  setAIStatus('Cleaning up transcript...', '');
  $('btn-ai-summarize').disabled = true;
  $('btn-ai-cleanup').disabled = true;
  try {
    const result = await streamAIJob(jobId, 'cleanup');
    setAIStatus(`Cleanup complete (${result.length} chars)`, 'ok');
  } catch (err) {
    setAIStatus(`Failed: ${err.message}`, 'error');
  } finally {
    $('btn-ai-summarize').disabled = false;
    $('btn-ai-cleanup').disabled = false;
  }
}

function copyAIResult() {
  const text = $('ai-modal-content').textContent;
  if (text && text !== 'Waiting for AI response...') {
    navigator.clipboard.writeText(text).then(() => window.ArtificeToast.success('Copied to clipboard', {duration: 3600}));
  }
}

function downloadAIResult() {
  const text = $('ai-modal-content').textContent;
  if (!text || text === 'Waiting for AI response...') return;
  const title = ($('ai-modal-title').textContent || 'ai-result').toLowerCase().replace(/\s+/g, '-');
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${title}-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  window.ArtificeToast.success('Downloaded', {duration: 3600});
}

// ─────────────────────────────────────────────────────────────────────────────
// ASR MODEL DOWNLOAD CONSENT DIALOG
//
// A consent surface, not a progress bar with a button. The four things it
// must get right, per the design brief:
//
//   1. Show the TRUE total (transitive size), not one model's size.
//      The API returns `total_size_human` which is already summed — use it.
//   2. Show where files land.  `cache_directory` is an absolute, resolved
//      path that differs per OS — display it verbatim.
//   3. Consent must be a decision, not a formality.  POST /consent first,
//      then POST /download.  Nothing begins until the user clicks Download.
//   4. Cancel is honest.  The backend emits `cancelling` (still running)
//      then `cancelled` (truly stopped).  Display "Stopping…" on cancelling
//      and do NOT say "Cancelled" until the event actually arrives.
// ─────────────────────────────────────────────────────────────────────────────

let _dlgModelKey = null;
let _dlgAllModels = [];
let _dlgCurrentInfo = null;
let _dlgSSE = null;
let _dlgStopping = false;
let _dlgTriggerEl = null;
// Screen reader: announce progress at 10-point intervals, not per byte.
let _dlgLastAnnouncedPct = -1;

function _esc(s) {
  return (s ?? '').toString()
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Open / close ─────────────────────────────────────────────────────────────

function openDownloadDialog(triggerEl) {
  _dlgTriggerEl = triggerEl || null;
  _dlgModelKey = null;
  _dlgAllModels = [];
  _dlgCurrentInfo = null;
  _dlgStopping = false;
  _dlgLastAnnouncedPct = -1;
  closeDownloadSSE();

  // Show overlay; seed with loading state while we fetch the model list
  $('download-modal-overlay').classList.remove('hidden');
  _setDlgState('loading');

  api('/models').then((data) => {
    _dlgAllModels = data && data.models ? data.models : [];
    if (_dlgAllModels.length === 0) {
      _setDlgError('No ASR models are registered. This may be a configuration error.');
      return;
    }
    // Auto-select the first registered model; the list order is the registry
    // order, so the primary model comes first.
    _selectModel(_dlgAllModels[0].key);
  }).catch((err) => {
    _setDlgError(`Could not reach the server: ${err.message}`);
  });

  // Focus the close button (a safe, always-present interactive element)
  $('dlg-close').focus();
  document.body.style.overflow = 'hidden';
}

function closeDownloadDialog() {
  closeDownloadSSE();
  $('download-modal-overlay').classList.add('hidden');
  document.body.style.overflow = '';
  // Return focus to whatever opened the dialog
  if (_dlgTriggerEl && typeof _dlgTriggerEl.focus === 'function') {
    _dlgTriggerEl.focus();
  }
  _dlgTriggerEl = null;
}

function closeDownloadSSE() {
  if (_dlgSSE) {
    _dlgSSE.close();
    _dlgSSE = null;
  }
}

// ── Model selection ─────────────────────────────────────────────────────────

function _selectModel(key) {
  _dlgModelKey = key;
  api(`/models/${key}`).then((info) => {
    _dlgCurrentInfo = info;
    _renderConsentView(info);
  }).catch((err) => {
    _setDlgError(`Could not load details for ${key}: ${err.message}`);
  });
}

// ── State machine ───────────────────────────────────────────────────────────

// States: loading | consent-view | token-view | downloading | stopping |
//         completed | error-view
function _setDlgState(state, extra) {
  const overlay     = $('download-modal-overlay');
  const consentView = $('dlg-consent-view');
  const progressSec = $('dlg-progress-section');
  const stateMsg    = $('dlg-state-message');
  const errorBox    = $('dlg-error-box');
  const btnCancel   = $('dlg-btn-cancel');
  const btnDownload = $('dlg-btn-download');

  // Reset visibility
  consentView.classList.add('hidden');
  progressSec.classList.add('hidden');
  stateMsg.classList.add('hidden');
  stateMsg.className = '';
  errorBox.classList.add('hidden');
  btnCancel.classList.remove('hidden');
  btnDownload.classList.remove('hidden');
  btnDownload.disabled = false;

  if (state === 'loading') {
    btnDownload.disabled = true;
    btnDownload.textContent = 'Loading…';
    btnCancel.classList.add('hidden');
    return;
  }

  if (state === 'consent-view') {
    consentView.classList.remove('hidden');
    btnDownload.textContent = 'Download';
    btnDownload.disabled = extra && extra.tokenRequired && !extra.hasToken;
    btnCancel.textContent = 'Close';
    btnCancel.disabled = false;
    return;
  }

  if (state === 'token-view') {
    consentView.classList.remove('hidden');
    $('dlg-token-section').classList.remove('hidden');
    btnDownload.textContent = 'Continue';
    btnDownload.disabled = false;
    btnCancel.textContent = 'Close';
    btnCancel.disabled = false;
    return;
  }

  if (state === 'downloading') {
    consentView.classList.add('hidden');
    progressSec.classList.remove('hidden');
    btnDownload.textContent = 'Stop';
    btnDownload.disabled = false;
    btnCancel.classList.add('hidden');
    return;
  }

  if (state === 'stopping') {
    consentView.classList.add('hidden');
    progressSec.classList.remove('hidden');
    btnDownload.textContent = 'Stop';
    btnDownload.disabled = true;   // cannot cancel again while unwinding
    btnCancel.classList.remove('hidden');
    btnCancel.classList.add('hidden');  // hide cancel while stopping
    stateMsg.classList.remove('hidden');
    stateMsg.className = 'stopping';
    stateMsg.textContent = 'Stopping — please wait…';
    return;
  }

  if (state === 'completed') {
    consentView.classList.add('hidden');
    progressSec.classList.remove('hidden');
    btnDownload.textContent = 'Done';
    btnDownload.disabled = false;
    btnCancel.classList.add('hidden');
    stateMsg.classList.remove('hidden');
    stateMsg.className = 'completed';
    stateMsg.textContent = 'All models downloaded successfully.';
    return;
  }

  if (state === 'error-view') {
    consentView.classList.add('hidden');
    progressSec.classList.remove('hidden');
    btnDownload.textContent = 'Retry';
    btnDownload.disabled = false;
    btnCancel.classList.remove('hidden');
    btnCancel.textContent = 'Close';
    btnCancel.disabled = false;
    stateMsg.classList.remove('hidden');
    stateMsg.className = 'error-state';
    stateMsg.textContent = extra && extra.message ? extra.message : 'An error occurred.';
    return;
  }
}

function _setDlgError(message) {
  $('dlg-error-box').classList.remove('hidden');
  $('dlg-error-text').textContent = message;
  $('dlg-btn-download').textContent = 'Retry';
  $('dlg-btn-download').disabled = false;
  $('dlg-btn-cancel').textContent = 'Close';
  $('dlg-btn-cancel').disabled = false;
}

// ── Consent view ─────────────────────────────────────────────────────────────

function _renderConsentView(info) {
  // Total size (large display) — this is the transitive total, not one model
  const totalEl = $('dlg-total-size');
  totalEl.textContent = info.total_size_human || '—';

  // Cache directory
  $('dlg-cache-path').textContent = info.cache_directory || '—';

  // Model breakdown list
  const listEl = $('dlg-model-list');
  listEl.innerHTML = '';
  (info.models || []).forEach((m, i) => {
    const item = document.createElement('div');
    item.className = 'dlg-model-item';
    item.innerHTML =
      '<span class="dlg-model-bar-label">' +
        `<span class="dlg-model-badge ${i === 0 ? 'main' : 'dep'}">${i === 0 ? 'Main' : 'Dep'}</span>` +
        ` <span class="dlg-model-name"><code>${_esc(m.hf_repo)}</code></span>` +
      '</span>' +
      `<span class="dlg-model-size">${m.size_human || '—'}</span>`;
    listEl.appendChild(item);
  });

  // Token section: show only when required AND no token has been saved yet.
  // The Settings panel is the authoritative token store; we check the
  // `requires_hf_token` on the selected model key.
  const tokenSection = $('dlg-token-section');
  if (info.requires_hf_token) {
    tokenSection.classList.remove('hidden');
    $('dlg-token-input').value = '';
    _setDlgState('token-view');
  } else {
    tokenSection.classList.add('hidden');
    _setDlgState('consent-view', { tokenRequired: false, hasToken: true });
  }

  // Download button: disabled until consent is given (on click, not on open)
  // Token-required case: disabled until token field has content
  const tokenInput = $('dlg-token-input');
  const updateDownloadEnabled = () => {
    if (info.requires_hf_token) {
      $('dlg-btn-download').disabled = tokenInput.value.trim().length === 0;
    }
  };
  tokenInput.oninput = updateDownloadEnabled;
  updateDownloadEnabled();
}

// ── Download flow ────────────────────────────────────────────────────────────

async function _doDownload(key) {
  // 1. Record consent
  await api(`/models/${key}/consent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ consent: true }),
  });

  // 2. If a token was typed in the dialog, save it via PATCH /config
  const tokenInput = $('dlg-token-input');
  const typedToken = tokenInput ? tokenInput.value.trim() : '';
  if (typedToken) {
    await api('/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_token: typedToken }),
    });
  }

  // 3. Start download
  await api(`/models/${key}/download`, { method: 'POST' });
}

function _startDownload(key) {
  _setDlgState('downloading');
  _resetProgressBars();

  _doDownload(key).then(() => {
    // Switch to SSE listening (download is running in background on server)
    _openProgressSSE(key);
  }).catch((err) => {
    // Show persistent error; do NOT switch to consent view (user must Retry)
    _setDlgState('error-view', { message: err.message });
    $('dlg-error-box').classList.remove('hidden');
    $('dlg-error-text').textContent = err.message;
    window.ArtificeToast.error(`Download failed: ${err.message}`);
  });
}

// ── SSE progress streaming ───────────────────────────────────────────────────

function _openProgressSSE(key) {
  closeDownloadSSE();
  _dlgSSE = new EventSource(`${API}/models/${key}/download/progress`);

  _dlgSSE.onerror = () => {
    // Network-level error or server stopped
    _dlgSSE.close();
    _dlgSSE = null;
    // If we were stopping, this is the normal end-of-stream after cancelled
    if (_dlgStopping) { return; }
    // Otherwise show error
    _setDlgState('error-view', { message: 'Lost connection to the download server.' });
  };

  _dlgSSE.addEventListener('message', (e) => {
    let event;
    try { event = JSON.parse(e.data); } catch (_) { return; }

    const t = event.type;

    // ── Progress ────────────────────────────────────────────────────────
    if (t === 'progress') {
      if (_dlgStopping) return;   // ignore ticks that arrive after cancelling

      const total    = event.total_bytes    || 1;
      const downloaded = event.downloaded_bytes || 0;
      const pct      = Math.min(Math.round((downloaded / total) * 100), 100);
      const modelIdx = event.model_idx      || 0;
      const modelTot = event.model_total    || 1;

      // Update overall bar
      $('dlg-overall-fill').style.width = pct + '%';
      $('dlg-progress-pct').textContent  = pct + '%';
      $('dlg-overall-bar').setAttribute('aria-valuenow', pct);

      // Update model bars
      _updateModelBar(modelIdx, modelTot, pct, event.hf_repo);

      // Update bytes counter
      const downloadedMB = (downloaded / 1_000_000).toFixed(1);
      const totalMB      = (total / 1_000_000).toFixed(1);
      $('dlg-bytes-row').textContent =
        `${downloadedMB} / ${totalMB} MB — Model ${modelIdx + 1} of ${modelTot}`;

      // Screen reader: announce at 10-point intervals
      const announcePct = Math.floor(pct / 10) * 10;
      if (announcePct !== _dlgLastAnnouncedPct && announcePct > 0) {
        _dlgLastAnnouncedPct = announcePct;
        _announceToScreenReader(`Download ${pct}% complete`);
      }
      return;
    }

    // ── Cancelling ─────────────────────────────────────────────────────
    if (t === 'cancelling') {
      _dlgStopping = true;
      _setDlgState('stopping');
      _announceToScreenReader('Download stopping. This may take a moment.');
      return;
    }

    // ── Cancelled ─────────────────────────────────────────────────────
    if (t === 'cancelled') {
      closeDownloadSSE();
      _dlgStopping = false;
      _setDlgState('error-view', { message: 'Download was cancelled. No files were saved.' });
      $('dlg-state-message').classList.remove('hidden');
      $('dlg-state-message').className = 'error-state';
      $('dlg-state-message').textContent = 'Download was cancelled. No files were saved.';
      $('dlg-btn-download').textContent = 'Close';
      $('dlg-btn-download').disabled = false;
      $('dlg-btn-cancel').classList.remove('hidden');
      $('dlg-btn-cancel').textContent = 'Close';
      return;
    }

    // ── Completed ───────────────────────────────────────────────────────
    if (t === 'completed') {
      closeDownloadSSE();
      _dlgStopping = false;
      _setDlgState('completed');
      _announceToScreenReader('All models downloaded successfully.');
      window.ArtificeToast.success('ASR models downloaded.', { duration: 4000 });
      return;
    }

    // ── Error ──────────────────────────────────────────────────────────
    if (t === 'error') {
      closeDownloadSSE();
      const msg = event.error || 'Download failed for an unknown reason.';
      _setDlgState('error-view', { message: msg });
      $('dlg-error-box').classList.remove('hidden');
      $('dlg-error-text').textContent = msg;
      // Announce once, not repeatedly
      _announceToScreenReader(`Download error: ${msg}`);
      return;
    }

    // ── Heartbeat ──────────────────────────────────────────────────────
    // Silently ignored — keeps the connection alive; no UI update needed.
  });
}

// ── Progress bar helpers ─────────────────────────────────────────────────────

function _resetProgressBars() {
  _dlgLastAnnouncedPct = -1;
  $('dlg-overall-fill').style.width = '0%';
  $('dlg-progress-pct').textContent = '0%';
  $('dlg-overall-bar').setAttribute('aria-valuenow', 0);
  $('dlg-bytes-row').textContent = '—';

  const barsEl = $('dlg-model-bars');
  barsEl.innerHTML = '';
  if (_dlgCurrentInfo && _dlgCurrentInfo.models) {
    _dlgCurrentInfo.models.forEach((m, i) => {
      const row = document.createElement('div');
      row.className = 'dlg-model-bar-row';
      row.id = `dlg-model-bar-${i}`;
      row.innerHTML =
        '<div class="dlg-model-bar-meta">' +
          `<span class="dlg-model-bar-label"><code>${_esc(m.hf_repo)}</code></span>` +
          '<span class="dlg-model-bar-pct" id="dlg-model-pct-' + i + '">0%</span>' +
        '</div>' +
        '<div class="dlg-model-bar-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-label="' + _esc(m.hf_repo) + '">' +
          `<div class="dlg-model-bar-fill" id="dlg-model-fill-${i}" style="width:0%"></div>` +
        '</div>';
      barsEl.appendChild(row);
    });
  }
}

function _updateModelBar(modelIdx, modelTotal, pct, hfRepo) {
  const fill = $(`dlg-model-fill-${modelIdx}`);
  const pctEl = $(`dlg-model-pct-${modelIdx}`);
  const row   = $(`dlg-model-bar-${modelIdx}`);
  if (!row) return;
  if (fill) {
    fill.style.width = pct + '%';
    fill.parentElement.setAttribute('aria-valuenow', pct);
    fill.parentElement.setAttribute('aria-label', hfRepo || `Model ${modelIdx + 1}`);
  }
  if (pctEl) pctEl.textContent = pct + '%';
}

// ── Screen reader announcements ─────────────────────────────────────────────

function _announceToScreenReader(text) {
  const live = $('dlg-live-region');
  if (!live) return;
  // Zero-timeout ensures the announcement is picked up after the node is painted
  setTimeout(() => { live.textContent = ''; }, 50);
  setTimeout(() => { live.textContent = text; }, 100);
}

// ── Focus trap ──────────────────────────────────────────────────────────────

function _handleDlgKeydown(e) {
  if (e.key === 'Escape') {
    e.stopPropagation();
    closeDownloadDialog();
  }
}

// ── Init ────────────────────────────────────────────────────────────────────

function initDownloadDialog() {
  // Trigger button
  const manageBtn = $('btn-manage-models');
  if (manageBtn) {
    manageBtn.addEventListener('click', (e) => {
      openDownloadDialog(e.currentTarget);
    });
  }

  // Close button
  $('dlg-close').addEventListener('click', closeDownloadDialog);

  // Overlay click — close only in consent/token state (not mid-download)
  $('download-modal-overlay').addEventListener('click', (e) => {
    if (e.target === $('download-modal-overlay')) {
      closeDownloadDialog();
    }
  });

  // Keyboard
  $('download-modal-overlay').addEventListener('keydown', _handleDlgKeydown);

  // Download button
  $('dlg-btn-download').addEventListener('click', () => {
    const state = $('dlg-btn-download').textContent.trim();

    if (state === 'Download' || state === 'Continue') {
      if (!_dlgModelKey) return;
      _startDownload(_dlgModelKey);
      return;
    }

    if (state === 'Stop') {
      if (!_dlgModelKey) return;
      // Request cancellation; UI will update when SSE emits cancelling/cancelled
      api(`/models/${_dlgModelKey}/download/cancel`, { method: 'POST' }).catch(function(err) {
        if (window.ArtificeToast) window.ArtificeToast.error("Could not cancel download: " + err.message);
      });
      _dlgStopping = true;
      _setDlgState('stopping');
      return;
    }

    if (state === 'Retry') {
      if (!_dlgModelKey) return;
      _dlgStopping = false;
      _dlgLastAnnouncedPct = -1;
      _startDownload(_dlgModelKey);
      return;
    }

    if (state === 'Done' || state === 'Close') {
      closeDownloadDialog();
      return;
    }
  });

  // Cancel button
  $('dlg-btn-cancel').addEventListener('click', () => {
    const state = $('dlg-btn-cancel').textContent.trim();
    if (state === 'Cancel') {
      // Revoke consent if it was recorded (user opened but didn't download)
      if (_dlgModelKey && !_dlgSSE) {
        api(`/models/${_dlgModelKey}/consent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ consent: false }),
        }).catch(function(err) {
          if (window.ArtificeToast) window.ArtificeToast.error("Could not revoke consent: " + err.message);
        });
      }
      closeDownloadDialog();
      return;
    }
    if (state === 'Close') {
      closeDownloadDialog();
    }
  });
}


// ---------------------------------------------------------------- startup

  document.addEventListener('DOMContentLoaded', () => {
    initAsrCopyButtons();
    loadCapabilities();
    initTabs();
    initTheme();
    initUploadForm();
    initAudioHighlight();
    initHealthPanel();
    initGlobalSearch();
    initKeyboardShortcuts();
    initSettingsPanel();
    initDownloadDialog();
    loadConfig();
    loadModelConfig();
    loadHealth();
    $('btn-library-refresh').addEventListener('click', loadLibrary);
    $('btn-search-close').addEventListener('click', closeSearchModal);
    $('btn-ai-close').addEventListener('click', closeAIModal);
    $('btn-ai-copy').addEventListener('click', copyAIResult);
    $('btn-ai-download').addEventListener('click', downloadAIResult);
    // Close AI modal on overlay click
    $('ai-modal').addEventListener('click', (e) => {
      if (e.target === $('ai-modal')) closeAIModal();
    });

    // Dictionary
    $('btn-save-dictionary').addEventListener('click', saveDictionary);
    $('btn-enroll-speaker').addEventListener('click', enrollSpeaker);
    $('btn-enroll-from-job').addEventListener('click', enrollFromJob);
    $('btn-match-speakers').addEventListener('click', matchSpeakers);

    // Load dictionary when its tab is activated
    const origTabClick = document.querySelector('.tab[data-tab="dictionary"]');
    if (origTabClick) {
      origTabClick.addEventListener('click', () => {
        loadDictionary();
        loadKnownSpeakers();
      });
    }
  });
})();
