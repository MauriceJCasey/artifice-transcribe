# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from pathlib import Path

from artifice_transcribe.config import Settings


class TestLegacyDBMigration:
    """Coverage for the ``_migrate_legacy_db`` path in ``Settings.model_post_init``."""

    # ------------------------------------------------------------------
    # Neither file present
    # ------------------------------------------------------------------

    def test_no_legacy_no_existing_noop(self, tmp_path, monkeypatch):
        """Neither legacy nor new database exists — nothing happens."""
        new_db = tmp_path / "new" / "transcribe.db"
        legacy = tmp_path / "legacy" / "transcribe.db"
        new_db_url = f"sqlite+aiosqlite:///{new_db}"

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_PATH", new_db)
        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_URL", new_db_url)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_DB_PATH", legacy)

        settings = Settings()
        # model_post_init runs _migrate_legacy_db, but self.database_url
        # (captured from the class definition) won't match the patched
        # _DEFAULT_DB_URL, so it returns early.  Align them and call again.
        settings.database_url = new_db_url
        settings._migrate_legacy_db()

        assert not new_db.exists()
        assert not legacy.exists()
        assert settings.database_url == new_db_url

    # ------------------------------------------------------------------
    # Legacy present, new absent → move
    # ------------------------------------------------------------------

    def test_legacy_exists_new_absent_moves(self, tmp_path, monkeypatch, caplog):
        """Legacy database exists and new does not — the file is *moved*."""
        new_db = tmp_path / "new" / "transcribe.db"
        new_db.parent.mkdir()
        legacy = tmp_path / "legacy" / "transcribe.db"
        legacy.parent.mkdir()
        legacy.write_text("interview-archive-data")

        new_db_url = f"sqlite+aiosqlite:///{new_db}"

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_PATH", new_db)
        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_URL", new_db_url)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_DB_PATH", legacy)

        settings = Settings()
        settings.database_url = new_db_url

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_db()

        assert new_db.exists()
        assert new_db.read_text() == "interview-archive-data"
        assert not legacy.exists()
        assert "Migrating legacy file" in caplog.text
        assert "Migration complete — file is now at" in caplog.text

    # ------------------------------------------------------------------
    # Both present → use new, warn
    # ------------------------------------------------------------------

    def test_both_exist_uses_new_warns(self, tmp_path, monkeypatch, caplog):
        """Both legacy and new databases exist — use new, log warning."""
        new_db = tmp_path / "new" / "transcribe.db"
        new_db.parent.mkdir()
        new_db.write_text("new-data")

        legacy = tmp_path / "legacy" / "transcribe.db"
        legacy.parent.mkdir()
        legacy.write_text("stale-data")

        new_db_url = f"sqlite+aiosqlite:///{new_db}"

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_PATH", new_db)
        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_URL", new_db_url)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_DB_PATH", legacy)

        settings = Settings()
        settings.database_url = new_db_url

        with caplog.at_level(logging.WARNING):
            settings._migrate_legacy_db()

        assert new_db.exists()
        assert new_db.read_text() == "new-data"  # untouched
        assert legacy.exists()  # left alone for manual recovery
        assert "Legacy file found" in caplog.text

    # ------------------------------------------------------------------
    # Custom DATABASE_URL → skip
    # ------------------------------------------------------------------

    def test_custom_database_url_skips_migration(self, tmp_path, monkeypatch, caplog):
        """When the user set DATABASE_URL, migration is skipped entirely."""
        custom_db = tmp_path / "custom" / "transcribe.db"
        custom_db.parent.mkdir()

        new_db = tmp_path / "new" / "transcribe.db"
        legacy = tmp_path / "legacy" / "transcribe.db"
        legacy.parent.mkdir()
        legacy.write_text("should-not-be-touched")

        custom_db_url = f"sqlite+aiosqlite:///{custom_db}"
        new_db_url = f"sqlite+aiosqlite:///{new_db}"

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_PATH", new_db)
        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_URL", new_db_url)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_DB_PATH", legacy)
        monkeypatch.setenv("DATABASE_URL", custom_db_url)

        settings = Settings()
        # model_post_init sees a non-default URL and returns immediately
        assert settings.database_url == custom_db_url

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_db()

        assert not new_db.exists()
        assert legacy.exists()
        assert "Migrating" not in caplog.text

    # ------------------------------------------------------------------
    # data_path returns the platform user-data directory
    # ------------------------------------------------------------------

    def test_data_path_uses_platformdirs(self, tmp_path, monkeypatch):
        """``data_path`` returns the resolved user-data directory, not
        ``./data``."""
        user_dir = tmp_path / "user-data"
        user_dir.mkdir()
        db_file = user_dir / "transcribe.db"

        monkeypatch.setattr("artifice_transcribe.config._USER_DATA_PATH", user_dir)
        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_DB_PATH", db_file)
        monkeypatch.setattr(
            "artifice_transcribe.config._DEFAULT_DB_URL",
            f"sqlite+aiosqlite:///{db_file}",
        )
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_DB_PATH", Path("/nonexistent"))

        settings = Settings()
        assert settings.data_path == user_dir


class TestLegacyUploadMigration:
    """Coverage for the ``_migrate_legacy_uploads`` path in
    ``Settings.model_post_init``."""

    # ------------------------------------------------------------------
    # Neither directory present
    # ------------------------------------------------------------------

    def test_no_legacy_no_default_noop(self, tmp_path, monkeypatch):
        """Neither legacy nor default upload directory exists — nothing happens."""
        default = tmp_path / "default" / "uploads"
        legacy = tmp_path / "legacy" / "uploads"

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)

        settings = Settings()
        # upload_dir captured from class definition won't match patched
        # _DEFAULT_UPLOAD_PATH, so align and call again.
        settings.upload_dir = str(default)
        settings._migrate_legacy_uploads()

        assert not default.exists()
        assert not legacy.exists()

    # ------------------------------------------------------------------
    # Legacy present, default absent → move files
    # ------------------------------------------------------------------

    def test_legacy_exists_default_absent_moves(self, tmp_path, monkeypatch, caplog):
        """Legacy upload directory has files, default does not — files are
        *moved*."""
        default = tmp_path / "default" / "uploads"
        legacy = tmp_path / "legacy" / "uploads"
        legacy.mkdir(parents=True)
        (legacy / "interview1.wav").write_text("audio-data-1")
        (legacy / "interview2.mp3").write_text("audio-data-2")

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)

        settings = Settings()
        settings.upload_dir = str(default)

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_uploads()

        assert default.exists()
        moved1 = default / "interview1.wav"
        moved2 = default / "interview2.mp3"
        assert moved1.exists()
        assert moved1.read_text() == "audio-data-1"
        assert moved2.exists()
        assert moved2.read_text() == "audio-data-2"

        assert not legacy.exists()  # emptied and cleaned up

        assert "Migration complete — moved 2 file(s)" in caplog.text

    # ------------------------------------------------------------------
    # Both have files, no name collisions → migrate non-conflicting files
    # ------------------------------------------------------------------

    def test_default_has_unrelated_file_legacy_still_migrates(self, tmp_path, monkeypatch, caplog):
        """Default upload directory has unrelated files — legacy files with
        non-colliding names are still migrated, and the empty legacy directory
        is cleaned up."""
        default = tmp_path / "default" / "uploads"
        default.mkdir(parents=True)
        (default / "existing.wav").write_text("new-data")

        legacy = tmp_path / "legacy" / "uploads"
        legacy.mkdir(parents=True)
        (legacy / "old.wav").write_text("old-data")

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)

        settings = Settings()
        settings.upload_dir = str(default)

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_uploads()

        assert default.exists()
        assert (default / "existing.wav").read_text() == "new-data"  # untouched
        assert (default / "old.wav").exists()  # migrated — no name collision
        assert (default / "old.wav").read_text() == "old-data"
        assert not legacy.exists()  # emptied and cleaned up
        assert "Migration complete — moved 1 file(s)" in caplog.text

    # ------------------------------------------------------------------
    # Both have files, one name collides → skip colliding, migrate rest
    # ------------------------------------------------------------------

    def test_both_exist_name_collision_warns(self, tmp_path, monkeypatch, caplog):
        """Legacy and default both have files, and one filename overlaps —
        the colliding file is skipped but non-colliding files are still
        migrated."""
        default = tmp_path / "default" / "uploads"
        default.mkdir(parents=True)
        (default / "shared.wav").write_text("new-version")

        legacy = tmp_path / "legacy" / "uploads"
        legacy.mkdir(parents=True)
        (legacy / "shared.wav").write_text("old-version")
        (legacy / "other.mp3").write_text("old-other")

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)

        settings = Settings()
        settings.upload_dir = str(default)

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_uploads()

        assert default.exists()
        assert (default / "shared.wav").read_text() == "new-version"  # default wins
        assert (default / "other.mp3").exists()  # migrated — no collision
        assert (default / "other.mp3").read_text() == "old-other"
        assert legacy.exists()  # still has shared.wav — not empty
        assert (legacy / "shared.wav").exists()  # left alone (collision)
        assert not (legacy / "other.mp3").exists()  # migrated away
        assert "File shared.wav already exists at destination" in caplog.text
        assert "Migration complete — moved 1 file(s)" in caplog.text

    # ------------------------------------------------------------------
    # Custom UPLOAD_DIR → skip
    # ------------------------------------------------------------------

    def test_custom_upload_dir_skips_migration(self, tmp_path, monkeypatch, caplog):
        """When the user set UPLOAD_DIR, migration is skipped entirely."""
        custom_dir = tmp_path / "custom" / "uploads"
        custom_dir.mkdir(parents=True)

        default = tmp_path / "default" / "uploads"
        legacy = tmp_path / "legacy" / "uploads"
        legacy.mkdir(parents=True)
        (legacy / "should-not-touch.wav").write_text("legacy-data")

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)
        monkeypatch.setenv("UPLOAD_DIR", str(custom_dir))

        settings = Settings()
        # model_post_init sees a non-default upload_dir and returns immediately
        assert settings.upload_dir == str(custom_dir)

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_uploads()

        assert not default.exists()
        assert legacy.exists()
        assert (legacy / "should-not-touch.wav").exists()
        assert "Migrating" not in caplog.text

    # ------------------------------------------------------------------
    # Legacy directory empty → noop, clean up empty dir
    # ------------------------------------------------------------------

    def test_legacy_empty_no_migration(self, tmp_path, monkeypatch, caplog):
        """An empty legacy ``./uploads/`` directory triggers no migration
        log line and is cleaned up silently."""
        default = tmp_path / "default" / "uploads"
        legacy = tmp_path / "legacy" / "uploads"
        legacy.mkdir(parents=True)  # empty

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", default)
        monkeypatch.setattr("artifice_transcribe.config._LEGACY_UPLOAD_PATH", legacy)

        settings = Settings()
        settings.upload_dir = str(default)

        with caplog.at_level(logging.INFO):
            settings._migrate_legacy_uploads()

        assert not legacy.exists()  # cleaned up
        assert "Migrating" not in caplog.text
        assert "moved" not in caplog.text

    # ------------------------------------------------------------------
    # upload_path uses the resolved platform directory
    # ------------------------------------------------------------------

    def test_upload_path_uses_platformdirs(self, tmp_path, monkeypatch):
        """``upload_path`` returns the resolved user-data subdirectory, not
        a relative ``./uploads``."""
        upload_dir = tmp_path / "user-data" / "uploads"
        upload_dir.mkdir(parents=True)

        monkeypatch.setattr("artifice_transcribe.config._DEFAULT_UPLOAD_PATH", upload_dir)
        # Prevent the migration from tripping over paths that don't exist
        monkeypatch.setattr(
            "artifice_transcribe.config._LEGACY_UPLOAD_PATH",
            tmp_path / "nonexistent-legacy",
        )

        settings = Settings()
        settings.upload_dir = str(upload_dir)
        assert settings.upload_path == upload_dir
