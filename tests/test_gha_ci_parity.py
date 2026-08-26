#!/usr/bin/env python3
import os
import tempfile
import unittest
import warnings

import sqlalchemy.exc


class TestGhaCiParity(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._old_env = {
            k: os.environ.get(k) for k in ("AEPSYCH_MODE", "CI", "SQLALCHEMY_WARN_20")
        }
        self._old_showwarning = warnings.showwarning
        self._old_filters = warnings.filters[:]
        os.environ["AEPSYCH_MODE"] = "test"
        os.environ["CI"] = "true"
        os.environ["SQLALCHEMY_WARN_20"] = "1"
        from aepsych.utils_logging import _set_test_warning_filters

        _set_test_warning_filters()
        warnings.filterwarnings("always", category=sqlalchemy.exc.RemovedIn20Warning)
        warnings.filterwarnings("always", category=sqlalchemy.exc.SAWarning)

    def tearDown(self):
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        warnings.showwarning = self._old_showwarning
        warnings.filters[:] = self._old_filters
        super().tearDown()

    def test_ci_parity_removed_in_20_promoted_to_exception(self):
        """Provide GHA CI parity: SA ``RemovedIn20Warning`` is promoted."""

        # Verify production filter promotes the warning emitted at
        # aepsych/database/db.py:391 via sqlalchemy.orm.unitofwork
        # ``_warn_for_cascade_backrefs``. The message matches the CI log.
        with self.assertRaises(sqlalchemy.exc.RemovedIn20Warning):
            warnings.warn(
                'Deprecated API features detected! These feature(s) are not compatible with SQLAlchemy 2.0. To prevent incompatible upgrades prior to updating applications, ensure requirements files are pinned to "sqlalchemy<2.0". Set environment variable SQLALCHEMY_WARN_20=1 to show all deprecation warnings. Set environment variable SQLALCHEMY_SILENCE_UBER_WARNING=1 to silence this message. (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)',
                category=sqlalchemy.exc.RemovedIn20Warning,
                stacklevel=2,
            )

        # Ignored warnings from utils_logging._IGNORED_WARNINGS must not raise.
        try:
            warnings.warn(
                "A not p.d., added jitter of 1e-06 to the diagonal",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            self.fail(f"Ignored warning was incorrectly promoted: {e}")

    def test_ci_parity_db_record_setup_cascade_backref(self):
        """Provide GHA CI parity for ``db.py:391 record.parent = master_table``."""

        from aepsych.database.db import Database

        tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmpdir.cleanup)
        log_path = os.path.join(os.getcwd(), "logs", "aepsych_server.log")
        self.addCleanup(lambda: os.path.exists(log_path) and os.unlink(log_path))
        self.addCleanup(
            lambda: os.path.isdir("logs")
            and not os.listdir("logs")
            and os.rmdir("logs")
        )

        db_path = os.path.join(tmpdir.name, "test.db")
        db = Database(db_path=db_path)

        def _cleanup_db():
            try:
                db.delete_db()
            except Exception:
                pass
            for suffix in ("", "-journal", "-wal", "-shm"):
                p = db_path + suffix
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        self.addCleanup(_cleanup_db)
        # record_setup at db.py:391 would emit cascade_backrefs warning before
        # the tables.py fix; under CI parity it would be promoted to exception.
        # Success means no RemovedIn20Warning was raised.
        master_table = db.record_setup(
            description="test", name="test", request={"test": "test request"}
        )
        self.assertIsNotNone(master_table)
        self.assertIsNotNone(master_table.unique_id)

        # Verify persistence via public DB API.
        fetched = db.get_master_record(master_table.unique_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.experiment_name, "test")
        records = db.get_master_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].unique_id, master_table.unique_id)

        # Raw/replay children should be accessible without warning.
        self.assertIsNotNone(db.get_replay_for(master_table.unique_id))
        _cleanup_db()
        self.assertFalse(os.path.exists(db_path))
