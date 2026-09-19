import sqlite3
from concurrent.futures import ThreadPoolExecutor

from astra_shared import live_source_store


def test_run_cas_rejects_stale_updates_and_closes_by_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "user_id": "u"})

    initial = live_source_store.retrieve_run(run_id)
    assert initial["_state_version"] == 0
    assert initial["_lifecycle_state"] == "active"

    assert live_source_store.update_run_cas(
        run_id,
        {"kind": "test", "sample": 1},
        expected_version=0,
        expected_generation=0,
    )
    assert not live_source_store.update_run_cas(
        run_id,
        {"kind": "test", "sample": "stale"},
        expected_version=0,
        expected_generation=0,
    )

    assert live_source_store.update_run_cas(
        run_id,
        {"kind": "test", "sample": 1},
        expected_version=None,
        expected_generation=0,
        new_lifecycle_state="closed",
    )
    assert live_source_store.retrieve_run(run_id)["_lifecycle_generation"] == 1
    assert not live_source_store.update_run_cas(
        run_id,
        {"kind": "test", "sample": 2},
        expected_version=None,
        expected_generation=0,
    )


def test_store_run_does_not_replace_existing_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "sample": "original"}, run_id="reserved-run")

    try:
        live_source_store.store_run({"kind": "test", "sample": "replacement"}, run_id=run_id)
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("store_run replaced an existing run id")

    assert live_source_store.retrieve_run(run_id)["sample"] == "original"


def test_point_commit_is_sqlite_authoritative_and_exact_retry_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "user_id": "u", "point_analysis": {}})
    row = {"key": f"{run_id}:2026-01-01T00:00:00Z:SAT-1:hash", "time_iso": "2026-01-01T00:00:00Z", "row": ["SAT-1"]}

    assert live_source_store.commit_point_run_cas(
        run_id,
        {"kind": "test", "point_analysis": {"csv_rows": [row]}},
        expected_version=0,
        projection_rows=[row],
    ) == "committed"
    assert live_source_store.commit_point_run_cas(
        run_id,
        {"kind": "test", "point_analysis": {"csv_rows": [row]}},
        expected_version=0,
        projection_rows=[row],
    ) == "already_committed"
    assert len(live_source_store.retrieve_point_projection_rows(run_id)) == 1
    assert live_source_store.retrieve_run(run_id)["_state_version"] == 1


def test_failed_run_is_durable_and_cannot_be_updated(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "user_id": "u"})
    assert live_source_store.fail_run(run_id, {"code": "boom", "message": "injected failure"})
    failed = live_source_store.retrieve_run(run_id)
    assert failed["_lifecycle_state"] == "failed"
    assert failed["_lifecycle_generation"] == 1
    assert failed["run_failure"]["schema"] == "earth_fixed_run_failure_v1"
    assert not live_source_store.update_run_cas(run_id, {"kind": "test", "sample": 1}, expected_version=None)


def test_concurrent_point_writers_have_one_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "user_id": "u", "point_analysis": {}})

    def commit(sample):
        row = {"key": f"{run_id}:2026-01-01T00:00:0{sample}Z:SAT-1:hash", "time_iso": f"2026-01-01T00:00:0{sample}Z", "row": ["SAT-1"]}
        return live_source_store.commit_point_run_cas(
            run_id,
            {"kind": "test", "sample": sample},
            expected_version=0,
            projection_rows=[row],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(commit, (1, 2)))
    assert outcomes == ["committed", "conflict"]
    assert live_source_store.retrieve_run(run_id)["_state_version"] == 1


def test_point_commit_rolls_back_projection_when_sqlite_snapshot_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test", "user_id": "u"})
    original_connect = live_source_store._connect

    class FailingConnection:
        def __init__(self):
            self.connection = original_connect()

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("UPDATE LIVE_RUNS"):
                raise sqlite3.IntegrityError("injected snapshot failure")
            return self.connection.execute(sql, params)

    monkeypatch.setattr(live_source_store, "_connect", lambda: FailingConnection())
    row = {"key": f"{run_id}:instant:SAT-1:hash", "time_iso": "instant", "row": ["SAT-1"]}
    try:
        live_source_store.commit_point_run_cas(
            run_id,
            {"kind": "test", "point_analysis": {"csv_rows": [row]}},
            expected_version=0,
            projection_rows=[row],
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("injected snapshot failure was not raised")
    assert live_source_store.retrieve_point_projection_rows(run_id) == []
    assert live_source_store.retrieve_run(run_id)["_state_version"] == 0
