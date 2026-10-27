"""Shared reference-data store for the strategy pipeline.

Every reference dataset (TAIFEX futures quotes, settlement calendars, TWSE
indices, later TXO quotes and corporate actions) lives under one root and
follows the same contract, so a stage can name the exact bytes it read with
one hash.

Root
    ``settings.pipeline_refdata_root`` (env ``POSEIDON_PIPELINE_REFDATA_ROOT``,
    default ``~/strategy-pipeline/refdata``). Pass ``root=`` to override.

Layout (one directory per dataset; dataset names match ``[a-z0-9_]+``)::

    <root>/<dataset>/manifest.json
    <root>/<dataset>/partitions/<YYYY-MM>.parquet
    <root>/<dataset>/raw/<raw name>                 # exact bytes a source returned
    <root>/<dataset>/raw/<raw name>.source.json     # its SourceRecord
    <root>/<dataset>/.lock                          # flock: exclusive to write, shared to read
                                                    # (a read-only mount is read without the lock)

Partitioning
    By calendar month of the dataset's ``date_column`` (a naive
    ``datetime64[ns]`` holding the Asia/Taipei trading date). Key ``YYYY-MM``.
    Each partition is one parquet file written without an index, rows sorted
    by the builder's sort key.

``manifest.json``::

    {
      "dataset": "taifex_futures_daily",
      "schema_version": 1,              # bump whenever parsing would change the rows for the same raw bytes
      "date_column": "trade_date",
      "sources": [{"name", "url", "params", "fetched_at", "sha256_raw"}, ...],
      "partitions": [{"key": "2026-10", "rows", "start", "end",
                      "sha256",          # content hash of the rows (see below)
                      "file_sha256"}],   # sha256 of the parquet bytes, checked on every read
      "coverage": {"start": "1998-07-21", "end": "2026-10-23"},
      "content_sha256": "...",          # dataset-level hash: put THIS in run input hashes
      "built_at": "2026-10-24T10:00:00Z",
      "code_version": "<git sha of the builder>"
    }

    ``sources`` lists every raw fetch the current rows were parsed from. A
    derived dataset (built from other datasets) lists them as
    ``url = "refdata://<dataset>"`` with ``sha256_raw`` = that dataset's
    ``content_sha256``. ``fetched_at`` is when these exact bytes were first
    fetched: a re-fetch returning the same bytes keeps the old record.

Hashes
    A partition's ``sha256`` hashes a canonical CSV rendering of its rows
    (column names, dtypes, values), so it does not change when pyarrow
    rewrites the same rows into different parquet bytes. ``content_sha256``
    hashes ``{dataset, schema_version, [[key, sha256], ...]}``: it changes iff
    the rows a reader can see change. ``file_sha256`` only guards integrity.

Writing (idempotent)
    :meth:`RefdataStore.commit` receives the rebuilt partitions; a partition
    whose content hash is unchanged is not rewritten, and when neither rows
    nor sources change the manifest is left byte-for-byte as it was (so
    ``built_at`` names the last real change). A daily job therefore re-fetches
    only the current and previous month, rebuilds those partitions and
    usually commits nothing. A changed ``schema_version`` requires a full
    rebuild (``full=True``), which also drops partitions absent from the
    rebuild.

Reading
    :meth:`RefdataStore.read` returns the rows with ``start <= date <= end``
    (both optional, inclusive) as one DataFrame and verifies every parquet
    file it opens against ``file_sha256``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from poseidon.core.config import settings
from poseidon.decision_loop.manifest import content_sha256

MANIFEST = "manifest.json"
PARTITIONS = "partitions"
RAW = "raw"
_DATASET_NAME = re.compile(r"[a-z0-9_]{1,64}")
_RAW_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_KEY = re.compile(r"\d{4}-\d{2}")


class RefdataError(RuntimeError):
    """The store refused an operation or found data that does not match its manifest."""


class DatasetMissingError(RefdataError):
    """No manifest exists for the dataset yet."""


@dataclass(frozen=True)
class SourceRecord:
    name: str  # raw file name inside <dataset>/raw/ ("" for a derived source)
    url: str
    params: dict
    fetched_at: str  # ISO-8601 UTC
    sha256_raw: str

    @classmethod
    def from_json(cls, value: Mapping) -> SourceRecord:
        return cls(value["name"], value["url"], dict(value["params"]), value["fetched_at"], value["sha256_raw"])


def refdata_root(override: str | os.PathLike | None = None) -> Path:
    return Path(override if override is not None else settings.pipeline_refdata_root).expanduser()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def month_key(value: date | datetime | pd.Timestamp) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    """Content hash of rows: column names, dtypes and a canonical CSV of the values."""

    header = json.dumps([[str(column), str(dtype)] for column, dtype in frame.dtypes.items()])
    body = frame.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S", na_rep="")
    return sha256_bytes((header + "\n" + body).encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _iso_day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class CommitResult:
    manifest: dict
    changed: bool
    written: tuple[str, ...]  # partition keys whose parquet file was (re)written
    removed: tuple[str, ...]


class RefdataStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = refdata_root(root)

    # --- paths and locking -------------------------------------------------

    def dataset_dir(self, dataset: str) -> Path:
        if not _DATASET_NAME.fullmatch(dataset):
            raise RefdataError(f"invalid dataset name {dataset!r}")
        return self.root / dataset

    @contextmanager
    def _lock(self, dataset: str, mode: int) -> Iterator[Path]:
        directory = self.dataset_dir(dataset)
        if mode == fcntl.LOCK_SH:
            # readers may sit on a read-only mount: lock the existing file if we can, else read unlocked
            try:
                handle = (directory / ".lock").open("r")
            except OSError:
                yield directory
                return
            with handle:
                fcntl.flock(handle, mode)
                yield directory
            return
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as handle:
            fcntl.flock(handle, mode)
            yield directory

    # --- raw source bytes ----------------------------------------------------

    def raw_path(self, dataset: str, name: str) -> Path:
        if not _RAW_NAME.fullmatch(name) or name.endswith(".source.json"):
            raise RefdataError(f"invalid raw name {name!r}")
        return self.dataset_dir(dataset) / RAW / name

    def raw_record(self, dataset: str, name: str) -> SourceRecord | None:
        """The stored record for ``name`` if its bytes are present and match it."""

        path = self.raw_path(dataset, name)
        sidecar = path.with_name(f"{name}.source.json")
        if not path.exists() or not sidecar.exists():
            return None
        record = SourceRecord.from_json(json.loads(sidecar.read_text()))
        if record.sha256_raw != _sha256_file(path):
            raise RefdataError(f"raw file {path} does not match its source record")
        return record

    def raw_bytes(self, dataset: str, name: str) -> bytes:
        if self.raw_record(dataset, name) is None:
            raise RefdataError(f"no raw file {name} in {dataset}")
        return self.raw_path(dataset, name).read_bytes()

    def put_raw(
        self,
        dataset: str,
        name: str,
        data: bytes,
        *,
        url: str,
        params: Mapping,
        fetched_at: str | None = None,
        equivalent: Callable[[bytes, bytes], bool] | None = None,
    ) -> tuple[SourceRecord, bool]:
        """Store fetched bytes; returns (record, changed).

        If a stored copy has the same bytes (or ``equivalent(old, new)`` says
        they carry the same data, e.g. an HTML page whose table did not
        change), the stored copy and its ``fetched_at`` are kept.
        """

        path = self.raw_path(dataset, name)
        with self._lock(dataset, fcntl.LOCK_EX):
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = self.raw_record(dataset, name)
            if existing is not None:
                old = path.read_bytes()
                if old == data or (equivalent is not None and equivalent(old, data)):
                    return existing, False
            record = SourceRecord(name, url, dict(params), fetched_at or utc_now(), sha256_bytes(data))
            _atomic_write(path, data)
            _atomic_write(path.with_name(f"{name}.source.json"), json.dumps(asdict(record), sort_keys=True).encode())
            return record, True

    # --- manifest and reads -------------------------------------------------

    def manifest(self, dataset: str) -> dict:
        path = self.dataset_dir(dataset) / MANIFEST
        if not path.exists():
            raise DatasetMissingError(f"refdata dataset {dataset} has no manifest under {self.root}")
        return json.loads(path.read_text())

    def exists(self, dataset: str) -> bool:
        return (self.dataset_dir(dataset) / MANIFEST).exists()

    def content_sha256(self, dataset: str) -> str:
        return self.manifest(dataset)["content_sha256"]

    def read(self, dataset: str, start=None, end=None, rows=None) -> pd.DataFrame:
        """Rows with ``start <= date_column <= end`` (inclusive, both optional).

        ``rows``: a frame -> frame filter applied to each partition as it is read (it must keep the
        date column), so a large dataset is never held whole in memory.
        """

        lower = pd.Timestamp(start) if start is not None else None
        upper = pd.Timestamp(end) if end is not None else None
        with self._lock(dataset, fcntl.LOCK_SH) as directory:
            manifest = self.manifest(dataset)
            column = manifest["date_column"]
            frames = []
            for part in manifest["partitions"]:
                if (lower is not None and pd.Timestamp(part["end"]) < lower) or (
                    upper is not None and pd.Timestamp(part["start"]) > upper
                ):
                    continue
                path = directory / PARTITIONS / f"{part['key']}.parquet"
                if not path.exists() or _sha256_file(path) != part["file_sha256"]:
                    raise RefdataError(f"{dataset} partition {part['key']} does not match its manifest")
                frame = pd.read_parquet(path)
                frames.append(frame if rows is None else rows(frame))
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        mask = pd.Series(True, index=frame.index)
        if lower is not None:
            mask &= frame[column] >= lower
        if upper is not None:
            mask &= frame[column] <= upper
        return frame[mask].reset_index(drop=True)

    # --- writes ----------------------------------------------------------------

    def commit(
        self,
        dataset: str,
        *,
        schema_version: int,
        date_column: str,
        partitions: Mapping[str, pd.DataFrame],
        sources: Sequence[SourceRecord],
        code_version: str,
        full: bool = False,
        meta: Mapping | None = None,
    ) -> CommitResult:
        """Write rebuilt partitions (an empty frame removes the key) and the manifest.

        ``sources`` is the complete source list of the dataset after this
        commit. Without ``full`` the partitions not named are kept. ``meta`` is a small JSON mapping
        kept in the manifest beside the content hash (it never enters it), e.g. the day a snapshot is
        complete through; a changed ``meta`` rewrites the manifest.
        """

        for key, frame in partitions.items():
            if not _KEY.fullmatch(key):
                raise RefdataError(f"invalid partition key {key!r}")
            if len(frame) and {month_key(value) for value in frame[date_column].unique()} != {key}:
                raise RefdataError(f"{dataset} partition {key} holds rows from another month")
        with self._lock(dataset, fcntl.LOCK_EX) as directory:
            try:
                old = self.manifest(dataset)
            except DatasetMissingError:
                old = None
            if old is not None and old["schema_version"] != schema_version and not full:
                raise RefdataError(f"{dataset} schema {old['schema_version']} -> {schema_version} needs a full rebuild")
            entries = {} if old is None or full else {part["key"]: part for part in old["partitions"]}
            previous = {} if old is None else {part["key"]: part for part in old["partitions"]}
            (directory / PARTITIONS).mkdir(exist_ok=True)
            written, removed = [], []
            for key in sorted(partitions):
                frame = partitions[key].reset_index(drop=True)
                path = directory / PARTITIONS / f"{key}.parquet"
                if not len(frame):
                    entries.pop(key, None)
                    continue
                sha = frame_sha256(frame)
                prior = previous.get(key)
                if (
                    prior is not None
                    and prior["sha256"] == sha
                    and path.exists()
                    and (_sha256_file(path) == prior["file_sha256"])
                ):
                    entries[key] = prior
                    continue
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                frame.to_parquet(temporary, engine="pyarrow", index=False)
                file_sha = _sha256_file(temporary)
                os.replace(temporary, path)
                written.append(key)
                entries[key] = {
                    "key": key,
                    "rows": len(frame),
                    "start": _iso_day(frame[date_column].min()),
                    "end": _iso_day(frame[date_column].max()),
                    "sha256": sha,
                    "file_sha256": file_sha,
                }
            for key in sorted(set(previous) - set(entries)):
                (directory / PARTITIONS / f"{key}.parquet").unlink(missing_ok=True)
                removed.append(key)
            ordered = [entries[key] for key in sorted(entries)]
            source_list = sorted((asdict(source) for source in sources), key=lambda item: (item["name"], item["url"]))
            content = content_sha256(
                {
                    "dataset": dataset,
                    "schema_version": schema_version,
                    "partitions": [[part["key"], part["sha256"]] for part in ordered],
                }
            )
            unchanged = (
                old is not None
                and old["content_sha256"] == content
                and old["sources"] == source_list
                and old["date_column"] == date_column
                and old["partitions"] == ordered
                and old.get("meta") == (None if meta is None else dict(meta))
            )
            if unchanged:
                return CommitResult(old, False, tuple(written), tuple(removed))
            manifest = {
                "dataset": dataset,
                "schema_version": schema_version,
                "date_column": date_column,
                "sources": source_list,
                "partitions": ordered,
                "coverage": {
                    "start": ordered[0]["start"] if ordered else None,
                    "end": ordered[-1]["end"] if ordered else None,
                },
                "content_sha256": content,
                "built_at": utc_now(),
                "code_version": code_version,
                **({} if meta is None else {"meta": dict(meta)}),
            }
            _atomic_write(directory / MANIFEST, json.dumps(manifest, indent=1, sort_keys=True).encode())
            return CommitResult(manifest, True, tuple(written), tuple(removed))
