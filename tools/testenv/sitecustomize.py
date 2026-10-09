"""Sandbox compatibility shim for running the test suite on Windows.

Two DSH sandbox behaviours break pytest and ChromaDB here:

1. The per-process directory under ``C:\\WINDOWS\\TEMP`` is not writable, so
   every ``tmp_path`` and ``mkdtemp`` call fails. Temporary files are
   redirected into the workspace instead.
2. A directory created with owner-only mode (``0o700``, what
   ``tempfile.mkdtemp`` uses) cannot be written into by this process, which
   surfaces as ``sqlite3.OperationalError: unable to open database file``.
   Directory creation is therefore requested with a permissive mode, which
   matters on Windows only because the sandbox reads the mode argument.

Imported automatically through PYTHONPATH; see tools/README.md. This file is
not part of the application and is never imported by the RAG runtime.
"""
import os
import pathlib
import tempfile

_root = pathlib.Path(__file__).resolve().parents[2]
_candidate = _root / '.tmp'
_candidate.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(_candidate)
os.environ['TEMP'] = str(_candidate)
os.environ['TMP'] = str(_candidate)

_os_mkdir = os.mkdir
_os_makedirs = os.makedirs


def _mkdir(path, mode=0o777, *args, **kwargs):
    return _os_mkdir(path, 0o777, *args, **kwargs)


def _makedirs(name, mode=0o777, exist_ok=False):
    return _os_makedirs(name, 0o777, exist_ok=exist_ok)


os.mkdir = _mkdir
os.makedirs = _makedirs

