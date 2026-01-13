"""Jupyter-safe file selection helpers (uses ipywidgets instead of Tk)."""

from __future__ import annotations
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple
import asyncio
try:
    from tkinter import Tk, filedialog  # Tk fallback for environments where widgets misbehave
except Exception:  # pragma: no cover
    Tk = None
    filedialog = None


def _should_use_widget() -> bool:
    """Return True when running in an IPython notebook environment where widgets make sense."""
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    shell = getattr(ip, "__class__", type("x", (), {})).__name__.lower()
    return "zmqshell" in shell or "terminalinteractiveshell" in shell

def _extract_upload(value: Any) -> Tuple[Optional[str], Optional[bytes]]:
    """Normalize ipywidgets FileUpload value to (filename, content_bytes)."""
    if value is None:
        return None, None

    # ipywidgets < 8: dict mapping filename -> {"metadata": {"name": ...}, "content": b""}
    if isinstance(value, dict) and value:
        key, payload = next(iter(value.items()))
        name = None
        content = None
        if isinstance(payload, dict):
            meta = payload.get("metadata") or {}
            name = meta.get("name") or payload.get("name") or key
            content = payload.get("content")
        return name, content

    # ipywidgets >= 8: tuple/list of UploadedFile objects
    if isinstance(value, (tuple, list)) and value:
        item = value[0]
        name = getattr(item, "name", None) or getattr(item, "metadata", {}).get("name") or getattr(item, "filename", None)
        if isinstance(item, dict):
            content = item.get("content")
        else:
            content = getattr(item, "content", None) or getattr(item, "data", None)
        return name, content

    return None, None


def uploaded_value_to_path(upload_or_value: Any, preferred_suffix: str = "") -> Optional[str]:
    """
    Convert an ipywidgets FileUpload (or its ``.value``) to a temp file path.

    - upload_or_value may be the widget itself, its .value, or the raw dict/tuple.
    - preferred_suffix is appended when the upload lacks an extension.
    """
    value = getattr(upload_or_value, "value", upload_or_value)
    filename, content = _extract_upload(value)
    if not content:
        return None

    suffix = Path(filename).suffix if filename else ""
    if not suffix and preferred_suffix:
        suffix = preferred_suffix if preferred_suffix.startswith(".") else f".{preferred_suffix}"

    safe_name = Path(filename or "uploaded_file").name
    tmp_dir = Path(tempfile.mkdtemp(prefix="notebook_upload_"))
    target = tmp_dir / (safe_name if safe_name else f"uploaded{suffix}")
    if suffix and not target.suffix:
        target = target.with_suffix(suffix)

    target.write_bytes(content)
    return str(target)


async def pick_file_via_widget_async(
    accept: str = "",
    description: str = "Select file",
    instruction: str = "Upload a file to continue",
    timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Async upload helper: shows an ipywidgets FileUpload and awaits the file.
    Works with VS Code / Jupyter because it yields control to the event loop.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        return None

    status = widgets.HTML(value=instruction)
    uploader = widgets.FileUpload(accept=accept, multiple=False, description=description)
    display(widgets.VBox([status, uploader]))

    start = time.time()
    while True:
        filename, content = _extract_upload(uploader.value)
        if content:
            path = uploaded_value_to_path(uploader.value)
            status.value = f"Selected: {Path(path).name}" if path else "File received"
            return path

        if timeout is not None and (time.time() - start) > timeout:
            status.value = "No file uploaded (timed out)."
            return None

        await asyncio.sleep(0.1)


def _run_coro_allow_nested(coro):
    """
    Run a coroutine even if the current event loop is already running (nest_asyncio).
    Falls back to asyncio.run when possible. Returns None on failure.
    """
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        try:
            return asyncio.run(coro)
        except Exception:
            return None

    if loop.is_running():
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
            return loop.run_until_complete(coro)
        except Exception:
            return None
    try:
        return loop.run_until_complete(coro)
    except Exception:
        try:
            return asyncio.run(coro)
        except Exception:
            return None


def pick_file_via_widget(accept: str = "", description: str = "Select file", instruction: str = "Upload a file to continue", timeout: Optional[float] = None) -> Optional[str]:
    """
    Sync wrapper around the async picker; works in notebooks by allowing nested event loops (via nest_asyncio).
    """
    return _run_coro_allow_nested(
        pick_file_via_widget_async(
            accept=accept,
            description=description,
            instruction=instruction,
            timeout=timeout,
        )
    )


async def ensure_local_file_async(
    lif_path: Any = None,
    prompt_for_file: bool = True,
    accept: str = ".lif",
    description: str = "Select LIF file",
    allow_tk_fallback: bool = True,
) -> str:
    """
    Async variant for notebooks: await an upload widget to get a temp path.
    """
    if lif_path:
        if isinstance(lif_path, (str, os.PathLike)):
            return str(lif_path)
        resolved = uploaded_value_to_path(lif_path, preferred_suffix=accept)
        if resolved:
            return resolved
        raise ValueError("Unsupported lif_path type; pass a file path or FileUpload widget/value.")

    if prompt_for_file:
        path = await pick_file_via_widget_async(accept=accept, description=description)
        if path:
            return path
        if allow_tk_fallback and Tk and filedialog:
            try:
                Tk().withdraw()
                path = filedialog.askopenfilename(title=description, filetypes=[("Files", f"*{accept}" if accept else "*.*")])
                if path:
                    return path
            except Exception:
                pass
        try:
            manual = input(f"Enter a path for {description.lower()} (or leave blank to cancel): ").strip()
        except Exception:
            manual = ""
        if manual:
            return manual
        raise SystemExit("No file selected via upload widget, Tk fallback, or manual input.")

    raise SystemExit("No file path provided and prompting is disabled.")


def ensure_local_file(
    lif_path: Any = None,
    prompt_for_file: bool = True,
    accept: str = ".lif",
    description: str = "Select LIF file",
    allow_tk_fallback: bool = True,
) -> str:
    """
    Resolve a file path in a Jupyter-safe way.

    - If ``lif_path`` is already a string/path, it is returned.
    - If ``lif_path`` is an ipywidgets FileUpload (or its ``.value``), the
      upload is persisted to a temp file and that path is returned.
    - If ``prompt_for_file`` is True and nothing was provided, an ipywidgets
      upload widget is shown (blocking until a selection arrives or timeout).
    """
    if lif_path:
        if isinstance(lif_path, (str, os.PathLike)):
            return str(lif_path)
        resolved = uploaded_value_to_path(lif_path, preferred_suffix=accept)
        if resolved:
            return resolved
        raise ValueError("Unsupported lif_path type; pass a file path or FileUpload widget/value.")

    if prompt_for_file:
        path = None
        if _should_use_widget():
            path = pick_file_via_widget(accept=accept, description=description)
        if not path and allow_tk_fallback and Tk and filedialog:
            try:
                Tk().withdraw()
                path = filedialog.askopenfilename(title=description, filetypes=[("Files", f"*{accept}" if accept else "*.*")])
            except Exception:
                path = None
        if not path:
            try:
                manual = input(f"Enter a path for {description.lower()} (or leave blank to cancel): ").strip()
            except Exception:
                manual = ""
            if manual:
                path = manual
        if path:
            return path
        raise SystemExit("No file selected via upload widget, Tk fallback, or manual input.")

    raise SystemExit("No file path provided and prompting is disabled.")


__all__ = [
    "ensure_local_file",
    "ensure_local_file_async",
    "pick_file_via_widget",
    "pick_file_via_widget_async",
    "uploaded_value_to_path",
]
