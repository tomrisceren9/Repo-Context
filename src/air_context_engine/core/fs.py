"""File-system helpers for restricted and network-mounted repositories."""

from pathlib import Path


def safe_resolve_path(
    target: str | Path,
    strict: bool = False,
    *,
    allow_fallback: bool = True,
) -> Path:
    """
    Resolve a path and optionally tolerate platform-specific permission failures.

    The fallback is useful for a trusted repository root on restricted Windows
    drives. Callers enforcing a containment boundary must set
    ``allow_fallback=False`` so resolution failures are handled fail-closed.

    Args:
        target: The target path to resolve.
        strict: Raise ``FileNotFoundError`` when the path does not exist.
        allow_fallback: Return a lexical absolute path if canonical resolution is
            unavailable. Disable this for untrusted child paths.

    Returns:
        The canonical path, or a lexical absolute path when fallback is enabled.
    """
    p = Path(target).expanduser()
    try:
        return p.resolve(strict=strict)
    except (OSError, ValueError) as exc:
        if strict and isinstance(exc, FileNotFoundError):
            raise
        if not allow_fallback:
            raise
        abs_path = p.absolute()
        if strict and not abs_path.exists():
            raise FileNotFoundError(f"No such file or directory: '{abs_path}'")
        return abs_path
