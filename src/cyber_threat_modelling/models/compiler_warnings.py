"""Narrow classification of a documented nonnumerical Numba cache warning."""

import re


def is_dynamic_global_cache_warning(message: str, category: str) -> bool:
    """Recognize only Numba's exact dynamic-global disk-cache warning."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", message)
    return category == "NumbaWarning" and re.fullmatch(
        r'Cannot cache compiled function "[^"\r\n]+" as it uses dynamic globals '
        r'\(such as ctypes pointers and large global arrays\)', plain
    ) is not None
