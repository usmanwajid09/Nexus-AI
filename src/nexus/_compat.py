"""Environment workarounds.

`langchain-core` and `langsmith` import `uuid_utils` (a Rust extension) solely
for `uuid7`. On locked-down Windows machines an Application Control policy can
block the extension's DLL, which would make the whole app unimportable. The
stdlib provides `uuid.uuid7` on Python 3.14+, so when the real package fails to
load we pre-register a pure-Python stand-in. On healthy machines the real
package imports and this shim never activates.
"""

import os
import sys
import time
import types
import uuid as _uuid


def _uuid7_fallback() -> _uuid.UUID:
    # RFC 9562 UUIDv7: 48-bit unix-ms timestamp, version/variant bits, random rest.
    ts_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ts_ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand & 0x3FFF_FFFF_FFFF_FFFF
    return _uuid.UUID(int=value)


def install_uuid_utils_shim() -> None:
    if "uuid_utils" in sys.modules:
        return
    try:
        import uuid_utils  # noqa: F401
    except ImportError:
        compat = types.ModuleType("uuid_utils.compat")
        package = types.ModuleType("uuid_utils")
        for name in ("UUID", "uuid1", "uuid3", "uuid4", "uuid5", "NAMESPACE_DNS", "NAMESPACE_URL"):
            setattr(compat, name, getattr(_uuid, name))
            setattr(package, name, getattr(_uuid, name))
        uuid7 = getattr(_uuid, "uuid7", _uuid7_fallback)
        compat.uuid7 = uuid7
        package.uuid7 = uuid7
        package.compat = compat
        sys.modules["uuid_utils"] = package
        sys.modules["uuid_utils.compat"] = compat
