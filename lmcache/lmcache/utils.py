import hashlib


_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    return _NVTX_COLORS[hash_value % len(_NVTX_COLORS)]


def _identity_decorator(func):
    return func


try:
    from nvtx import annotate
except Exception:
    annotate = None


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    if annotate is None:
        return _identity_decorator(func)
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)
