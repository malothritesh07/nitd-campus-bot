"""LangSmith tracing.

Two rules this module enforces:

1. Tracing must never change or break an answer. If the key is missing, the
   package is absent, or the network blocks api.smith.langchain.com, every
   decorator degrades to a plain pass-through.
2. Non-LLM paths are traced too. Most of this bot never calls a model — the
   interesting failures are "did the metadata filter match", "did the fuzzy name
   lookup pick the wrong person". LangSmith shows those as regular runs.
"""
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

ENABLED = (os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
           and bool((os.getenv("LANGSMITH_API_KEY") or "").strip()))

_status = "disabled"

if ENABLED:

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"]    = "true"
    os.environ.setdefault("LANGCHAIN_API_KEY",  os.environ["LANGSMITH_API_KEY"])
    os.environ.setdefault("LANGCHAIN_PROJECT",  os.getenv("LANGSMITH_PROJECT", "nitd-campus-bot"))
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT",
                                                          "https://api.smith.langchain.com"))
    try:
        from langsmith import traceable as _traceable
        from langsmith import Client as _Client
        _status = "on"
    except Exception as e:
        ENABLED, _status = False, f"import failed: {str(e)[:60]}"

_LS_CLIENT = None
if ENABLED:


    def _probe(c):


        list(c.list_projects(limit=1))

    def _build_client():
        try:
            c = _Client()
            _probe(c)
            return c, "verified"
        except Exception:
            try:
                import requests, urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                s = requests.Session()
                s.verify = False
                c = _Client(session=s)
                _probe(c)
                return c, "unverified (TLS interception)"
            except Exception as e:
                return None, f"unreachable: {str(e)[:70]}"

    _LS_CLIENT, _status = _build_client()
    if _LS_CLIENT is None:
        ENABLED = False

if not ENABLED:
    def _traceable(*d_args, **d_kwargs):
        """No-op stand-in with the same call signature."""
        def wrap(fn): return fn
        return wrap(d_args[0]) if d_args and callable(d_args[0]) else wrap


def trace(name=None, run_type="chain", **meta):
    """Decorator. Falls back to a pass-through if anything about tracing fails."""
    def deco(fn):
        if not ENABLED:
            return fn
        try:
            return _traceable(run_type=run_type, name=name or fn.__name__,
                              metadata=meta or None, client=_LS_CLIENT)(fn)
        except Exception:
            return fn


    if callable(name):
        fn, name = name, None
        return deco(fn)
    return deco


def client():
    """LangSmith client, or None."""
    return _LS_CLIENT


def check() -> dict:
    """Startup probe — reports status without ever raising."""
    if not ENABLED:
        return {"tracing": False, "status": _status}
    try:
        list(_LS_CLIENT.list_projects(limit=1))
        reachable = True
    except Exception as e:
        reachable = f"unreachable: {str(e)[:70]}"
    return {"tracing": True, "status": _status,
            "project": os.environ["LANGCHAIN_PROJECT"],
            "endpoint": os.environ["LANGCHAIN_ENDPOINT"], "reachable": reachable}


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=2))
