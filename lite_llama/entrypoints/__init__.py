"""Server entrypoints: the OpenAI-compatible HTTP API.

Kept in its own package because serving is an optional extra. FastAPI and uvicorn
are imported inside :mod:`lite_llama.entrypoints.api_server`, so importing
``lite_llama`` on a machine that only runs offline generation costs nothing and
fails nowhere.
"""

from .protocol import ChatCompletionRequest, CompletionRequest

__all__ = ["ChatCompletionRequest", "CompletionRequest"]
