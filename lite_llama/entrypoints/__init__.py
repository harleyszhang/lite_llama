"""Server entrypoints: the OpenAI-compatible HTTP API.

Re-exports the request models :class:`CompletionRequest` and
:class:`ChatCompletionRequest`; the server itself lives in
:mod:`lite_llama.entrypoints.api_server`.

Usage:
    from lite_llama.entrypoints import CompletionRequest
"""

from .protocol import ChatCompletionRequest, CompletionRequest

__all__ = ["ChatCompletionRequest", "CompletionRequest"]
