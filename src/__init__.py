"""
Package init for the RAG QA validator source code.

Compatibility shim: ragas (as of the version pinned in requirements.txt) unconditionally imports ChatVertexAI from langchain_community.chat_models.vertexai at import time, even for projects that never use Google VertexAI. Recent langchain-community releases removed that submodule, which breaks importing ragas entirely for OpenAI-only projects like this one (see README > Limitations for details on this upstream bug).

This shim registers a lightweight stand-in module before ragas is imported, so the import succeeds without VertexAI being installed. ChatVertexAI is never actually instantiated anywhere in this project.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # pragma: no cover - never instantiated
            """Stub - VertexAI is not used in this project."""

        _stub.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _stub