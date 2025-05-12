# onetokenpy/__init__.py

"""
onetokenpy - High-level utilities for seamless LLM interaction, context, and tool use.
"""

__version__ = "0.1.0" # Or your current version

# Import key components from core.py to make them available at the top level
from .core import (
    # Clients
    open_router_client,
    anthropic_client,
    # Attachments Handling
    Attachments,
    ensure_attachments_tagged,
    remove_attachments_tag,

    # Chat and One-Shot
    AttachmentsTracker, # Though less likely to be used directly by end-users
    new_chat,
    ask,
    _CtxToolChatClaudette, # Exposing for type hinting or advanced use, can be omitted if not needed
    _CtxToolChatCosette, # Exposing for type hinting or advanced use, can be omitted if not needed

    # Tool Dispatch
    set_tool_namespace,
    contents,
    # call_func is usually not called directly by users, cosette uses it internally
)

from . import attachments_handlers # Ensure handlers are registered
from .helpers import llm_picker
# You might also want to expose specific exceptions if you define custom ones
# from .exceptions import OnetokenpyError

# Define __all__ to control what `from onetokenpy import *` imports
__all__ = [
    "open_router_client",
    "anthropic_client",
    "Attachments",
    "ensure_attachments_tagged",
    "remove_attachments_tag",
    "new_chat",
    "ask",
    "set_tool_namespace",
    "_CtxToolChatClaudette", # Optional: include if users might need to type hint it
    "_CtxToolChatCosette", # Optional: include if users might need to type hint it
    "AttachmentsTracker", # Optional
    "llm_picker",
    "contents",
]
