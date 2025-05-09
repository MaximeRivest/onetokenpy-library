"""
onetokenpy.core - High-level utilities for seamless LLM interaction, attachments, and tool use.

Author: Maxime Rivest
License: MIT
"""

import os
import re
import uuid
import mimetypes
import warnings
import ast # For ast.literal_eval if needed, though cosette handles it
from pathlib import Path
from typing import Any, Callable, List, Tuple, Dict, Union # Added Union
from copy import deepcopy

import anthropic
import openai
import cosette.core as cosette
import claudette.core as claudette

_DEFAULT_MODEL = "openai/gpt-4.1-nano"


# =================== CLIENTS ===================
def anthropic_client(model: str = "claude-3-5-haiku-20241022", api_key: str | None = None):
    """
    Return a claudette Client for Anthropic.
    Requires ANTHROPIC_API_KEY in environment.

    Args:
        model: The model slug for Anthropic (default: "claude-3-5-haiku-20241022").
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment for OpenRouter access.")
    return claudette.Client(
        model=model, # This sets cosette.Client.model
        cli=anthropic.Anthropic(
            api_key=api_key,
        )
    )

def open_router_client(model: str = _DEFAULT_MODEL, api_key: str | None = None):
    """
    Return a cosette Client routed through OpenRouter.
    Requires OPENROUTER_API_KEY in environment.

    Args:
        model: The model slug for OpenRouter (default: "openai/gpt-4.1-nano").
        api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY environment variable.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment for OpenRouter access.")
    return cosette.Client(
        model=model, # This sets cosette.Client.model
        cli=openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    )
# =================== ATTACHMENTS HANDLING ===================
class Attachments:
    """
    Collects attachments items (text, images, files, URLs, etc.) for LLM input.
    Converts and serializes these into model-ready parts,
    with minimal dependence on external packages.
    """
    HANDLERS: List[Tuple[str | re.Pattern, Callable[[Path | str], dict], int]] = []

    # ------------------------------------------------------------------
    # ctor / add
    # ------------------------------------------------------------------
    def __init__(self, *items: Any):
        self.items: List[dict] = []
        for it in items:
            self.add(it)

    def add(self, item: Any):
        """
        Process *item* with the best handler and store its dict representation.
        Supported items:
        - Another Attachments instance (items are extended)
        - str or Path: Processed by registered handlers or default path/text logic.
        - bytes: Assumed to be raw image bytes.
        - Other types: Converted to string.
        """
        if isinstance(item, Attachments):
            self.items.extend(item.items)
            return

        # ------------------------------------------------------------------
        # main dispatch
        # ------------------------------------------------------------------
        processed: dict
        if isinstance(item, (str, Path)):
            s = str(item)
            p = Path(s)
            handler, h_arg = self._find_handler(s, p)
            if handler:                            # custom handler
                processed = handler(h_arg)
            else:                                 # built-ins / fallback
                processed = self._default_process_path(p, s)
        elif isinstance(item, bytes):              # raw image bytes
            processed = {"type": "image", "content": item,
                         "identifier": f"image_bytes_{id(item)}"}
        else:                                      # fallback plain text
            processed = {"type": "text", "content": str(item),
                         "identifier": f"text_obj_{uuid.uuid4().hex}"}

        processed = ensure_attachments_tagged(processed)
        self.items.append(processed)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _default_process_path(self, p: Path, s: str) -> dict:
        "Very lightweight default processing when no specific handler exists."
        if p.exists() and p.is_file():
            mime_type, _ = mimetypes.guess_type(s)
            mime_type = mime_type or ""
            if mime_type.startswith("image/"):
                return {"type": "image", "content": p.read_bytes(), "identifier": s}
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception as e:
                warnings.warn(f"Error reading file '{s}' as text: {e}", UserWarning)
                txt = f"[Error reading file content: {Path(s).name}]"
            return {"type": "text", "content": txt, "identifier": s}

        return {"type": "text", "content": s, "identifier": f"text_str_{uuid.uuid4().hex}"}

    @classmethod
    def register_handler(cls, pattern: str | re.Pattern, priority: int = 0):
        """
        Decorator to plug custom handlers for attachment items.
        """
        def deco(fn: Callable[[Path | str], dict]):
            actual_pattern: str | re.Pattern
            if isinstance(pattern, str) and not pattern.startswith("."):
                actual_pattern = re.compile(pattern)
            else:
                actual_pattern = pattern
            cls.HANDLERS.append((actual_pattern, fn, priority))
            cls.HANDLERS.sort(key=lambda x: x[2], reverse=True)
            return fn
        return deco

    def _find_handler(self, item_str: str, path: Path) -> Tuple[Callable[[Path | str], dict] | None, Path | str | None]:
        for pat, fn, _prio in self.HANDLERS:
            if isinstance(pat, str):
                if path.suffix.lower() == pat.lower():
                    return fn, path
            elif isinstance(pat, re.Pattern):
                if pat.search(item_str):
                    return fn, item_str
        return None, None

    # ------------------------------------------------------------------
    # CRUD helpers (deprecated)
    # ------------------------------------------------------------------
    def _add_text(self, txt: str, name: str | None = None):
        warnings.warn("`_add_text` is deprecated. Use `Attachments.add(text_content)` instead.", DeprecationWarning, stacklevel=2)
        self.add(txt)

    def _add_image(self, bts: bytes, name: str | None = None):
        warnings.warn("`_add_image` is deprecated. Use `Attachments.add(image_bytes)` instead.", DeprecationWarning, stacklevel=2)
        self.add(bts)

    def remove(self, identifier: str):
        "Remove items matching *identifier* from the attachments."
        self.items = [it for it in self.items if it.get("identifier") != identifier]

    def clear(self):
        "Remove all items from the attachments."
        self.items.clear()

    def show(self):
        "Print a summary of attachments items."
        print("Attachments items:")
        if not self.items:
            print("(empty)")
            return
        for it in self.items:
            content_preview = str(it.get('content', ''))[:50].replace('\n', ' ') + "..." \
                if isinstance(it.get('content'), str) else type(it.get('content')).__name__
            print(f" - Identifier: {it.get('identifier')}")
            print(f"   Type: {it['type']}")
            print(f"   Content preview: {content_preview}")

    # ------------------------------------------------------------------
    # to-model
    # ------------------------------------------------------------------
    def to_model_format(self, model_name: str):
        """
        Iterator yielding each part in the OpenAI content format,
        respecting model capabilities (e.g., text-only models).
        """
        text_only_models = {"o1-preview", "o1-mini", "o3-mini"}
        allow_images = model_name not in text_only_models

        for item in self.items:
            if item["type"] == "image":
                if allow_images:
                    yield item["content"]
                else:
                    yield f"[Image omitted due to model capabilities: {item['identifier']}]"
            elif item["type"] == "text":
                yield item["content"]
            else:
                yield f"[Unsupported Attachments type '{item['type']}': {item['identifier']}]"


def ensure_attachments_tagged(item: dict) -> dict:
    """
    Ensures a attachments item has an 'identifier' and, if it's text,
    is wrapped in <attachments id="...">...</attachments> tags unless already present.
    """
    if not isinstance(item, dict) or "type" not in item or "content" not in item:
        return item

    item.setdefault("identifier", f"attach_item_{uuid.uuid4().hex}")
    identifier = str(item["identifier"])

    if item["type"] == "text" and isinstance(item["content"], str):
        content = item["content"]
        tag_pattern = re.compile(
            fr"(?s)<attachments\s+id=['\"]?{re.escape(identifier)}['\"]?>.*?</attachments>"
        )
        if not tag_pattern.search(content):
            item["content"] = f"<attachments id='{identifier}'>{content}</attachments>"
    return item


def remove_attachments_tag(text: str, attachments_id: str | None = None) -> str:
    """
    Remove either a specific or all <attachments> blocks from a text string.
    Also normalizes multiple newlines to a maximum of two.
    """
    if not isinstance(text, str):
        return text

    pattern: re.Pattern
    if attachments_id:
        pattern = re.compile(
            fr"<attachments\s+id=['\"]?{re.escape(str(attachments_id))}['\"]?>.*?</attachments>",
            re.DOTALL | re.IGNORECASE,
        )
    else:
        pattern = re.compile(r"<attachments\s+id=['\"][^'\"]*['\"]?>.*?</attachments>", re.DOTALL | re.IGNORECASE)

    cleaned_text = pattern.sub("", text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


# =================== CHAT and ONE-SHOT ===================

class AttachmentsTracker:
    """Tracks which attachments items (by identifier) have been sent to the model."""
    def __init__(self):
        self.sent_ids: set[str] = set()

    def mark_sent(self, item_id: str | None):
        if item_id:
            self.sent_ids.add(item_id)

    def is_sent(self, item_id: str | None) -> bool:
        return item_id is not None and item_id in self.sent_ids

    def clear(self):
        self.sent_ids.clear()

_CURRENT_CHAT_TOOL_NAMESPACE: Dict[str, Callable] | None = None

def new_chat(
    model: str = _DEFAULT_MODEL,
    sp: str | None = None,
    attach: Attachments | str | Path | list | None = None,
    tools: list | None = None,
    auto_toolloop: bool = True, 
    max_tool_steps: int = 10,
    trace_tool_calls: bool = False,
    api_key: str | None = None,
    cli: Union[cosette.Client, claudette.Client] | None = None,
    **chat_kwargs
):
    """
    Create a chat object tightly integrating attachments and cosette or claudette style tool use.
    Args:
        model: The model to use for the chat.
        sp: The system prompt to use for the chat.
        attach: The attachments to use for the chat.
        tools: The tools to use for the chat.
        auto_toolloop: Whether to automatically loop through tools.
        max_tool_steps: The maximum number of tool steps to use.
        trace_tool_calls: Whether to trace tool calls.
        api_key: The API key to use for the chat.
        cli: The client to use for the chat.
        **chat_kwargs: Additional keyword arguments to pass to the chat.
    Returns:
        A chat object.
    """
    was_custom_cli_provided_to_new_chat = False # Flag to track if cli was explicit
    if cli is None:
        actual_cli = open_router_client(model, api_key)
    else:
        actual_cli = cli
        was_custom_cli_provided_to_new_chat = True

    initial_attach: Attachments
    if attach is None:
        initial_attach = Attachments()
    elif isinstance(attach, Attachments):
        initial_attach = attach
    elif isinstance(attach, (str, Path)):
        initial_attach = Attachments(attach)
    elif isinstance(attach, list):
        initial_attach = Attachments(*attach)
    else:
        warnings.warn(f"Unexpected attachments type: {type(attach)}. Initializing empty attachments.", UserWarning)
        initial_attach = Attachments()

    onetoken_chat_params = {
        'auto_toolloop': auto_toolloop,
        'max_tool_steps': max_tool_steps,
        'trace_tool_calls': trace_tool_calls,
        'api_key': api_key, 
        '_custom_cli_passed_to_new_chat': was_custom_cli_provided_to_new_chat,
    }

    final_init_kwargs = {**onetoken_chat_params, **chat_kwargs}

    attachprocessed_tools = []
    tool_function_map: Dict[str, Callable] = {}
    if tools:
        for tool_item in tools:
            if callable(tool_item):
                tool_function_map[tool_item.__name__] = tool_item
                attachprocessed_tools.append(tool_item)
            elif isinstance(tool_item, dict) and tool_item.get("type") == "function" and "function" in tool_item:
                attachprocessed_tools.append(tool_item)
                warnings.warn(f"Tool '{tool_item['function'].get('name', 'Unnamed Dict Tool')}' is a dict. "
                              "onetokenpy's automatic tool resolution works best with callable functions. "
                              "Ensure this tool can be found via set_tool_namespace or globals if not a direct callable.", UserWarning)

            else:
                attachprocessed_tools.append(tool_item)

    if isinstance(actual_cli, claudette.Client):
        return _CtxToolChatClaudette(
            cli=actual_cli,
            model=model,
            attach=initial_attach,
            sp=sp or "You are a helpful and concise assistant.",
            tools=attachprocessed_tools, 
            tool_function_map=tool_function_map, 
            **final_init_kwargs
        )
    else:
        return _CtxToolChatCosette(
            cli=actual_cli,
            model=model,
            attach=initial_attach,
            sp=sp or "You are a helpful and concise assistant.",
            tools=attachprocessed_tools, 
            tool_function_map=tool_function_map, 
            **final_init_kwargs
        )

class _CtxToolChatCosette(cosette.Chat):
    """
    Enhanced chat object for onetokenpy.
    (Docstring details match previous version)
    """
    def __init__(self, *args,
                 model: str,
                 cli: cosette.Client,
                 attach: Attachments,
                 api_key: str | None = None,
                 auto_toolloop: bool = True,
                 max_tool_steps: int = 10,
                 trace_tool_calls: bool = False,
                 tool_function_map: Dict[str, Callable] | None = None, # MODIFICATION 3
                 # kwargs here include `tools`, `sp` from new_chat, and passthroughs like `temperature`
                 **kwargs):

        self._attach: Attachments = attach
        self._tracker: AttachmentsTracker = AttachmentsTracker()
        self._in_toolloop: bool = False

        self._api_key: str | None = api_key
        # Retrieve and remove _custom_cli_passed_to_new_chat before it goes to super()
        self._custom_cli_provided: bool = kwargs.pop('_custom_cli_passed_to_new_chat', False)

        self._auto_toolloop: bool = auto_toolloop
        self._max_tool_steps: int = max_tool_steps
        self._trace_tool_calls: bool = trace_tool_calls

        self._cli: cosette.Client = cli # This is the cosette.Client instance
        self._model: str = model # Authoritative model name for this chat instance
        self._model_explicitly_set_on_chat: bool = True

        # --- MODIFICATION 4: Store the tool function map ---
        self._tool_function_map: Dict[str, Callable] = tool_function_map or {}

        # `cli` is the cosette.Client. `model` is the string name.
        # `kwargs` here will include `tools` and `sp` for cosette.Chat's __init__.
        super().__init__(*args, cli=self._cli, model=self._model, **kwargs)
        # After super().__init__, self.c (cosette.Chat's client) is self._cli.
        # And cosette.Chat.model is self._model.

        if self._custom_cli_provided: # Check if new_chat was given an explicit cli
             # This check is more relevant if the provided cli's model might differ
             # from the 'model' argument passed to new_chat.
            self._check_model_cli_mismatch(verbose_initial_check=True)
        else: # If cli was created by open_router_client, it should match `model`
            if self._cli.model != self._model:
                 warnings.warn(f"[onetokenpy] Internal Inconsistency: _CtxToolChat model '{self._model}' "
                               f"differs from created client's model '{self._cli.model}'. "
                               "This should not happen. Synchronizing client to chat model.", UserWarning)
                 self.sync_cli_to_model() # Attempt to fix

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, new_model_name: str):
        if new_model_name == self._model:
            return
        
        old_model = self._model
        self._model = new_model_name
        self._model_explicitly_set_on_chat = True

        if not self._custom_cli_provided:
            try:
                self._cli = open_router_client(new_model_name, self._api_key)
                self.c = self._cli # Update client used by cosette.Chat internals
            except Exception as e:
                self._model = old_model # Revert on failure
                warnings.warn(f"Failed to switch to model '{new_model_name}': {e}. "
                              "Reverted to previous model '{old_model}'.", UserWarning)
        else:
            self._check_model_cli_mismatch(warn_on_mismatch=True)

    @property
    def cli(self) -> cosette.Client: # This is cosette.Client
        return self._cli

    @cli.setter
    def cli(self, new_client: cosette.Client): # new_client is cosette.Client
        self._cli = new_client
        self.c = new_client # Update client used by cosette.Chat internals
        self._custom_cli_provided = True

        client_model_attr = getattr(new_client, "model", None)
        if client_model_attr and isinstance(client_model_attr, str):
            if not self._model_explicitly_set_on_chat or self._model == _DEFAULT_MODEL:
                self._model = client_model_attr
                super().model = client_model_attr # Keep cosette.Chat.model in sync
                self._model_explicitly_set_on_chat = False
        self._check_model_cli_mismatch(warn_on_mismatch=True)


    def _check_model_cli_mismatch(self, verbose_initial_check: bool = False, warn_on_mismatch: bool = False):
        # self._cli is cosette.Client. Its model is self._cli.model.
        # self._model is _CtxToolChat.model.
        cli_model_name = getattr(self._cli, "model", None)

        if cli_model_name is None:
            # This case should be rare if using open_router_client or standard cosette.Client
            if verbose_initial_check or warn_on_mismatch:
                warnings.warn(
                    f"[onetokenpy] The chat's client (cosette.Client) does not have a `.model` attribute. "
                    f"Cannot verify its compatibility with chat.model ('{self._model}').",
                    UserWarning
                )
            return

        if cli_model_name != self._model:
            # This means _CtxToolChat.model differs from cosette.Client.model
            msg = (f"[onetokenpy] MODEL-CLIENT MISMATCH: _CtxToolChat.model is '{self._model}' but "
                   f"the underlying cosette.Client's model is '{cli_model_name}'.\n"
                   f"- _CtxToolChat.model ('{self._model}') dictates attachments processing and is intended model.\n"
                   f"- cosette.Client.model ('{cli_model_name}') is what will be used in API calls by cosette.\n"
                   "To resolve, consider:\n"
                   "  chat.sync_model_to_cli()  # Sets _CtxToolChat.model = cosette.Client.model\n"
                   "  chat.sync_cli_to_model()  # Sets cosette.Client.model = _CtxToolChat.model\n"
                   "  chat.model = 'new_model_name' # Sets _CtxToolChat.model and updates client if not custom.\n"
                   "  chat.cli = new_cosette_client # Sets new client, may update _CtxToolChat.model.")
            if warn_on_mismatch:
                warnings.warn(msg, UserWarning)
            elif verbose_initial_check:
                print(msg)

    def sync_cli_to_model(self):
        """Sets the `cosette.Client.model` attribute to match `_CtxToolChat.model`."""
        if not hasattr(self._cli, "model"):
            warnings.warn("[onetokenpy] Client (cosette.Client) has no `.model` attribute to synchronize.", UserWarning)
            return

        try:
            # cosette.Client.model is a simple attribute, not a property usually
            setattr(self._cli, "model", self._model)
            if getattr(self._cli, "model") == self._model:
                print(f"[onetokenpy] Underlying client (cosette.Client) model synchronized to: '{self._model}'")
            else: # Should not happen if setattr works
                warnings.warn(f"[onetokenpy] Attempted to set client.model to '{self._model}', "
                              f"but it remained '{getattr(self._cli, 'model', 'N/A')}'.", UserWarning)
        except Exception as e: # More general exception
            warnings.warn(f"[onetokenpy] Error synchronizing client.model to chat.model: {e}", UserWarning)
        self._check_model_cli_mismatch(warn_on_mismatch=True)


    def sync_model_to_cli(self):
        """Sets `_CtxToolChat.model` to match the `cosette.Client.model` attribute."""
        cli_model_name = getattr(self._cli, "model", None)
        if cli_model_name is None:
            warnings.warn("[onetokenpy] Client (cosette.Client) has no `.model` attribute to read from.", UserWarning)
            return
        if not isinstance(cli_model_name, str):
            warnings.warn(f"[onetokenpy] Client's `.model` attribute is not a string ('{cli_model_name}'). "
                          "Cannot sync chat.model.", UserWarning)
            return

        self.model = cli_model_name # Use the setter for _CtxToolChat.model
        self._model_explicitly_set_on_chat = False
        print(f"[onetokenpy] Chat model synchronized to client's model: '{self._model}'")
        self._check_model_cli_mismatch()


    def __call__(self,
                 message: Any | None = None,
                 attach: Attachments | str | Path | list | None = None,
                 merge_attachments : bool = True,
                 **kwargs): # kwargs for the API call (temperature, max_tokens, etc.)
        global _CURRENT_CHAT_TOOL_NAMESPACE # MODIFICATION 5: Set global before calling super
        _CURRENT_CHAT_TOOL_NAMESPACE = self._tool_function_map

        try:
            self._check_model_cli_mismatch(warn_on_mismatch=True)
            effective_attach = self._attach
            if attach:
                new_attach = attach if isinstance(attach, Attachments) else Attachments(*(attach if isinstance(attach, list) else [attach]))
                if merge_attachments:
                    combined_items = list(self._attach.items); combined_items.extend(new_attach.items)
                    effective_attach = Attachments(*combined_items)
                else: effective_attach = new_attach

            prompt_parts = []
            text_only_models = {"o1-preview", "o1-mini", "o3-mini"}
            allow_images = self.model not in text_only_models

            for item_dict in effective_attach.items:
                item_id = item_dict.get('identifier')
                if self._tracker.is_sent(item_id) and effective_attach is self._attach : continue
                if item_dict["type"] == "image":
                    if allow_images: prompt_parts.append(item_dict["content"])
                    else: prompt_parts.append(f"[Image omitted: {item_dict.get('identifier')}]")
                elif item_dict["type"] == "text": prompt_parts.append(item_dict["content"])
                else: prompt_parts.append(f"[Unsupported: {item_dict.get('identifier')}]")
                if effective_attach is self._attach or not self._tracker.is_sent(item_id):
                    self._tracker.mark_sent(item_id)

            if message is not None: prompt_parts.append(str(message))
            completion_kwargs = {k: v for k, v in kwargs.items() if k not in ("model", "tools")}
            
            result = super().__call__(prompt_parts, **completion_kwargs)

            # Cosette and Claudette has 2 slightly different ways of returning the stop reason, we handle that here.
            # Check if result has 'choices' and it's a non-empty list
            if hasattr(result, "choices") and isinstance(result.choices, list) and result.choices:
                ch = result.choices[0]
            else:
                ch = None

            if (self._auto_toolloop and self.tools):
                if ch is not None:
                    if getattr(ch, "finish_reason", None) == "tool_calls" and not self._in_toolloop:
                        self._in_toolloop = True
                        try:
                            trace_fn = print if self._trace_tool_calls else None
                            result = self.toolloop(None, max_steps=self._max_tool_steps, trace_func=trace_fn, **completion_kwargs)
                        finally: self._in_toolloop = False
                elif getattr(result, "stop_reason", None) == "tool_use" and not self._in_toolloop:
                    self._in_toolloop = True
                    try:
                        trace_fn = print if self._trace_tool_calls else None
                        result = self.toolloop(None, max_steps=self._max_tool_steps, trace_func=trace_fn, **completion_kwargs)
                    finally: self._in_toolloop = False
            return result
        finally:
            _CURRENT_CHAT_TOOL_NAMESPACE = None # MODIFICATION 6: Clear global after call

    def add_attachments(self, item: Any):
        self._attach.add(item)

    def remove_attachments(self, identifier: str):
        self._attach.remove(identifier)
        self._tracker.sent_ids.discard(identifier)
        for msg_idx, msg_obj in enumerate(self.h):
            if isinstance(msg_obj, dict) and 'content' in msg_obj:
                content_val = msg_obj['content']
                if isinstance(content_val, str):
                    msg_obj['content'] = remove_attachments_tag(content_val, identifier)
                elif isinstance(content_val, list):
                    new_content_list = []
                    changed = False
                    for part in content_val:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            original_text = part.get('text', '')
                            cleaned_text = remove_attachments_tag(original_text, identifier)
                            if cleaned_text != original_text: changed = True
                            if cleaned_text or part.get('type') != 'text':
                                new_content_list.append({**part, 'text': cleaned_text})
                        else:
                            new_content_list.append(part)
                    if changed: msg_obj['content'] = new_content_list
            self.h[msg_idx] = msg_obj


    def clear_attachments(self):
        self._attach.clear()
        self._tracker.clear()
        for msg_idx, msg_obj in enumerate(self.h):
            if isinstance(msg_obj, dict) and 'content' in msg_obj:
                content_val = msg_obj['content']
                if isinstance(content_val, str):
                    msg_obj['content'] = remove_attachments_tag(content_val)
                elif isinstance(content_val, list):
                    new_content_list = []
                    changed = False
                    for part in content_val:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            original_text = part.get('text', '')
                            cleaned_text = remove_attachments_tag(original_text)
                            if cleaned_text != original_text: changed = True
                            if cleaned_text or part.get('type') != 'text':
                                new_content_list.append({**part, 'text': cleaned_text})
                        else:
                            new_content_list.append(part)
                    if changed: msg_obj['content'] = new_content_list
            self.h[msg_idx] = msg_obj

    def show_attachments(self):
        self._attach.show()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        # --- Client Recreation ---
        new_cli: Union[cosette.Client, claudette.Client] # Type hint for cosette.core.Client
        if self._custom_cli_provided:
            try:
                new_cli = deepcopy(self._cli, memo)
            except TypeError as e:
                if "RLock" in str(e) or "pickle" in str(e) or "_thread.RLock" in str(e):
                    warnings.warn(
                        f"Deepcopying a Chat object with a custom client ('{type(self._cli).__name__}') that "
                        f"is not fully deepcopyable. The copied chat will receive a new default "
                        f"OpenRouter client configured for the model '{self.model}'. "
                        f"The original custom client's specific state or type is not preserved "
                        f"if it couldn't be deepcopied.",
                        UserWarning
                    )
                    new_cli = open_router_client(self.model, self._api_key)
                else:
                    raise
        else:
            new_cli = open_router_client(self.model, self._api_key)

        # --- Deepcopy other _CtxToolChat specific mutable attributes ---
        new_attach = deepcopy(self._attach, memo)
        new_tracker_sent_ids = deepcopy(self._tracker.sent_ids, memo)
        new_tool_function_map = deepcopy(self._tool_function_map, memo)

        # --- Prepare arguments for _CtxToolChat's __init__ and its super().__init__ ---
        # Includes: tools schema, sp, history, and other API params (e.g., temperature)

        copied_super_tools = deepcopy(self.tools, memo) # self.tools from cosette.Chat
        copied_super_sp = self.sp # self.sp from cosette.Chat (string, immutable)
        copied_history = deepcopy(self.h, memo) # self.h from cosette.Chat
        
        # self.kwargs from cosette.Chat holds other passthrough API parameters
        # copied_super_api_kwargs = deepcopy(self.kwargs, memo) # Old line causing AttributeError
        # API-specific kwargs (like temperature) are typically stored in the client's kwargs
        copied_cli_api_kwargs = deepcopy(getattr(self._cli, 'kwargs', {}), memo)

        # These are kwargs for _CtxToolChat's __init__, which will be passed to
        # _CtxToolChat's own logic and also to super().__init__ (cosette.Chat).
        init_kwargs_for_new_chat_obj = {
            **copied_cli_api_kwargs, # Use kwargs from the client
            'tools': copied_super_tools,
            'sp': copied_super_sp,
            '_custom_cli_passed_to_new_chat': self._custom_cli_provided,
        }

        # --- Create the new _CtxToolChat instance ---
        new_chat_obj = self.__class__(
            cli=new_cli,
            model=self.model, # Uses self._model
            attach=new_attach,
            api_key=self._api_key,
            auto_toolloop=self._auto_toolloop,
            max_tool_steps=self._max_tool_steps,
            trace_tool_calls=self._trace_tool_calls,
            tool_function_map=new_tool_function_map,
            **init_kwargs_for_new_chat_obj
        )
        memo[id(self)] = new_chat_obj

        # --- Set attributes not fully handled by __init__ or needing post-init restoration ---
        new_chat_obj.h = copied_history # Set history after object creation
        new_chat_obj._tracker.sent_ids = new_tracker_sent_ids # __init__ creates a new tracker
        new_chat_obj._in_toolloop = self._in_toolloop # Copy current toolloop state

        return new_chat_obj



class _CtxToolChatClaudette(claudette.Chat):
    """
    Enhanced chat object for onetokenpy.
    (Docstring details match previous version)
    """
    def __init__(self, *args,
                 model: str,
                 cli: claudette.Client,
                 attach: Attachments,
                 api_key: str | None = None,
                 auto_toolloop: bool = True,
                 max_tool_steps: int = 10,
                 trace_tool_calls: bool = False,
                 tool_function_map: Dict[str, Callable] | None = None, # MODIFICATION 3
                 # kwargs here include `tools`, `sp` from new_chat, and passthroughs like `temperature`
                 **kwargs):

        self._attach: Attachments = attach
        self._tracker: AttachmentsTracker = AttachmentsTracker()
        self._in_toolloop: bool = False

        self._api_key: str | None = api_key
        # Retrieve and remove _custom_cli_passed_to_new_chat before it goes to super()
        self._custom_cli_provided: bool = kwargs.pop('_custom_cli_passed_to_new_chat', False)

        self._auto_toolloop: bool = auto_toolloop
        self._max_tool_steps: int = max_tool_steps
        self._trace_tool_calls: bool = trace_tool_calls

        self._cli: claudette.Client = cli # This is the cosette.Client instance
        self._model: str = model # Authoritative model name for this chat instance
        self._model_explicitly_set_on_chat: bool = True

        # --- MODIFICATION 4: Store the tool function map ---
        self._tool_function_map: Dict[str, Callable] = tool_function_map or {}

        # `cli` is the cosette.Client. `model` is the string name.
        # `kwargs` here will include `tools` and `sp` for cosette.Chat's __init__.
        super().__init__(*args, cli=self._cli, model=self._model, **kwargs)
        # After super().__init__, self.c (cosette.Chat's client) is self._cli.
        # And cosette.Chat.model is self._model.

        if self._custom_cli_provided: # Check if new_chat was given an explicit cli
             # This check is more relevant if the provided cli's model might differ
             # from the 'model' argument passed to new_chat.
            self._check_model_cli_mismatch(verbose_initial_check=True)
        else: # If cli was created by open_router_client, it should match `model`
            if self._cli.model != self._model:
                 warnings.warn(f"[onetokenpy] Internal Inconsistency: _CtxToolChat model '{self._model}' "
                               f"differs from created client's model '{self._cli.model}'. "
                               "This should not happen. Synchronizing client to chat model.", UserWarning)
                 self.sync_cli_to_model() # Attempt to fix

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, new_model_name: str):
        if new_model_name == self._model:
            return
        
        old_model = self._model
        self._model = new_model_name
        self._model_explicitly_set_on_chat = True

        if not self._custom_cli_provided:
            try:
                self._cli = open_router_client(new_model_name, self._api_key)
                self.c = self._cli # Update client used by cosette.Chat internals
            except Exception as e:
                self._model = old_model # Revert on failure
                warnings.warn(f"Failed to switch to model '{new_model_name}': {e}. "
                              "Reverted to previous model '{old_model}'.", UserWarning)
        else:
            self._check_model_cli_mismatch(warn_on_mismatch=True)

    @property
    def cli(self) -> claudette.Client: # This is cosette.Client
        return self._cli

    @cli.setter
    def cli(self, new_client: claudette.Client): # new_client is cosette.Client
        self._cli = new_client
        self.c = new_client # Update client used by cosette.Chat internals
        self._custom_cli_provided = True

        client_model_attr = getattr(new_client, "model", None)
        if client_model_attr and isinstance(client_model_attr, str):
            if not self._model_explicitly_set_on_chat or self._model == _DEFAULT_MODEL:
                self._model = client_model_attr
                super().model = client_model_attr # Keep cosette.Chat.model in sync
                self._model_explicitly_set_on_chat = False
        self._check_model_cli_mismatch(warn_on_mismatch=True)


    def _check_model_cli_mismatch(self, verbose_initial_check: bool = False, warn_on_mismatch: bool = False):
        # self._cli is cosette.Client. Its model is self._cli.model.
        # self._model is _CtxToolChat.model.
        cli_model_name = getattr(self._cli, "model", None)

        if cli_model_name is None:
            # This case should be rare if using open_router_client or standard cosette.Client
            if verbose_initial_check or warn_on_mismatch:
                warnings.warn(
                    f"[onetokenpy] The chat's client (cosette.Client) does not have a `.model` attribute. "
                    f"Cannot verify its compatibility with chat.model ('{self._model}').",
                    UserWarning
                )
            return

        if cli_model_name != self._model:
            # This means _CtxToolChat.model differs from cosette.Client.model
            msg = (f"[onetokenpy] MODEL-CLIENT MISMATCH: _CtxToolChat.model is '{self._model}' but "
                   f"the underlying cosette.Client's model is '{cli_model_name}'.\n"
                   f"- _CtxToolChat.model ('{self._model}') dictates attachments processing and is intended model.\n"
                   f"- cosette.Client.model ('{cli_model_name}') is what will be used in API calls by cosette.\n"
                   "To resolve, consider:\n"
                   "  chat.sync_model_to_cli()  # Sets _CtxToolChat.model = cosette.Client.model\n"
                   "  chat.sync_cli_to_model()  # Sets cosette.Client.model = _CtxToolChat.model\n"
                   "  chat.model = 'new_model_name' # Sets _CtxToolChat.model and updates client if not custom.\n"
                   "  chat.cli = new_cosette_client # Sets new client, may update _CtxToolChat.model.")
            if warn_on_mismatch:
                warnings.warn(msg, UserWarning)
            elif verbose_initial_check:
                print(msg)

    def sync_cli_to_model(self):
        """Sets the `cosette.Client.model` attribute to match `_CtxToolChat.model`."""
        if not hasattr(self._cli, "model"):
            warnings.warn("[onetokenpy] Client (cosette.Client) has no `.model` attribute to synchronize.", UserWarning)
            return

        try:
            # cosette.Client.model is a simple attribute, not a property usually
            setattr(self._cli, "model", self._model)
            if getattr(self._cli, "model") == self._model:
                print(f"[onetokenpy] Underlying client (cosette.Client) model synchronized to: '{self._model}'")
            else: # Should not happen if setattr works
                warnings.warn(f"[onetokenpy] Attempted to set client.model to '{self._model}', "
                              f"but it remained '{getattr(self._cli, 'model', 'N/A')}'.", UserWarning)
        except Exception as e: # More general exception
            warnings.warn(f"[onetokenpy] Error synchronizing client.model to chat.model: {e}", UserWarning)
        self._check_model_cli_mismatch(warn_on_mismatch=True)


    def sync_model_to_cli(self):
        """Sets `_CtxToolChat.model` to match the `cosette.Client.model` attribute."""
        cli_model_name = getattr(self._cli, "model", None)
        if cli_model_name is None:
            warnings.warn("[onetokenpy] Client (cosette.Client) has no `.model` attribute to read from.", UserWarning)
            return
        if not isinstance(cli_model_name, str):
            warnings.warn(f"[onetokenpy] Client's `.model` attribute is not a string ('{cli_model_name}'). "
                          "Cannot sync chat.model.", UserWarning)
            return

        self.model = cli_model_name # Use the setter for _CtxToolChat.model
        self._model_explicitly_set_on_chat = False
        print(f"[onetokenpy] Chat model synchronized to client's model: '{self._model}'")
        self._check_model_cli_mismatch()


    def __call__(self,
                 message: Any | None = None,
                 attach: Attachments | str | Path | list | None = None,
                 merge_attachments : bool = True,
                 **kwargs): # kwargs for the API call (temperature, max_tokens, etc.)
        global _CURRENT_CHAT_TOOL_NAMESPACE # MODIFICATION 5: Set global before calling super
        _CURRENT_CHAT_TOOL_NAMESPACE = self._tool_function_map

        try:
            self._check_model_cli_mismatch(warn_on_mismatch=True)
            effective_attach = self._attach
            if attach:
                new_attach = attach if isinstance(attach, Attachments) else Attachments(*(attach if isinstance(attach, list) else [attach]))
                if merge_attachments:
                    combined_items = list(self._attach.items); combined_items.extend(new_attach.items)
                    effective_attach = Attachments(*combined_items)
                else: effective_attach = new_attach

            prompt_parts = []
            text_only_models = {"o1-preview", "o1-mini", "o3-mini"}
            allow_images = self.model not in text_only_models

            for item_dict in effective_attach.items:
                item_id = item_dict.get('identifier')
                if self._tracker.is_sent(item_id) and effective_attach is self._attach : continue
                if item_dict["type"] == "image":
                    if allow_images: prompt_parts.append(item_dict["content"])
                    else: prompt_parts.append(f"[Image omitted: {item_dict.get('identifier')}]")
                elif item_dict["type"] == "text": prompt_parts.append(item_dict["content"])
                else: prompt_parts.append(f"[Unsupported: {item_dict.get('identifier')}]")
                if effective_attach is self._attach or not self._tracker.is_sent(item_id):
                    self._tracker.mark_sent(item_id)

            if message is not None: prompt_parts.append(str(message))
            completion_kwargs = {k: v for k, v in kwargs.items() if k not in ("model", "tools")}
            
            result = super().__call__(prompt_parts, **completion_kwargs)

            # Cosette and Claudette has 2 slightly different ways of returning the stop reason, we handle that here.
            # Check if result has 'choices' and it's a non-empty list
            if hasattr(result, "choices") and isinstance(result.choices, list) and result.choices:
                ch = result.choices[0]
            else:
                ch = None

            if (self._auto_toolloop and self.tools):
                if ch is not None:
                    if getattr(ch, "finish_reason", None) == "tool_calls" and not self._in_toolloop:
                        self._in_toolloop = True
                        try:
                            trace_fn = print if self._trace_tool_calls else None
                            result = self.toolloop(None, max_steps=self._max_tool_steps, trace_func=trace_fn, **completion_kwargs)
                        finally: self._in_toolloop = False
                elif getattr(result, "stop_reason", None) == "tool_use" and not self._in_toolloop:
                    self._in_toolloop = True
                    try:
                        trace_fn = print if self._trace_tool_calls else None
                        result = self.toolloop(None, max_steps=self._max_tool_steps, trace_func=trace_fn, **completion_kwargs)
                    finally: self._in_toolloop = False
            return result
        finally:
            _CURRENT_CHAT_TOOL_NAMESPACE = None # MODIFICATION 6: Clear global after call

    def add_attachments(self, item: Any):
        self._attach.add(item)

    def remove_attachments(self, identifier: str):
        self._attach.remove(identifier)
        self._tracker.sent_ids.discard(identifier)
        for msg_idx, msg_obj in enumerate(self.h):
            if isinstance(msg_obj, dict) and 'content' in msg_obj:
                content_val = msg_obj['content']
                if isinstance(content_val, str):
                    msg_obj['content'] = remove_attachments_tag(content_val, identifier)
                elif isinstance(content_val, list):
                    new_content_list = []
                    changed = False
                    for part in content_val:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            original_text = part.get('text', '')
                            cleaned_text = remove_attachments_tag(original_text, identifier)
                            if cleaned_text != original_text: changed = True
                            if cleaned_text or part.get('type') != 'text':
                                new_content_list.append({**part, 'text': cleaned_text})
                        else:
                            new_content_list.append(part)
                    if changed: msg_obj['content'] = new_content_list
            self.h[msg_idx] = msg_obj


    def clear_attachments(self):
        self._attach.clear()
        self._tracker.clear()
        for msg_idx, msg_obj in enumerate(self.h):
            if isinstance(msg_obj, dict) and 'content' in msg_obj:
                content_val = msg_obj['content']
                if isinstance(content_val, str):
                    msg_obj['content'] = remove_attachments_tag(content_val)
                elif isinstance(content_val, list):
                    new_content_list = []
                    changed = False
                    for part in content_val:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            original_text = part.get('text', '')
                            cleaned_text = remove_attachments_tag(original_text)
                            if cleaned_text != original_text: changed = True
                            if cleaned_text or part.get('type') != 'text':
                                new_content_list.append({**part, 'text': cleaned_text})
                        else:
                            new_content_list.append(part)
                    if changed: msg_obj['content'] = new_content_list
            self.h[msg_idx] = msg_obj

    def show_attachments(self):
        self._attach.show()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        # --- Client Recreation ---
        new_cli: Union[cosette.Client, claudette.Client] # Type hint for cosette.core.Client
        if self._custom_cli_provided:
            try:
                new_cli = deepcopy(self._cli, memo)
            except TypeError as e:
                if "RLock" in str(e) or "pickle" in str(e) or "_thread.RLock" in str(e):
                    warnings.warn(
                        f"Deepcopying a Chat object with a custom client ('{type(self._cli).__name__}') that "
                        f"is not fully deepcopyable. The copied chat will receive a new default "
                        f"OpenRouter client configured for the model '{self.model}'. "
                        f"The original custom client's specific state or type is not preserved "
                        f"if it couldn't be deepcopied.",
                        UserWarning
                    )
                    new_cli = open_router_client(self.model, self._api_key)
                else:
                    raise
        else:
            new_cli = open_router_client(self.model, self._api_key)

        # --- Deepcopy other _CtxToolChat specific mutable attributes ---
        new_attach = deepcopy(self._attach, memo)
        new_tracker_sent_ids = deepcopy(self._tracker.sent_ids, memo)
        new_tool_function_map = deepcopy(self._tool_function_map, memo)

        # --- Prepare arguments for _CtxToolChat's __init__ and its super().__init__ ---
        # Includes: tools schema, sp, history, and other API params (e.g., temperature)

        copied_super_tools = deepcopy(self.tools, memo) # self.tools from cosette.Chat
        copied_super_sp = self.sp # self.sp from cosette.Chat (string, immutable)
        copied_history = deepcopy(self.h, memo) # self.h from cosette.Chat
        
        # self.kwargs from cosette.Chat holds other passthrough API parameters
        # copied_super_api_kwargs = deepcopy(self.kwargs, memo) # Old line causing AttributeError
        # API-specific kwargs (like temperature) are typically stored in the client's kwargs
        copied_cli_api_kwargs = deepcopy(getattr(self._cli, 'kwargs', {}), memo)

        # These are kwargs for _CtxToolChat's __init__, which will be passed to
        # _CtxToolChat's own logic and also to super().__init__ (cosette.Chat).
        init_kwargs_for_new_chat_obj = {
            **copied_cli_api_kwargs, # Use kwargs from the client
            'tools': copied_super_tools,
            'sp': copied_super_sp,
            '_custom_cli_passed_to_new_chat': self._custom_cli_provided,
        }

        # --- Create the new _CtxToolChat instance ---
        new_chat_obj = self.__class__(
            cli=new_cli,
            model=self.model, # Uses self._model
            attach=new_attach,
            api_key=self._api_key,
            auto_toolloop=self._auto_toolloop,
            max_tool_steps=self._max_tool_steps,
            trace_tool_calls=self._trace_tool_calls,
            tool_function_map=new_tool_function_map,
            **init_kwargs_for_new_chat_obj
        )
        memo[id(self)] = new_chat_obj

        # --- Set attributes not fully handled by __init__ or needing post-init restoration ---
        new_chat_obj.h = copied_history # Set history after object creation
        new_chat_obj._tracker.sent_ids = new_tracker_sent_ids # __init__ creates a new tracker
        new_chat_obj._in_toolloop = self._in_toolloop # Copy current toolloop state

        return new_chat_obj


def ask(
    prompt: str | list | Any,
    model: str = _DEFAULT_MODEL,
    sp: str | None = None,
    attach: Attachments | str | Path | list | None = None,
    tools: list | None = None,
    auto_toolloop: bool | None = None,
    max_tool_steps: int = 10,
    trace_tool_calls: bool = False,
    **chat_kwargs
) -> str:
    """
    One-shot LLM call with attachments, tools, and auto tool loop support.
    (Docstring arguments match previous version)
    """
    if auto_toolloop is None:
        auto_toolloop = bool(tools) # Enable auto_toolloop if tools are provided

    # `new_chat` will handle passing `cli`, `api_key` etc. from `chat_kwargs`
    # and also `auto_toolloop`, `max_tool_steps`, `trace_tool_calls`.
    if "cli" in chat_kwargs and isinstance(chat_kwargs["cli"], claudette.Client):
        model = chat_kwargs["cli"].model


    chat = new_chat(
        model=model,
        sp=sp,
        attach=attach,
        tools=tools,
        auto_toolloop=auto_toolloop,
        max_tool_steps=max_tool_steps,
        trace_tool_calls=trace_tool_calls,
        **chat_kwargs,
    )
    # Any API-specific kwargs in `chat_kwargs` (like temperature) are passed to `new_chat`
    # and then to `_CtxToolChat`'s constructor.
    # `_CtxToolChat.__call__` will then use them.
    result = chat(prompt) # No need to pass chat_kwargs here again, they are part of chat obj state or default __call__

    if isinstance(result, claudette.Chat):
        return claudette.contents(result)
    else:
        return cosette.contents(result)


# ----------- Tool dispatch: function caller -----------
_user_set_global_tool_ns: Dict[str, Callable] | None = None # Keep for explicit user setting

def set_tool_namespace(namespace: Dict[str, Callable]): # Changed type hint
    """
    Sets a user-defined global namespace for `call_func` to find tool functions.
    This is a fallback if tools are not passed directly or found elsewhere.
    """
    global _user_set_global_tool_ns
    _user_set_global_tool_ns = namespace

def call_func(func_name: str, args: dict, ns: Dict | None = None) -> Any: # ns from cosette is List[Dict]
    """
    Dispatches tool calls by name.
    Looks for `func_name` in:
    1. Tools passed directly to the current chat instance (via _CURRENT_CHAT_TOOL_NAMESPACE).
    2. The `ns` dictionary if provided by cosette (though its format might not be a direct name->callable map).
    3. The globally set namespace via `set_tool_namespace()`.
    4. `globals()` of this module (onetokenpy.core).
    """
    # MODIFICATION 7: Prioritize _CURRENT_CHAT_TOOL_NAMESPACE
    if _CURRENT_CHAT_TOOL_NAMESPACE and func_name in _CURRENT_CHAT_TOOL_NAMESPACE:
        f = _CURRENT_CHAT_TOOL_NAMESPACE[func_name]
        if callable(f):
            return f(**args)
        else: # Should not happen if tool_function_map is built correctly
            warnings.warn(f"Tool '{func_name}' found in chat's tool map but is not callable.", UserWarning)


    # `ns` from cosette (self.tools in Chat) is a list of tool definitions,
    # not a direct name -> callable map that this call_func was originally designed for.
    # We'll skip trying to use it directly unless we parse it here.
    # For now, we rely on _CURRENT_CHAT_TOOL_NAMESPACE or the fallbacks.
    # If you want to use `ns` from cosette, you'd iterate through it:
    # if ns:
    #    for tool_def in ns: # ns is List[Dict] where each dict is like OpenAI tool schema
    #        if isinstance(tool_def, dict) and tool_def.get("function", {}).get("name") == func_name:
    #            # Problem: how to get the *callable* from just the definition?
    #            # This is why _CURRENT_CHAT_TOOL_NAMESPACE is better.
    #            pass


    # Fallback to user-set global namespace
    if _user_set_global_tool_ns and func_name in _user_set_global_tool_ns:
        f = _user_set_global_tool_ns[func_name]
    elif func_name in globals(): # Fallback to current module's globals (less likely for user tools)
        f = globals()[func_name]
    else:
        raise NameError(
            f"Tool function '{func_name}' not found. Searched in:\n"
            "1. Tools provided directly to the current chat.\n"
            "2. User-set global namespace (via set_tool_namespace()).\n"
            "3. onetokenpy.core.globals()."
        )

    if not callable(f):
        raise TypeError(f"Object '{func_name}' found but is not callable.")
    return f(**args)


import cosette.core
cosette.core.call_func = call_func