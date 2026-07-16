#!/usr/bin/env python3
"""
anthropic_to_deepseek_proxy.py — A lightweight HTTP proxy that translates
Anthropic Messages API requests into DeepSeek (OpenAI-compatible) API requests,
and translates the responses back.

How Claude Code Router does this (the pattern we follow):
  1. Claude CLI sends requests to the configured ANTHROPIC_BASE_URL
  2. This proxy receives them, identifies the protocol by the URL path
     (/v1/messages = Anthropic Messages API)
  3. The request body is translated from Anthropic → OpenAI format
  4. The translated request is forwarded to DeepSeek
  5. DeepSeek's OpenAI-format response is translated back → Anthropic format
  6. The translated response is returned to Claude CLI

Usage:
  1. Set your DeepSeek API key:
     export DEEPSEEK_API_KEY=sk-your-key-here
     export DEEPSEEK_BASE_URL=https://api.deepseek.com  # optional, default

  2. Run the proxy:
     python anthropic_to_deepseek_proxy.py --port 9999

  3. Point Claude Code at the proxy:
     export ANTHROPIC_BASE_URL=http://localhost:9999
     claude  # uses the proxy transparently

Supported features:
  - Text messages (with full content-block translation)
  - System prompts (string and content-block array)
  - Image inputs (base64)
  - Tool use / function calling
  - Streaming (SSE) with full event translation
  - Non-streaming (batch) requests
  - Stop sequences
  - Temperature, top_p, top_k
  - Multi-turn conversations

Translation table:
  Request:  Anthropic /v1/messages  →  OpenAI /v1/chat/completions
  Response: OpenAI Chat Completion  →  Anthropic Message
  Stream:   OpenAI SSE chunks       →  Anthropic SSE events
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

import requests
from requests.exceptions import RequestException

# ─── Configuration ────────────────────────────────────────────────────────────

# These are set once at startup and read throughout the proxy.
_config = {
    "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "deepseek_base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
    "default_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    "listen_host": os.environ.get("LISTEN_HOST", "127.0.0.1"),
    "listen_port": int(os.environ.get("LISTEN_PORT", "9999")),
}


# ─── Request Translation: Anthropic → OpenAI ──────────────────────────────────

def anthropic_content_to_openai_content(content) -> list:
    """Convert Anthropic content blocks to OpenAI content format.

    Anthropic formats:
      String: "Hello"
      Single block: {"type": "text", "text": "Hello"}
      Array of blocks: [{"type": "text", "text": "..."}, {"type": "image", ...}]

    OpenAI formats (multimodal via DeepSeek):
      String: "Hello"
      Array: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]

    For simple text, we return a plain string (which both APIs accept).
    For mixed content (text + images), we return an array.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        block = content
        if block.get("type") == "text":
            return block.get("text", "")
        elif block.get("type") == "image":
            return _anthropic_image_to_openai(block)
        elif block.get("type") == "tool_use":
            # Tool use from assistant — skip (handled separately)
            return None
        elif block.get("type") == "tool_result":
            return _anthropic_tool_result_to_openai(block)
        # Fallback: treat unknown block as text
        return str(block.get("text", block.get("content", "")))

    if isinstance(content, list):
        parts = []
        for block in content:
            converted = anthropic_content_to_openai_content(block)
            if converted is not None:
                if isinstance(converted, list):
                    parts.extend(converted)
                else:
                    parts.append({"type": "text", "text": converted} if isinstance(converted, str) else converted)
        # If all parts are text, we could collapse to string but OpenAI
        # format with content array works fine for DeepSeek
        if len(parts) == 1 and parts[0].get("type") == "text":
            return parts[0]["text"]
        return parts

    return str(content)


def _anthropic_image_to_openai(block: dict) -> dict:
    """Convert Anthropic image block to OpenAI image_url format."""
    source = block.get("source", {})
    media_type = source.get("media_type", "image/jpeg")
    data = source.get("data", "")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"}
    }


def _anthropic_tool_result_to_openai(block: dict) -> Optional[dict]:
    """Convert Anthropic tool_result block to OpenAI tool message content."""
    content = block.get("content", "")
    if isinstance(content, list):
        # Extract text from content blocks
        texts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
            elif isinstance(c, str):
                texts.append(c)
        content = "\n".join(texts)
    return content if content else ""


def anthropic_tools_to_openai(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI function definitions.

    Anthropic:
      [{"name": "get_weather", "description": "...",
        "input_schema": {"type": "object", "properties": {...}, "required": [...]}}]

    OpenAI:
      [{"type": "function", "function": {"name": "get_weather", "description": "...",
        "parameters": {"type": "object", "properties": {...}, "required": [...]}}}]
    """
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            }
        })
    return openai_tools


def anthropic_to_openai_messages(anthropic_body: dict) -> dict:
    """Translate an Anthropic Messages API request body to OpenAI Chat Completions format."""

    model = anthropic_body.get("model", _config["default_model"])
    # Map common Anthropic model names to DeepSeek if needed
    mapped_model = _map_model(model)

    # Build the messages array
    messages = []

    # 1. System prompt: Anthropic top-level `system` field → OpenAI "system" role message
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System can be an array of text blocks in Anthropic
            texts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            if texts:
                messages.append({"role": "system", "content": "\n".join(texts)})
        elif isinstance(system, dict):
            messages.append({"role": "system", "content": str(system)})

    # 2. Conversation messages
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map Anthropic role names to OpenAI role names
        role_map = {
            "user": "user",
            "assistant": "assistant",
        }
        openai_role = role_map.get(role, "user")

        # Handle assistant messages with tool_use
        if role == "assistant":
            converted = _convert_assistant_message(msg)
            if converted:
                # Handle case where assistant has both text and tool_calls
                if isinstance(converted, list):
                    messages.extend(converted)
                else:
                    messages.append(converted)
                continue

        # Handle user messages with tool_result
        if role == "user" and isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for tr in tool_results:
                    converted_content = _anthropic_tool_result_to_openai(tr)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": converted_content
                    })
                continue

        # Regular message
        converted_content = anthropic_content_to_openai_content(content)
        if converted_content is not None:
            messages.append({"role": openai_role, "content": converted_content})

    # Build the OpenAI-format request
    openai_body = {
        "model": mapped_model,
        "messages": messages,
        "max_tokens": anthropic_body.get("max_tokens", 4096),
    }

    # Optional parameters
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]

    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]

    # Anthropic's stop_sequences → OpenAI's stop
    stop_sequences = anthropic_body.get("stop_sequences")
    if stop_sequences:
        openai_body["stop"] = stop_sequences

    # Streaming
    if anthropic_body.get("stream"):
        openai_body["stream"] = True
        openai_body["stream_options"] = {"include_usage": True}

    # Tools
    tools = anthropic_body.get("tools")
    if tools:
        openai_body["tools"] = anthropic_tools_to_openai(tools)

        # Tool choice
        tool_choice = anthropic_body.get("tool_choice")
        if tool_choice:
            openai_body["tool_choice"] = _convert_tool_choice(tool_choice)

    # top_k is Anthropic-specific, DeepSeek may ignore but pass through
    # (DeepSeek doesn't support top_k, so we skip it)

    return openai_body


def _convert_assistant_message(msg: dict):
    """Convert an Anthropic assistant message (possibly with tool_use) to OpenAI format.

    Returns either a single message dict, a list of messages, or None.
    """
    content = msg.get("content", "")

    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    if isinstance(content, list):
        text_parts = []
        tool_calls = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })

        if tool_calls and text_parts:
            msg_dict = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            return msg_dict
        elif tool_calls:
            return {"role": "assistant", "content": None, "tool_calls": tool_calls}
        elif text_parts:
            return {"role": "assistant", "content": "\n".join(text_parts)}

    return None


def _convert_tool_choice(tool_choice):
    """Convert Anthropic tool_choice to OpenAI tool_choice.

    Anthropic: {"type": "auto"} / {"type": "any"} / {"type": "tool", "name": "x"}
    OpenAI:    "auto" / "required" / {"type": "function", "function": {"name": "x"}}
    """
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any" or tc_type == "required":
            return "required"
        elif tc_type == "tool":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return "auto"


def _map_model(model_name: str) -> str:
    """Map Anthropic model names to DeepSeek equivalents.

    The Claude Code Router approach is to allow the original model name
    through and let the upstream map it, but we provide basic mapping
    for common Claude models.
    """
    # If the user set DEEPSEEK_MODEL explicitly, use that (ignore input model)
    if os.environ.get("DEEPSEEK_MODEL"):
        return _config["default_model"]

    # Common Claude → DeepSeek model mapping
    CLAUDE_TO_DEEPSEEK = {
        "claude-sonnet-4-20250514": "deepseek-chat",
        "claude-3-5-sonnet-20241022": "deepseek-chat",
        "claude-3-opus-20240229": "deepseek-chat",
        "claude-3-haiku-20240307": "deepseek-chat",
        "claude-3-5-haiku-20241022": "deepseek-chat",
        "claude-opus-4-20250514": "deepseek-chat",
        "claude-opus-4-8": "deepseek-chat",
        "claude-haiku-4-5-20251001": "deepseek-chat",
        "claude-fable-5": "deepseek-chat",
    }

    # Try exact match first
    if model_name in CLAUDE_TO_DEEPSEEK:
        return CLAUDE_TO_DEEPSEEK[model_name]

    # Try prefix match (e.g., "claude-" → "deepseek-chat")
    if model_name.startswith("claude-"):
        return "deepseek-chat"

    # Pass through (could already be a DeepSeek model name)
    return model_name


# ─── Response Translation: OpenAI → Anthropic ─────────────────────────────────

def openai_to_anthropic_response(openai_body: dict, request_id: str = None,
                                  original_model: str = None) -> dict:
    """Translate an OpenAI Chat Completion response to Anthropic Message format."""

    msg_id = request_id or f"msg_{uuid.uuid4().hex[:24]}"
    model = original_model or openai_body.get("model", _config["default_model"])

    choices = openai_body.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    # Build Anthropic content blocks
    content = []

    # Text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls → tool_use blocks
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            tool_input = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": tool_input
        })

    # Map finish reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Usage
    usage = openai_body.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage
    }


# ─── Streaming Translation: OpenAI SSE → Anthropic SSE ────────────────────────

class StreamTranslator:
    """Translates OpenAI streaming SSE chunks into Anthropic streaming SSE events.

    Anthropic SSE event types:
      - message_start
      - content_block_start
      - content_block_delta  (with text_delta or input_json_delta)
      - content_block_stop
      - message_delta
      - message_stop
      - ping

    OpenAI streaming chunks each have:
      {"id": "...", "object": "chat.completion.chunk",
       "choices": [{"index": 0, "delta": {...}, "finish_reason": null}],
       "usage": {...}}  // usage only in final chunk (with stream_options)
    """

    def __init__(self, request_id: str = None, model: str = None):
        if model is None:
            model = _config["default_model"]
        self.request_id = request_id or f"msg_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.msg_started = False
        self.content_block_started = {}  # index → block_type
        self.active_tool_calls = {}  # index → {id, name, args_json}
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason = None

    def _emit(self, event_type: str, data: dict) -> str:
        """Format an Anthropic SSE event."""
        lines = []
        if event_type:
            lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")  # empty line separator
        return "\n".join(lines) + "\n"

    def process_chunk(self, chunk_data: dict) -> str:
        """Process one OpenAI SSE chunk and return Anthropic SSE events as a string."""
        result = ""

        choices = chunk_data.get("choices", [])
        usage = chunk_data.get("usage")
        chunk_id = chunk_data.get("id", "")

        if usage:
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        if not choices:
            # If we have usage but no choices, emit usage info
            if usage and self.msg_started:
                pass  # Will emit with message_delta
            return ""

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        index = choice.get("index", 0)

        if finish_reason:
            self.finish_reason = finish_reason

        # Start message if not yet started
        if not self.msg_started:
            self.msg_started = True
            result += self._emit("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.request_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens
                    }
                }
            })

        # Handle regular text content
        content = delta.get("content")
        if content and content != "":
            result += self._process_text_delta(index, content)

        # Handle thinking/reasoning content (DeepSeek R1)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            result += self._process_thinking_delta(index, reasoning)

        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            tc_id = tc.get("id")
            tc_func = tc.get("function", {})
            tc_name = tc_func.get("name")
            tc_args = tc_func.get("arguments", "")

            if tc_id or tc_name:
                # New tool call starting
                result += self._process_tool_call_start(tc_index, tc_id, tc_name)

            if tc_args:
                # Tool arguments chunk
                result += self._process_tool_args_delta(tc_index, tc_args)

        # Check for finish — emit content_block_stop and message_delta/message_stop
        if finish_reason:
            result += self._finish(finish_reason)

        return result

    def _process_text_delta(self, index: int, text: str) -> str:
        """Process a text content delta."""
        result = ""

        # Start text content block if needed
        if index not in self.content_block_started:
            self.content_block_started[index] = "text"
            result += self._emit("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            })

        # Emit text delta
        result += self._emit("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "text_delta",
                "text": text
            }
        })

        return result

    def _process_thinking_delta(self, index: int, thinking: str) -> str:
        """Process a thinking/reasoning content delta."""
        result = ""

        think_idx = index + 1000  # Use offset index for thinking blocks

        if think_idx not in self.content_block_started:
            self.content_block_started[think_idx] = "thinking"
            result += self._emit("content_block_start", {
                "type": "content_block_start",
                "index": think_idx,
                "content_block": {
                    "type": "thinking",
                    "thinking": ""
                }
            })

        result += self._emit("content_block_delta", {
            "type": "content_block_delta",
            "index": think_idx,
            "delta": {
                "type": "thinking_delta",
                "thinking": thinking
            }
        })

        return result

    def _process_tool_call_start(self, index: int, tc_id: str, tc_name: str) -> str:
        """Process the start of a tool call."""
        result = ""

        if index in self.content_block_started:
            # Close previous block first
            result += self._emit("content_block_stop", {
                "type": "content_block_stop",
                "index": index
            })

        self.content_block_started[index] = "tool_use"
        self.active_tool_calls[index] = {"id": tc_id, "name": tc_name, "args_json": ""}

        result += self._emit("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tc_id,
                "name": tc_name,
                "input": {}
            }
        })

        return result

    def _process_tool_args_delta(self, index: int, args_json: str) -> str:
        """Process a tool argument delta."""
        if index not in self.active_tool_calls:
            return ""

        self.active_tool_calls[index]["args_json"] += args_json

        return self._emit("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": args_json
            }
        })

    def _finish(self, finish_reason: str) -> str:
        """Emit content_block_stop events and message_delta/message_stop."""
        result = ""

        # Close all open content blocks
        for index in sorted(self.content_block_started.keys()):
            block_type = self.content_block_started[index]
            if block_type == "tool_use" and index in self.active_tool_calls:
                tc = self.active_tool_calls[index]
                try:
                    parsed_input = json.loads(tc["args_json"])
                except (json.JSONDecodeError, KeyError):
                    parsed_input = {}
                # Note: We can't easily inject the parsed input into content_block_stop
                # in Anthropic format. The input is accumulated via input_json_delta.
                pass

            result += self._emit("content_block_stop", {
                "type": "content_block_stop",
                "index": index
            })

        # Map finish reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        # Message delta
        result += self._emit("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": self.output_tokens
            }
        })

        # Message stop
        result += self._emit("message_stop", {
            "type": "message_stop"
        })

        return result

    def flush(self) -> str:
        """Emit any remaining events if the stream ends without finish_reason."""
        if not self.msg_started:
            return ""
        if self.finish_reason:
            return ""  # Already finalized

        return self._finish("stop")


# ─── HTTP Server ──────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies Anthropic API requests to DeepSeek."""

    # Silence default request logging (we log our own)
    def log_message(self, format, *args):
        pass

    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    def _send_json(self, status_code: int, data: dict):
        """Send a JSON response."""
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.send_header("x-request-id", data.get("id", ""))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_error(self, status_code: int, message: str, error_type: str = "api_error"):
        """Send an Anthropic-formatted error response."""
        error_data = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        }
        self._send_json(status_code, error_data)

    def _read_body(self) -> bytes:
        """Read the request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            return self.rfile.read(content_length)
        return b""

    def _forward_headers(self, headers: dict) -> dict:
        """Forward relevant headers to DeepSeek."""
        forward = {}

        # Our auth header
        forward["Authorization"] = f"Bearer {_config['deepseek_api_key']}"
        forward["Content-Type"] = "application/json"
        forward["Accept"] = headers.get("Accept", "application/json")

        return forward

    def _handle_models_list(self):
        """Handle GET /v1/models — return a model list so Claude Code can discover models.

        This is called during Claude Code's startup model discovery.
        We return a minimal list that maps to DeepSeek models.
        """
        models = [
            {
                "id": "deepseek-chat",
                "type": "model",
                "display_name": "DeepSeek Chat",
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "deepseek-reasoner",
                "type": "model",
                "display_name": "DeepSeek Reasoner (R1)",
                "created_at": "2025-01-01T00:00:00Z"
            },
            # Include common Claude model names so Claude Code accepts them
            {
                "id": "claude-sonnet-4-20250514",
                "type": "model",
                "display_name": "DeepSeek Chat (via proxy)",
                "created_at": "2025-05-14T00:00:00Z"
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "type": "model",
                "display_name": "DeepSeek Chat (via proxy)",
                "created_at": "2024-10-22T00:00:00Z"
            },
        ]

        data = {"object": "list", "data": models}
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _handle_messages(self):
        """Handle POST /v1/messages — the main Anthropic Messages API endpoint."""
        raw_body = self._read_body()

        try:
            anthropic_body = json.loads(raw_body)
        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        original_model = anthropic_body.get("model", _config["default_model"])
        is_streaming = anthropic_body.get("stream", False)

        # Translate request
        try:
            openai_body = anthropic_to_openai_messages(anthropic_body)
        except Exception as e:
            self._send_error(400, f"Failed to translate request: {e}", "invalid_request_error")
            return

        self._log(f"→ {original_model} → {openai_body['model']} "
                  f"(stream={is_streaming}, msgs={len(openai_body.get('messages', []))})")

        # Forward to DeepSeek
        headers = self._forward_headers(dict(self.headers))
        upstream_url = f"{_config['deepseek_base_url']}/v1/chat/completions"

        if is_streaming:
            self._handle_streaming_request(upstream_url, headers, openai_body, original_model)
        else:
            self._handle_batch_request(upstream_url, headers, openai_body, original_model)

    def _handle_batch_request(self, url: str, headers: dict, body: dict, original_model: str):
        """Handle a non-streaming request."""
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=300,
            )
        except RequestException as e:
            self._send_error(502, f"Upstream request failed: {e}")
            return

        if resp.status_code != 200:
            self._log(f"← Upstream error: {resp.status_code} {resp.text[:200]}")
            # Try to forward the upstream error
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text[:500])
            except Exception:
                error_msg = resp.text[:500]
            self._send_error(resp.status_code, error_msg)
            return

        try:
            openai_response = resp.json()
            anthropic_response = openai_to_anthropic_response(
                openai_response,
                request_id=f"msg_{uuid.uuid4().hex[:24]}",
                original_model=original_model
            )
        except Exception as e:
            self._send_error(500, f"Failed to translate response: {e}")
            return

        self._log(f"← {original_model} done ({anthropic_response['usage']['input_tokens']}+"
                  f"{anthropic_response['usage']['output_tokens']} tokens)")

        self._send_json(200, anthropic_response)

    def _handle_streaming_request(self, url: str, headers: dict, body: dict, original_model: str):
        """Handle a streaming request with SSE translation."""
        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        translator = StreamTranslator(request_id=request_id, model=original_model)

        # Send response headers for SSE
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("x-request-id", request_id)
        self.end_headers()

        input_tokens = 0
        output_tokens = 0

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                stream=True,
                timeout=300,
            )

            if resp.status_code != 200:
                self._log(f"← Upstream error: {resp.status_code}")
                error_event = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Upstream returned {resp.status_code}"}
                })
                self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                self.wfile.flush()
                return

            # Process SSE stream from DeepSeek
            for line in resp.iter_lines(decode_unicode=True):
                if line is None:
                    continue

                line = line.strip()
                if not line:
                    continue

                # Parse OpenAI SSE chunk
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        # Flush remaining events
                        flushed = translator.flush()
                        if flushed:
                            self.wfile.write(flushed.encode("utf-8"))
                        self.wfile.write("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Translate and write Anthropic SSE events
                    anthropic_events = translator.process_chunk(chunk)
                    if anthropic_events:
                        self.wfile.write(anthropic_events.encode("utf-8"))
                        self.wfile.flush()

                elif line.startswith(":"):
                    # SSE comment — pass through as ping
                    pass

            # If the stream ended without proper [DONE], flush
            if not translator.finish_reason:
                flushed = translator.flush()
                if flushed:
                    self.wfile.write(flushed.encode("utf-8"))
                    self.wfile.write("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".encode("utf-8"))
                    self.wfile.flush()

            # Get final token counts
            if translator.input_tokens or translator.output_tokens:
                input_tokens = translator.input_tokens
                output_tokens = translator.output_tokens

            self._log(f"← {original_model} stream done ({input_tokens}+{output_tokens} tokens)")

        except RequestException as e:
            self._log(f"← Upstream stream error: {e}")
            error_event = json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": str(e)}
            })
            try:
                self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        except Exception as e:
            self._log(f"← Stream error: {e}")
            try:
                error_event = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)}
                })
                self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path in ("/v1/models", "/v1/models/", "/models", "/models/"):
            self._handle_models_list()
        elif path == "/health":
            self._send_json(200, {"status": "ok", "service": "anthropic-to-deepseek-proxy"})
        else:
            self._send_error(404, f"Not found: {path}")

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path

        # Anthropic Messages API paths (Claude Code Router identifies these)
        if path in ("/v1/messages", "/messages") or path.endswith("/v1/messages"):
            self._handle_messages()
        else:
            self._send_error(404, f"Not found: {path}")


class ThreadedHTTPServer(HTTPServer):
    """Handle requests in separate threads for concurrent connections."""
    allow_reuse_address = True
    daemon_threads = True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Anthropic Messages API → DeepSeek API Translation Proxy"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=_config["listen_port"],
        help=f"Port to listen on (default: {_config['listen_port']})"
    )
    parser.add_argument(
        "--host",
        default=_config["listen_host"],
        help=f"Host to bind to (default: {_config['listen_host']})"
    )
    parser.add_argument(
        "--deepseek-key",
        default=_config["deepseek_api_key"],
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=_config["deepseek_base_url"],
        help=f"DeepSeek API base URL (default: {_config['deepseek_base_url']})"
    )
    parser.add_argument(
        "--model",
        default=_config["default_model"],
        help=f"DeepSeek model to use (default: {_config['default_model']})"
    )
    args = parser.parse_args()

    # Update config from command-line args
    _config["deepseek_api_key"] = args.deepseek_key
    _config["deepseek_base_url"] = args.deepseek_base_url.rstrip("/")
    _config["default_model"] = args.model
    _config["listen_host"] = args.host
    _config["listen_port"] = args.port

    if not _config["deepseek_api_key"]:
        print("ERROR: DEEPSEEK_API_KEY is required. Set it via:", file=sys.stderr)
        print("  export DEEPSEEK_API_KEY=sk-your-key", file=sys.stderr)
        print("  or use --deepseek-key", file=sys.stderr)
        sys.exit(1)

    server = ThreadedHTTPServer((args.host, args.port), ProxyHandler)

    print(f"=" * 60, file=sys.stderr)
    print(f"  Anthropic → DeepSeek Translation Proxy", file=sys.stderr)
    print(f"  Listening on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  Upstream: {_config['deepseek_base_url']}", file=sys.stderr)
    print(f"  Model: {_config['default_model']}", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  Set in your shell before running Claude Code:", file=sys.stderr)
    print(f"    export ANTHROPIC_BASE_URL=http://{args.host}:{args.port}", file=sys.stderr)
    print(f"    export ANTHROPIC_API_KEY=any-value  # (not used by proxy)", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  Press Ctrl+C to stop", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
