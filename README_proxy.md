# Anthropic → DeepSeek Translation Proxy

A lightweight HTTP proxy that translates **Anthropic Messages API** requests into **DeepSeek (OpenAI-compatible) API** requests, and translates responses back.

This lets you use **Claude Code** with **DeepSeek** as the backend model, by intercepting Claude Code's API calls and translating them on the fly.

## How It Works (inspired by claude-code-router)

The [claude-code-router](https://github.com/musistudio/claude-code-router) project works as a local API gateway. The pattern is:

```
┌─────────────┐     Anthropic API      ┌──────────────┐     OpenAI API      ┌─────────────┐
│  Claude CLI  │ ─────────────────────→ │  This Proxy   │ ─────────────────→ │  DeepSeek API │
│              │ ←───────────────────── │               │ ←───────────────── │              │
└─────────────┘   Anthropic Response   └──────────────┘   OpenAI Response   └─────────────┘
```

1. **Claude Code** sends requests to the configured `ANTHROPIC_BASE_URL`
2. The proxy identifies the protocol by the URL path (`/v1/messages` = Anthropic Messages API)
3. The request body is translated from **Anthropic → OpenAI** format
4. The translated request is forwarded to **DeepSeek**
5. DeepSeek's OpenAI-format response is translated back to **Anthropic** format
6. The translated response is returned to Claude Code

## Requirements

- Python 3.8+
- `requests` library (`pip install requests`)

## Quick Start

### 1. Set your DeepSeek API key

```bash
export DEEPSEEK_API_KEY=sk-your-deepseek-key
```

### 2. Start the proxy

```bash
python anthropic_to_deepseek_proxy.py
```

The proxy starts on `http://127.0.0.1:9999` by default.

### 3. Point Claude Code at the proxy

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9999
export ANTHROPIC_API_KEY=any-value  # required but not used by the proxy

# Now run Claude Code — it will use DeepSeek via the proxy
claude
```

## Configuration

All settings can be configured via environment variables or command-line arguments:

| Env Variable | CLI Flag | Default | Description |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | `--deepseek-key` | (required) | Your DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `--deepseek-base-url` | `https://api.deepseek.com` | DeepSeek API base URL |
| `DEEPSEEK_MODEL` | `--model` | `deepseek-chat` | DeepSeek model to use |
| `LISTEN_HOST` | `--host` | `127.0.0.1` | Proxy listen address |
| `LISTEN_PORT` | `--port` | `9999` | Proxy listen port |

### Using DeepSeek Reasoner (R1)

```bash
export DEEPSEEK_MODEL=deepseek-reasoner
python anthropic_to_deepseek_proxy.py
```

### Custom port

```bash
python anthropic_to_deepseek_proxy.py --port 8888
```

## Model Mapping

Claude model names in requests are automatically mapped to DeepSeek:

| Claude Model | DeepSeek Model |
|---|---|
| `claude-sonnet-4-20250514` | `deepseek-chat` |
| `claude-3-5-sonnet-20241022` | `deepseek-chat` |
| `claude-opus-4-20250514` | `deepseek-chat` |
| Any `claude-*` prefix | `deepseek-chat` |

If `DEEPSEEK_MODEL` env var is set, all requests use that model regardless of the input.

## Supported Features

### Request Translation (Anthropic → OpenAI)

| Feature | Supported |
|---|---|
| Text messages (string & content blocks) | ✅ |
| System prompts (string & array) | ✅ |
| Multi-turn conversations | ✅ |
| Image inputs (base64) | ✅ |
| Tool use / function calling | ✅ |
| Stop sequences (`stop_sequences` → `stop`) | ✅ |
| Temperature, top_p | ✅ |
| Streaming (`stream: true`) | ✅ |

### Response Translation (OpenAI → Anthropic)

| Feature | Supported |
|---|---|
| Text content blocks | ✅ |
| Tool use (`tool_calls` → `tool_use`) | ✅ |
| Stop reasons (`end_turn`, `max_tokens`, `tool_use`) | ✅ |
| Token usage | ✅ |
| Streaming SSE events | ✅ |

### Streaming Event Types

The proxy translates OpenAI streaming chunks into proper Anthropic SSE events:

- `message_start` — message metadata
- `content_block_start` — new content block (text or tool_use)
- `content_block_delta` — content update (`text_delta` or `input_json_delta`)
- `content_block_stop` — content block complete
- `message_delta` — stop_reason and usage
- `message_stop` — stream end

## Testing the Translation Logic

Run the unit tests to verify translations without making real API calls:

```bash
python test_translation.py
```

## Testing Against the Real DeepSeek API

Once the proxy is running with a valid API key, test it directly:

```bash
# Non-streaming request
curl -s http://127.0.0.1:9999/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello in 3 words"}]
  }' | jq .

# Streaming request
curl -s http://127.0.0.1:9999/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "max_tokens": 100,
    "stream": true,
    "messages": [{"role": "user", "content": "Count from 1 to 5"}]
  }'
```

## Architecture

```
anthropic_to_deepseek_proxy.py
├── _config                      # Global configuration dict
├── anthropic_to_openai_messages()   # Request: Anthropic → OpenAI
│   ├── anthropic_content_to_openai_content()  # Content block translation
│   ├── anthropic_tools_to_openai()            # Tool definition translation
│   ├── _convert_assistant_message()           # Assistant message (tool_use)
│   ├── _convert_tool_choice()                 # Tool choice mapping
│   └── _map_model()                           # Model name mapping
├── openai_to_anthropic_response()  # Response: OpenAI → Anthropic
└── StreamTranslator                # SSE stream translation
    ├── process_chunk()             # OpenAI chunk → Anthropic SSE
    ├── _process_text_delta()       # Text delta events
    ├── _process_thinking_delta()   # Reasoning delta (R1)
    ├── _process_tool_call_start()  # Tool call start
    ├── _process_tool_args_delta()  # Tool argument deltas
    └── _finish()                   # End-of-stream events
```

## Limitations

- **DeepSeek-specific features** like `reasoning_content` (R1) are mapped to Anthropic `thinking_delta` events, but full Anthropic extended thinking support is not yet implemented
- **Anthropic-specific features** like `cache_control` (prompt caching) are silently dropped
- `top_k` is Anthropic-only and not forwarded to DeepSeek
- Complex multi-modal scenarios (multiple images + text in one block) may need further refinement

## Troubleshooting

**Claude Code says "No models available"**
- Make sure the proxy is running and reachable
- Check `curl http://127.0.0.1:9999/v1/models` returns a model list

**"Authentication Fails" error**
- Verify `DEEPSEEK_API_KEY` is set correctly
- Check DeepSeek has credits available

**Streaming doesn't work**
- Make sure the Accept header includes `text/event-stream`
- Try without streaming first to verify connectivity

**Tool use errors in Claude Code**
- DeepSeek's function calling format may differ slightly. Check the proxy logs for details.
