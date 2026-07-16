#!/usr/bin/env python3
"""Quick test of the translation logic (no real API calls needed)."""
import json
import sys
sys.path.insert(0, ".")

from anthropic_to_deepseek_proxy import (
    anthropic_to_openai_messages,
    openai_to_anthropic_response,
    anthropic_tools_to_openai,
    StreamTranslator,
)

def test_basic():
    print("=== Test 1: Basic text message ===")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
        ]
    }
    result = anthropic_to_openai_messages(body)
    assert result["model"] == "deepseek-chat", f"Model mismatch: {result['model']}"
    assert len(result["messages"]) == 2  # system + user
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["content"] == "Hello!"
    print(f"  Model: {result['model']}")
    print(f"  Messages: {json.dumps(result['messages'], indent=4)}")
    print("  ✅ PASS")

def test_tools():
    print("\n=== Test 2: Tools translation ===")
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "What is the weather?"}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }]
    }
    result = anthropic_to_openai_messages(body)
    assert "tools" in result
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"] == "get_weather"
    print(f"  OpenAI tools: {json.dumps(result['tools'], indent=4)}")
    print("  ✅ PASS")

def test_response():
    print("\n=== Test 3: Response translation ===")
    resp = {
        "id": "chatcmpl-123",
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    result = openai_to_anthropic_response(resp, "msg_123", "claude-sonnet-4-20250514")
    assert result["role"] == "assistant"
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hi there!"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5
    print(f"  Anthropic response: {json.dumps(result, indent=4)}")
    print("  ✅ PASS")

def test_tool_response():
    print("\n=== Test 4: Tool call response ===")
    resp = {
        "id": "chatcmpl-456",
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing"}'
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    }
    result = openai_to_anthropic_response(resp, "msg_456", "claude-3-5-sonnet-20241022")
    assert result["stop_reason"] == "tool_use"
    tool_block = result["content"][0]
    assert tool_block["type"] == "tool_use"
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"location": "Beijing"}
    print(f"  Anthropic tool response: {json.dumps(result, indent=4)}")
    print("  ✅ PASS")

def test_image():
    print("\n=== Test 5: Image input ===")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aaaa"}}
            ]
        }]
    }
    result = anthropic_to_openai_messages(body)
    msgs = result["messages"]
    assert len(msgs) == 1  # no system prompt, just user
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64,aaaa" in content[1]["image_url"]["url"]
    print(f"  Content array: {json.dumps(content, indent=4)}")
    print("  ✅ PASS")

def test_stream_translator():
    print("\n=== Test 6: Stream translation ===")
    t = StreamTranslator(request_id="msg_stream_test", model="deepseek-chat")

    # Simulate receiving chunks
    chunks = [
        # First chunk with content
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
        # Second chunk
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
        # Final chunk with finish_reason and usage
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]

    output = ""
    for chunk in chunks:
        output += t.process_chunk(chunk)

    # Verify key Anthropic SSE events
    assert "message_start" in output
    assert "content_block_start" in output
    assert "content_block_delta" in output
    assert "text_delta" in output
    assert "content_block_stop" in output
    assert "message_delta" in output
    assert "message_stop" in output

    # Check actual text content
    assert "Hello" in output
    assert " world" in output
    assert "end_turn" in output

    print(f"  Output ({len(output)} bytes):")
    print(output[:500])
    print("  ...")
    print("  ✅ PASS")

def test_stop_sequences():
    print("\n=== Test 7: Stop sequences ===")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stop_sequences": ["END", "STOP"]
    }
    result = anthropic_to_openai_messages(body)
    assert result["stop"] == ["END", "STOP"]
    print(f"  Stop: {result['stop']}")
    print("  ✅ PASS")

def test_streaming_flag():
    print("\n=== Test 8: Streaming flag ===")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True
    }
    result = anthropic_to_openai_messages(body)
    assert result["stream"] is True
    assert "stream_options" in result
    assert result["stream_options"]["include_usage"] is True
    print(f"  Stream: {result['stream']}, stream_options: {result['stream_options']}")
    print("  ✅ PASS")

if __name__ == "__main__":
    test_basic()
    test_tools()
    test_response()
    test_tool_response()
    test_image()
    test_stream_translator()
    test_stop_sequences()
    test_streaming_flag()
    print("\n" + "=" * 50)
    print("✅ All 8 tests passed!")
    print("=" * 50)
