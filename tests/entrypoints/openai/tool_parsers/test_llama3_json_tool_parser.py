# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import (
    Llama3JsonToolParser)


@pytest.fixture
def parser():
    # Use a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return Llama3JsonToolParser(tokenizer)


def test_extract_tool_calls_simple(parser):
    # Test with a simple tool call
    model_output = ('Here is the result: {"name": "getOpenIncidentsTool", '
                    '"parameters": {}} Would you like to know more?')
    result = parser.extract_tool_calls(model_output, None)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[0].function.arguments == "{}"
    assert result.content is None


def test_extract_tool_calls_with_arguments(parser):
    # Test with a tool call that has arguments
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test query", '
        '"limit": 10}}')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test query"' in result.tool_calls[0].function.arguments
    assert '"limit": 10' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_no_json(parser):
    # Test with text that doesn't contain a JSON object
    model_output = "This is just some text without any tool calls"
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_invalid_json(parser):
    # Test with invalid JSON
    model_output = '{"name": "invalidTool", "parameters": {invalid json}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_with_arguments_key(parser):
    # Test with a tool call that uses "arguments" instead of "parameters"
    model_output = '{"name": "searchTool", "arguments": {"query": "test"}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test"' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_multiple_json(parser):
    # Test with multiple JSONs separated by semicolons
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3

    # Check first tool call
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test1"' in result.tool_calls[0].function.arguments

    # Check second tool call
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[1].function.arguments == "{}"

    # Check third tool call
    assert result.tool_calls[2].function.name == "searchTool"
    assert '"query": "test2"' in result.tool_calls[2].function.arguments


def test_extract_tool_calls_multiple_json_with_whitespace(parser):
    # Test with multiple JSONs separated by semicolons and extra whitespace
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}} ; '
        '{"name": "getOpenIncidentsTool", "parameters": {}} ; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_multiple_json_with_surrounding_text(parser):
    # Test with multiple JSONs and surrounding text
    model_output = (
        'Here are the results: '
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}} '
        'Would you like to know more?')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_with_control_characters(parser):
    # Test with JSON containing control characters (literal newlines)
    model_output = ('{"name": "write_file", "parameters": {"content": "Line 1\nLine 2\nLine 3", "filename": "test.txt"}}')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "write_file"
    
    # Check that the arguments contain the properly escaped content
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["content"] == "Line 1\nLine 2\nLine 3"
    assert args["filename"] == "test.txt"


def test_extract_tool_calls_with_various_control_chars(parser):
    # Test with various control characters
    model_output = ('{"name": "process_text", "parameters": {"text": "Tab:\tNewline:\nCarriage:\rBell:\x07Delete:\x7f"}}')
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "process_text"
    
    # Check that the arguments were parsed correctly
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    expected_text = "Tab:\tNewline:\nCarriage:\rBell:\x07Delete:\x7f"
    assert args["text"] == expected_text


def test_sanitize_json_string():
    # Test the sanitize_json_string utility function directly
    from vllm.entrypoints.openai.tool_parsers.utils import sanitize_json_string
    
    # Test with literal newlines
    input_str = '{"message": "line1\nline2"}'
    expected = '{"message": "line1\\nline2"}'
    result = sanitize_json_string(input_str)
    assert result == expected
    
    # Test with various control characters
    input_str = '{"text": "null:\x00tab:\tbell:\x07newline:\ndelete:\x7f"}'
    result = sanitize_json_string(input_str)
    
    # Should contain escaped versions
    assert '\\u0000' in result  # null
    assert '\t' in result  # tab preserved
    assert '\\u0007' in result  # bell
    assert '\\n' in result  # newline
    assert '\\u007f' in result  # delete
    
    # Verify the result is valid JSON
    import json
    json.loads(result)  # Should not raise an exception
