# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Relay middleware."""

from __future__ import annotations

import logging
import json
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage,
    FunctionMessage,
    RemoveMessage,
    AIMessageChunk,
    HumanMessageChunk,
    FunctionMessageChunk,
    ToolMessageChunk,
    SystemMessageChunk,
    ChatMessageChunk,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.types import Command, Send

from nemo_relay import AnnotatedLLMRequest, AnnotatedLLMResponse, LLMRequest
from nemo_relay.codecs import LlmCodec

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest
    from nemo_relay import JsonObject

_logger = logging.getLogger(__name__)

LANGCHAIN_MODEL_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"
_LANGCHAIN_MODELED_REQUEST_KEYS = {"messages", "model", "tool_choice", "tools"}
_LC_TO_RELAY_MESSAGE_ROLE = {
    "human": "user",
    "ai": "assistant",
}

_RELAY_ROLE_TO_LC_MESSAGE_TYPE = {v: k for k, v in _LC_TO_RELAY_MESSAGE_ROLE.items()}

_FINISH_REASON_MAP = {
    "stop": "complete",
    "end_turn": "complete",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "max_tokens": "length",
    "length": "length",
    "content_filter": "content_filter",
}


def get_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model name from a LangChain chat model."""
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class LangChainCodec(LlmCodec):
    """Translate LangChain ``ModelRequest`` payloads for request intercepts."""

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a LangChain-shaped request payload into an annotated request."""
        payload = request.content
        messages = payload.get("messages", [])
        annotated_messages: list[dict[str, Any]] = []
        for message in messages:
            msg_type = message["type"]
            role = _LC_TO_RELAY_MESSAGE_ROLE.get(msg_type, msg_type)
            content = []
            tool_calls = []
            am = {"role": role}
            if role == "tool":
                am["tool_call_id"] = str(message.get("tool_call_id") or "")

            for block in message["content_blocks"]:
                if block["type"] == "tool_call":
                    tool_calls.append({
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("args", {}))
                        }
                    })
                else:
                    content.append(block)

            am["content"] = content
            if tool_calls:
                am["tool_calls"] = tool_calls

            annotated_messages.append(am)

        model = payload.get("model")
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        extra = {key: value for key, value in payload.items() if key not in _LANGCHAIN_MODELED_REQUEST_KEYS}

        alr = AnnotatedLLMRequest(
            annotated_messages,
            model=model if isinstance(model, str) else None,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice if isinstance(tool_choice, str | dict) else None,
            extra=extra or None,
        )
        return alr

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode annotated request edits back into a LangChain-shaped payload."""
        payload = dict(original.content)
        payload.update(annotated.extra)

        messages: list[JsonObject] = []
        for message in annotated.messages:
            role = message["role"]
            msg_type = _RELAY_ROLE_TO_LC_MESSAGE_TYPE.get(role, role)
            messages.append({"type": msg_type, "content_blocks": message["content"]})

        payload["messages"] = messages
        if annotated.model is not None:
            payload["model"] = annotated.model
        if annotated.tools is not None:
            payload["tools"] = annotated.tools
        if annotated.tool_choice is not None:
            payload["tool_choice"] = annotated.tool_choice

        return LLMRequest(dict(original.headers), payload)

    def decode_response(self, response: Any) -> AnnotatedLLMResponse:
        """Decode a serialized LangChain ``ModelResponse`` for observability."""
        payload = _model_response_payload_from_json(response)

        messages = payload.get("messages")
        last_ai_message = None
        for msg in reversed(messages):
            if msg["type"] == "ai":
                last_ai_message = msg
                break

        if last_ai_message is None:
            # Fall back to the last message
            last_ai_message = messages[-1]

        content_blocks = last_ai_message["content_blocks"]

        msg_id = None
        tool_calls = []
        model_name = None
        finish_reason = None
        usage = None
        message_text = None

        for block in reversed(content_blocks):
            if msg_id is None:
                msg_id = block.get("id")

            if block.get("type") == "tool_call":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "name": str(block["name"]),
                        "arguments": block.get("args") or {},
                    }
                )
            if model_name is None:
                for key in ("model_name", "model", "model_id"):
                    value = block.get(key)
                    if isinstance(value, str) and value:
                        model_name = value
                        break

            if finish_reason is None:
                finish_reason = _message_finish_reason(block)
            if usage is None:
                usage = _message_usage(block)

        extra = {}
        if "structured_response" in payload:
            extra["structured_response"] = payload["structured_response"]

        alr = AnnotatedLLMResponse(
            id=msg_id,
            model=model_name,
            message=message_text,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=usage,
            extra=extra or None,
        )
        return alr

def _relay_message_to_lc_message(message: JsonObject) -> BaseMessage:
    """Convert a NeMo Relay message dict to a LangChain `BaseMessage`."""
    type_ = message["type"]
    if type_ == "human":
        return HumanMessage(content_blocks=message["content_blocks"])
    if type_ == "ai":
        return AIMessage(content_blocks=message["content_blocks"])
    if type_ == "system":
        return SystemMessage(content_blocks=message["content_blocks"])
    if type_ == "chat":
        return ChatMessage(content_blocks=message["content_blocks"])
    if type_ == "function":
        return FunctionMessage(content_blocks=message["content_blocks"])
    if type_ == "tool":
        return ToolMessage(content_blocks=message["content_blocks"])
    if type_ == "remove":
        return RemoveMessage(content_blocks=message["content_blocks"])
    if type_ == "AIMessageChunk":
        return AIMessageChunk(content_blocks=message["content_blocks"])
    if type_ == "HumanMessageChunk":
        return HumanMessageChunk(content_blocks=message["content_blocks"])
    if type_ == "FunctionMessageChunk":
        return FunctionMessageChunk(content_blocks=message["content_blocks"])
    if type_ == "ToolMessageChunk":
        return ToolMessageChunk(content_blocks=message["content_blocks"])
    if type_ == "SystemMessageChunk":
        return SystemMessageChunk(content_blocks=message["content_blocks"])
    if type_ == "ChatMessageChunk":
        return ChatMessageChunk(content_blocks=message["content_blocks"])
    msg = f"Got unexpected message type: {type_}"
    raise ValueError(msg)

def _relay_messages_to_lc_messages(
        relay_messages: list[JsonObject],
        extract_system_message: bool) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """
    Convert a list of NeMo Relay message dicts to a list of LangChain `BaseMessage`s.
    when extract_system_message is True, the first system message encountered will be returned separately.
    """
    lc_messages: list[BaseMessage] = []
    system_message: SystemMessage | None = None
    for message in relay_messages:
        try:
            lc_message = _relay_message_to_lc_message(message)
            if extract_system_message and system_message is None and isinstance(lc_message, SystemMessage):
                system_message = lc_message
            else:
                lc_messages.append(lc_message)
        except Exception:
            _logger.error("Failed to convert relay message to LangChain message: %s", message, exc_info=True)

    return system_message, lc_messages

def _lc_messages_to_json(messages: list[BaseMessage]) -> list[JsonObject]:
    json_messages: list[JsonObject] = []
    for msg in messages:
        jm = {"type": msg.type, "content_blocks": msg.content_blocks}
        if msg.type == "tool":
            jm["tool_call_id"] = getattr(msg, "tool_call_id", "")

        json_messages.append(jm)

    return json_messages

def lc_model_request_to_relay_llm_request(model_name: str | None, request: ModelRequest[Any]) -> LLMRequest:
    """
    Serialize a LangChain ``ModelRequest`` instance into a NeMo Relay ``LLMRequest``.
    """
    lc_messages = []
    if request.system_message is not None:
        lc_messages.append(request.system_message)

    lc_messages.extend(request.messages)
    messages = _lc_messages_to_json(lc_messages)

    payload = {
        "messages": messages,
    }
    if model_name:
        payload["model"] = model_name
    if request.model_settings:
        payload["model_settings"] = request.model_settings
    if request.response_format is not None:
        payload["response_format"] = repr(request.response_format)
    return LLMRequest({}, payload)

def payload_to_model_request(
    lc_request: ModelRequest[Any],
    relay_request: LLMRequest,
) -> ModelRequest[Any]:
    """Apply supported NeMo Relay request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    relay_messages = relay_request.content["messages"]
    system_message, lc_messages = _relay_messages_to_lc_messages(relay_messages, extract_system_message=True)

    if system_message is not None:
        overrides["system_message"] = system_message

    overrides["messages"] = lc_messages

    if system_message is not None:
        overrides["system_message"] = system_message

    overrides["messages"] = lc_messages

    model_settings = relay_request.content.get("model_settings")
    if isinstance(model_settings, dict):
        # Using dict() to ensure we have a copy
        model_settings_copy = dict(model_settings)
        extra_headers = model_settings_copy.get("extra_headers")
        if not isinstance(extra_headers, dict):
            extra_headers = {}
        overrides["model_settings"] = model_settings_copy
    else:
        overrides["model_settings"] = {}
        extra_headers = {}

    if len(relay_request.headers) > 0:
        extra_headers.update(relay_request.headers)
        overrides["model_settings"]["extra_headers"] = extra_headers

    if "tool_choice" in relay_request.content:
        overrides["tool_choice"] = relay_request.content["tool_choice"]

    return lc_request.override(**overrides) if overrides else lc_request


def _model_response_payload(response: ModelResponse[Any], codec: Any) -> dict[str, Any]:
    messages: list[JsonObject] = _lc_messages_to_json(response.result)

    payload: JsonObject = {
        "messages": messages,
    }

    if response.structured_response is not None:
        payload["structured_response"] = codec.to_json(response.structured_response)

    return payload


def _model_response_payload_from_json(payload: Any) -> dict[str, Any]:
    try:
        response_payload = payload[LANGCHAIN_MODEL_RESPONSE_KEY]
    except KeyError:
        raise TypeError("expected serialized LangChain ModelResponse payload")

    if not isinstance(response_payload, dict):
        raise TypeError("expected serialized LangChain ModelResponse object")

    return response_payload


def _model_response_from_payload(payload: Any, codec: Any) -> ModelResponse[Any] | None:
    if not isinstance(payload, dict):
        return None

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return None

    _, lc_messages = _relay_messages_to_lc_messages(raw_messages, extract_system_message=False)

    structured_response = None
    if "structured_response" in payload:
        structured_response = codec.from_json(payload["structured_response"])

    mr = ModelResponse(
        result=lc_messages,
        structured_response=structured_response,
    )
    return mr


def model_response_to_json(response: ModelResponse[Any], codec: Any) -> Any:
    """Serialize ``ModelResponse`` without losing Python-only fields."""
    return {
        LANGCHAIN_MODEL_RESPONSE_KEY: _model_response_payload(response, codec),
    }


def _message_content_text(message: BaseMessage) -> str | None:
    return None
    content = message.content
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else None
    return str(content)


def _message_finish_reason(message: BaseMessage) -> str | dict[str, str] | None:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("finish_reason", "stop_reason"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return _FINISH_REASON_MAP.get(value, {"unknown": value})
    return None


def _message_usage(message: BaseMessage) -> dict[str, Any] | None:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None

    mapped: dict[str, Any] = {}
    for source, target in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            mapped[target] = value

    return mapped or None

def model_response_from_json(payload: Any, codec: Any) -> ModelResponse[Any]:
    """Deserialize a ``ModelResponse`` serialized by ``best_effort_model_response_to_json``."""
    payload_data = _model_response_payload_from_json(payload)
    return _model_response_from_payload(payload_data, codec)


def _prepare_lc_payloads(payload: Any) -> Any:
    """
    Convert a LangChain payload to a JSON-serializable structure

    Typically the entry point to this method is a LangChain dictionary containing LC message objects, and the returned
    dictionary should contain the same structure, but the values are JSON serializable representations
    """
    if isinstance(payload, dict):
        prepared = {}
        for key, value in payload.items():
            prepared[key] = _prepare_lc_payloads(value)
    elif isinstance(payload, list | tuple | set):
        prepared = []
        for value in payload:
            prepared.append(_prepare_lc_payloads(value))
    elif isinstance(payload, Command):
        prepared = {
            "type": "command",
            "command": {
                "graph": _prepare_lc_payloads(payload.graph),
                "update": _prepare_lc_payloads(payload.update),
                "resume": _prepare_lc_payloads(payload.resume),
                "goto": _prepare_lc_payloads(payload.goto),
            },
        }
    elif isinstance(payload, Send):
        prepared = {
            "type": "send",
            "send": {
                "node": payload.node,
                "arg": _prepare_lc_payloads(payload.arg),
            },
        }
    elif isinstance(payload, ToolMessage):
        prepared = {
            "type": "tool_message",
            "tool_call": {
                "name": payload.name,
                "id": payload.id,
                "tool_call_id": payload.tool_call_id,
                "content": payload.content,
            },
        }
    elif isinstance(payload, BaseMessage):
        prepared = {
            "type": "message",
            "message": messages_to_dict([payload]),
        }
    else:
        prepared = payload

    return prepared
