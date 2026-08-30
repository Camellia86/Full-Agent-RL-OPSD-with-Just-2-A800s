"""Function-calling reward used by the teacher-free OPSD stage."""

from __future__ import annotations

import collections
import json
import re
from typing import Any

import torch
from rouge_score import rouge_scorer


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _parse_label(label: str) -> tuple[list[dict[str, Any]], str | None]:
    if "</think>" not in label:
        raise ValueError("label does not contain </think>")
    answer = label.split("</think>", 1)[1].strip()
    calls: list[dict[str, Any]] = []
    if "<tool_call>" in answer:
        for payload in re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL):
            call = json.loads(payload)
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
        return calls, None
    return calls, answer


def _parse_generation(content: str) -> tuple[list[dict[str, Any]], str | None]:
    thoughts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    if len(thoughts) != 1:
        raise ValueError("generation must contain exactly one <think>...</think> block")
    answer = content.split("</think>", 1)[1].strip()
    calls: list[dict[str, Any]] = []
    if "<tool_call>" in answer:
        for payload in re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL):
            call = json.loads(payload)
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
        return calls, None
    return calls, answer


def _generation_from_query(query: str) -> str:
    return query.split("<|im_start|>assistant\n")[-1]


def _extract_tools(prompt: str) -> dict[str, dict[str, Any]]:
    blocks = re.findall(r"<tools>(.*?)</tools>", prompt, re.DOTALL)
    if not blocks:
        return {}
    raw = blocks[-1].strip()
    try:
        parsed = json.loads(raw)
        tool_objects = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        tool_objects = [json.loads(line) for line in raw.splitlines() if line.strip()]

    tools: dict[str, dict[str, Any]] = {}
    for tool in tool_objects:
        if not tool:
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tool = tool["function"]
        parameters = tool.get("parameters", tool.get("inputSchema", {}))
        if isinstance(parameters, dict) and isinstance(parameters.get("properties"), dict):
            parameters = parameters["properties"]
        tools[tool["name"]] = parameters if isinstance(parameters, dict) else {}
    return tools


def _calls_are_valid(calls: list[dict[str, Any]], tools: dict[str, dict[str, Any]]) -> bool:
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments", {})
        if name not in tools or not isinstance(arguments, dict):
            return False
        if any(key not in tools[name] for key in arguments):
            return False
    return True


def _text_score(prediction: str, target: str) -> float:
    if prediction == target:
        return 1.0
    if not prediction.strip() or not target.strip():
        return 0.0
    return float(_ROUGE.score(target, prediction)["rougeL"].fmeasure)


def _argument_score(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    remaining = dict(target)
    intersection = 0.0
    for key, predicted_value in predicted.items():
        if key not in remaining:
            continue
        target_value = remaining.pop(key)
        if isinstance(predicted_value, str) and isinstance(target_value, str):
            intersection += _text_score(predicted_value, target_value)
        elif isinstance(predicted_value, (int, float, bool)) and isinstance(target_value, (int, float, bool)):
            intersection += float(predicted_value == target_value)
        else:
            intersection += float(str(predicted_value) == str(target_value))
    union = len(predicted) + len(remaining)
    return intersection / union if union else 1.0


def _pop_best_match(bucket: list[dict[str, Any]], predicted: dict[str, Any]) -> float:
    scores = [_argument_score(predicted.get("arguments", {}), item.get("arguments", {})) for item in bucket]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_index]
    bucket.pop(best_index)
    return best_score


def _function_call_score(target_calls: list[dict[str, Any]], predicted_calls: list[dict[str, Any]]) -> float:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for call in target_calls:
        buckets[call["name"]].append(call)

    intersection = 0.0
    for call in predicted_calls:
        bucket = buckets.get(call["name"])
        if bucket:
            intersection += _pop_best_match(bucket, call)
    union = sum(len(bucket) for bucket in buckets.values()) + len(predicted_calls)
    return intersection / union if union else 1.0


def reward_func(queries, prompts, labels):
    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []

    for query, prompt, label in zip(queries, prompts, labels):
        target_calls, target_reply = _parse_label(label)
        tools = _extract_tools(prompt)
        try:
            predicted_calls, predicted_reply = _parse_generation(_generation_from_query(query))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            rewards.append(-1.0)
            format_rewards.append(-1.0)
            answer_rewards.append(0.0)
            continue

        if not _calls_are_valid(predicted_calls, tools):
            rewards.append(-1.0)
            format_rewards.append(-1.0)
            answer_rewards.append(0.0)
            continue

        format_rewards.append(1.0)
        if target_calls:
            score = _function_call_score(target_calls, predicted_calls) if predicted_calls else 0.0
        else:
            score = _text_score(predicted_reply or "", target_reply or "")
        rewards.append(score)
        answer_rewards.append(score)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    format_tensor = torch.tensor(format_rewards, dtype=torch.float32)
    answer_tensor = torch.tensor(answer_rewards, dtype=torch.float32)
    return {
        "rewards": reward_tensor,
        "scores": answer_tensor,
        "extra_logs": {
            "format_rewards": format_tensor,
            "answer_rewards": answer_tensor,
        },
    }
