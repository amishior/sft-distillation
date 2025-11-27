# sft_distill/utils.py

import json
import re
from typing import Any, Dict

from pydantic import BaseModel


JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 输出中尽量稳地抽取 JSON 对象"""
    text = text.strip()

    # 先尝试整体解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 再尝试匹配第一段 JSON
    match = JSON_PATTERN.search(text)
    if not match:
        raise ValueError("未在文本中找到 JSON 片段")

    snippet = match.group()
    return json.loads(snippet)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """兼容 Pydantic v1/v2 的序列化"""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # type: ignore[call-arg]
