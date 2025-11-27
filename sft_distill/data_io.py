# sft_distill/data_io.py

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .models import EvaluatedData, KnowledgeSlice
from .utils import model_to_dict


def load_knowledge_slices(input_path: str) -> List[KnowledgeSlice]:
    """加载 JSONL 格式的知识切片"""
    slices: List[KnowledgeSlice] = []
    path = Path(input_path)

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                slices.append(
                    KnowledgeSlice(
                        id=data.get("id", f"slice_{line_num}"),
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                    )
                )
            except Exception as e:
                print(f"[data_io] 加载第 {line_num} 行失败: {e}")

    print(f"[data_io] 成功加载 {len(slices)} 个知识切片")
    return slices


def save_jsonl(items: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_generated_raw(generated_data: Iterable, path: Path) -> None:
    save_jsonl((model_to_dict(item) for item in generated_data), path)


def compute_avg_score(evaluated_data: List[EvaluatedData]) -> float:
    if not evaluated_data:
        return 0.0
    return float(np.mean([d.avg_score for d in evaluated_data]))
