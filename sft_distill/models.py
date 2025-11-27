# sft_distill/models.py

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class KnowledgeSlice(BaseModel):
    """输入的领域知识切片"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SFTDataPoint(BaseModel):
    """生成的 SFT 数据点"""
    id: str
    instruction: str
    input: str
    output: str
    source_id: str
    source_content: str
    generate_time: str
    generator_model: str


class QualityScore(BaseModel):
    """质量评估分数"""
    data_id: str
    evaluator_model: str
    overall_score: float
    dimension_scores: Dict[str, float]
    feedback: str
    evaluate_time: str


class EvaluatedData(BaseModel):
    """带评估结果的完整数据"""
    data: SFTDataPoint
    scores: List[QualityScore]
    avg_score: float
    is_high_quality: bool

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat()
