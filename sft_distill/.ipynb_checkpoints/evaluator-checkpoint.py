# sft_distill/evaluator.py

import asyncio
from datetime import datetime
from typing import Dict, List

from .llm_client import LLMAPIClient
from .models import QualityScore, SFTDataPoint
from .prompts import DEFAULT_EVAL_PROMPT
from .utils import extract_json_from_text


class QualityEvaluator:
    """多维度质量评估器"""

    def __init__(
        self,
        llm_clients: Dict[str, LLMAPIClient],
        eval_prompt: str = DEFAULT_EVAL_PROMPT,
    ):
        self.llm_clients = llm_clients
        self.eval_prompt = eval_prompt

    async def evaluate_single(
        self,
        sft_data: SFTDataPoint,
        client_name: str = "primary",
        max_tokens: int = 1024,
    ) -> QualityScore:
        """使用指定 LLM 评估单条数据，始终返回 QualityScore"""

        client = self.llm_clients[client_name]

        messages = [
            {
                "role": "system",
                "content": "You are a data quality evaluation expert.",
            },
            {
                "role": "user",
                "content": self.eval_prompt.format(
                    instruction=sft_data.instruction,
                    input=sft_data.input,
                    output=sft_data.output,
                ),
            },
        ]

        try:
            response = await client.chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )

            eval_result = extract_json_from_text(response)

            overall_score = float(eval_result["overall_score"])
            dimension_scores = {
                k: float(v) for k, v in eval_result["dimension_scores"].items()
            }

            return QualityScore(
                data_id=sft_data.id,
                evaluator_model=client.model,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                feedback=str(eval_result.get("feedback", "")),
                evaluate_time=datetime.now().isoformat(),
            )
        except Exception as e:
            print(f"[QualityEvaluator] 评估失败 data_id={sft_data.id} with {client.model}: {e}")
            return QualityScore(
                data_id=sft_data.id,
                evaluator_model=client.model,
                overall_score=0.0,
                dimension_scores={},
                feedback="评估失败",
                evaluate_time=datetime.now().isoformat(),
            )

    async def evaluate_multi(self, sft_data: SFTDataPoint) -> List[QualityScore]:
        """使用多个 LLM 评估单条数据"""
        tasks = [
            self.evaluate_single(sft_data, name)
            for name in self.llm_clients.keys()
        ]
        return await asyncio.gather(*tasks)
