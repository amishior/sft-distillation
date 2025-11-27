# sft_distill/generator.py

from datetime import datetime
from typing import Any, Dict, List, Optional

from .llm_client import LLMAPIClient
from .models import KnowledgeSlice, SFTDataPoint
from .prompts import DEFAULT_GENERATION_PROMPT
from .utils import extract_json_from_text


class SFTDataGenerator:
    """SFT 数据生成器"""

    def __init__(
        self,
        llm_client: LLMAPIClient,
        prompt_template: Optional[str] = None,
        max_tokens: int = 2048,
        temperature_base: float = 0.7,
        temperature_step: float = 0.1,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_GENERATION_PROMPT
        self.max_tokens = max_tokens
        self.temperature_base = temperature_base
        self.temperature_step = temperature_step

    async def generate_from_slice(
        self,
        slice_data: KnowledgeSlice,
        num_samples: int = 3,
    ) -> List[SFTDataPoint]:
        """从单个知识切片生成多条 SFT 数据"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates high-quality synthetic data.",
            },
            {
                "role": "user",
                "content": self.prompt_template.format(content=slice_data.content),
            },
        ]

        results: List[SFTDataPoint] = []

        for i in range(num_samples):
            temperature = self.temperature_base + i * self.temperature_step
            try:
                response = await self.llm_client.chat_completion(
                    messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                )

                data: Dict[str, Any] = extract_json_from_text(response)

                sft_data = SFTDataPoint(
                    id=f"{slice_data.id}_gen{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    instruction=data["instruction"],
                    input=data.get("input", "") or "",
                    output=data["output"],
                    source_id=slice_data.id,
                    source_content=slice_data.content[:200],  # 保存摘要
                    generate_time=datetime.now().isoformat(),
                    generator_model=self.llm_client.model,
                )
                results.append(sft_data)
            except Exception as e:
                print(f"[SFTDataGenerator] 生成数据失败 slice={slice_data.id}, i={i}: {e}")
                continue

        return results
