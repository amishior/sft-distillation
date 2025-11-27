# sft_distill/pipeline.py

import asyncio
import json
import random
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data_io import (
    compute_avg_score,
    load_knowledge_slices,
    save_generated_raw,
    save_jsonl,
)
from .filtering import DataFilter
from .generator import SFTDataGenerator
from .llm_client import LLMAPIClient
from .models import EvaluatedData, SFTDataPoint
from .evaluator import QualityEvaluator


class DatasetDistillationPipeline:
    """完整的数据集蒸馏工作流"""

    def __init__(self, config: Dict[str, Any]):
        # 初始化生成器
        self.generator_client = LLMAPIClient(config["generator"])
        self.generator = SFTDataGenerator(
            self.generator_client,
            prompt_template=config.get("generation_prompt"),
            max_tokens=config.get("generation_max_tokens", 2048),
            temperature_base=config.get("temperature_base", 0.7),
            temperature_step=config.get("temperature_step", 0.1),
        )

        # 初始化评估器
        self.evaluator_clients: Dict[str, LLMAPIClient] = {
            name: LLMAPIClient(client_config)
            for name, client_config in config["evaluators"].items()
        }
        self.evaluator = QualityEvaluator(self.evaluator_clients)

        # 初始化过滤器（可注入 embed_fn 做向量去重）
        self.filter = DataFilter(
            min_score_threshold=config.get("min_score_threshold", 7.0),
            min_votes=config.get("min_votes", 2),
            embed_fn=config.get("embed_fn"),  # 可选，默认 None
        )

        # 配置参数
        self.samples_per_slice: int = config.get("samples_per_slice", 3)
        self.train_val_split: float = config.get("train_val_split", 0.9)
        self.enable_dedup: bool = config.get("enable_dedup", True)

        # 存储路径
        self.output_dir = Path(config.get("output_dir", "./distilled_dataset"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 批次大小
        self.gen_batch_size: int = config.get("gen_batch_size", 10)
        self.eval_batch_size: int = config.get("eval_batch_size", 20)

    # ---------- 生成阶段 ----------

    async def generate_batch(
        self,
        slices: List,
        batch_size: Optional[int] = None,
    ) -> List[SFTDataPoint]:
        """批量生成 SFT 数据"""
        batch_size = batch_size or self.gen_batch_size
        all_generated: List[SFTDataPoint] = []

        async with self.generator_client:
            total_batches = (len(slices) - 1) // batch_size + 1 if slices else 0
            for i in range(0, len(slices), batch_size):
                batch = slices[i : i + batch_size]
                print(f"[Pipeline] 生成批次 {i // batch_size + 1}/{total_batches}")

                tasks = [
                    self.generator.generate_from_slice(slice_data, self.samples_per_slice)
                    for slice_data in batch
                ]

                batch_results = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"[Pipeline] 生成任务异常: {result}")
                        continue
                    if isinstance(result, list):
                        all_generated.extend(result)

        print(f"[Pipeline] 总共生成 {len(all_generated)} 条原始数据")
        return all_generated

    # ---------- 评估阶段 ----------

    async def evaluate_batch(
        self,
        data_list: List[SFTDataPoint],
        batch_size: Optional[int] = None,
    ) -> List[EvaluatedData]:
        """批量评估生成数据"""
        batch_size = batch_size or self.eval_batch_size
        evaluated_data: List[EvaluatedData] = []

        async with AsyncExitStack() as stack:
            # 打开所有评估器的 session
            for client in self.evaluator_clients.values():
                await stack.enter_async_context(client)

            total_batches = (len(data_list) - 1) // batch_size + 1 if data_list else 0

            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                print(f"[Pipeline] 评估批次 {i // batch_size + 1}/{total_batches}")

                tasks = [
                    self.evaluator.evaluate_multi(data)
                    for data in batch
                ]

                batch_scores = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                for data, scores in zip(batch, batch_scores):
                    if isinstance(scores, Exception):
                        print(f"[Pipeline] 评估任务异常 data_id={data.id}: {scores}")
                        evaluated_data.append(
                            EvaluatedData(
                                data=data,
                                scores=[],
                                avg_score=0.0,
                                is_high_quality=False,
                            )
                        )
                        continue

                    valid_scores = [s for s in scores if s is not None]
                    if not valid_scores:
                        evaluated_data.append(
                            EvaluatedData(
                                data=data,
                                scores=[],
                                avg_score=0.0,
                                is_high_quality=False,
                            )
                        )
                        continue

                    avg_score = float(
                        np.mean([s.overall_score for s in valid_scores])
                    )

                    evaluated_data.append(
                        EvaluatedData(
                            data=data,
                            scores=valid_scores,
                            avg_score=avg_score,
                            is_high_quality=avg_score
                            >= self.filter.min_score_threshold,
                        )
                    )

        print(f"[Pipeline] 完成评估，共 {len(evaluated_data)} 条数据")
        return evaluated_data

    # ---------- 划分 & 保存 ----------

    def _convert_to_sharegpt(
        self, data_list: List[EvaluatedData]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in data_list:
            conv = {
                "id": item.data.id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{item.data.instruction}\n\n{item.data.input}".strip(),
                    },
                    {"from": "gpt", "value": item.data.output},
                ],
                "avg_score": item.avg_score,
                "dimension_scores": {
                    score.evaluator_model: score.dimension_scores
                    for score in item.scores
                },
            }
            items.append(conv)
        return items

    def save_splits(
        self,
        high_quality_data: List[EvaluatedData],
        train_ratio: float,
    ) -> None:
        """保存训练集和评测集"""
        random.shuffle(high_quality_data)

        train_size = int(len(high_quality_data) * train_ratio)
        train_data = high_quality_data[:train_size]
        val_data = high_quality_data[train_size:]

        train_items = self._convert_to_sharegpt(train_data)
        val_items = self._convert_to_sharegpt(val_data)

        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        save_jsonl(train_items, train_path)
        save_jsonl(val_items, val_path)

        metadata = {
            "total_samples": len(high_quality_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "avg_score": compute_avg_score(high_quality_data),
            "generator_model": self.generator_client.model,
            "evaluator_models": [
                client.model for client in self.evaluator_clients.values()
            ],
            "min_score_threshold": self.filter.min_score_threshold,
            "train_ratio": train_ratio,
            "creation_time": datetime.now().isoformat(),
        }

        meta_path = self.output_dir / "metadata.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[Pipeline] 数据已保存至 {self.output_dir}")
        print(f"[Pipeline] 训练集: {len(train_data)} 条")
        print(f"[Pipeline] 评测集: {len(val_data)} 条")
        print(f"[Pipeline] 平均质量分数: {metadata['avg_score']:.2f}")

    # ---------- 总流程 ----------

    async def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> List[EvaluatedData]:
        """执行完整工作流"""

        if output_path:
            self.output_dir = Path(output_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("开始数据集蒸馏工作流")
        print("=" * 60)

        # Step 1: 加载数据
        print("\n[Step 1] 加载知识切片...")
        slices = load_knowledge_slices(input_path)

        # Step 2: 生成 SFT 数据
        print("\n[Step 2] 生成 SFT 数据...")
        generated_data = await self.generate_batch(slices)

        # 保存中间结果
        raw_path = self.output_dir / "generated_raw.jsonl"
        save_generated_raw(generated_data, raw_path)

        # Step 3: 质量评估
        print("\n[Step 3] 多 LLM 质量评估...")
        evaluated_data = await self.evaluate_batch(generated_data)

        # Step 4: 筛选与去重
        print("\n[Step 4] 筛选高质量数据...")
        high_quality = self.filter.filter_by_score(evaluated_data)

        if self.enable_dedup:
            print("\n[Step 5] 语义去重...")
            high_quality = self.filter.deduplicate_by_semantic_similarity(high_quality)
        else:
            print("\n[Step 5] 跳过去重...")

        # Step 6: 划分训练/评测集
        print("\n[Step 6] 划分训练集和评测集...")
        self.save_splits(high_quality, self.train_val_split)

        print("\n" + "=" * 60)
        print("数据集蒸馏完成！")
        print("=" * 60)

        return high_quality
