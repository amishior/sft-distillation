# sft_distill/filtering.py

from typing import Callable, Iterable, List, Optional

import numpy as np

from .models import EvaluatedData


class DataFilter:
    """数据筛选和去重模块"""

    def __init__(
        self,
        min_score_threshold: float = 7.0,
        min_votes: int = 2,
        embed_fn: Optional[Callable[[Iterable[str]], np.ndarray]] = None,
    ):
        """
        embed_fn: 可选的句向量函数，输入文本列表，输出 shape [N, D] 的 ndarray。
        不提供时使用 Jaccard 文本相似作为简单替代。
        """
        self.min_score_threshold = min_score_threshold
        self.min_votes = min_votes
        self.embed_fn = embed_fn

    def filter_by_score(self, evaluated_data: List[EvaluatedData]) -> List[EvaluatedData]:
        """基于平均分和评估器数量筛选高质量数据"""
        filtered = [
            data
            for data in evaluated_data
            if data.avg_score >= self.min_score_threshold
            and len(data.scores) >= self.min_votes
        ]
        print(f"[DataFilter] 筛选后保留 {len(filtered)}/{len(evaluated_data)} 条数据")
        return filtered

    def _dedup_jaccard(
        self,
        data_list: List[EvaluatedData],
        threshold: float,
    ) -> List[EvaluatedData]:
        """基于指令 Jaccard 相似度的去重实现（降级方案）"""

        def similarity(a: str, b: str) -> float:
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            union = len(set_a | set_b)
            return len(set_a & set_b) / union if union > 0 else 0.0

        unique_data: List[EvaluatedData] = []
        seen_instructions: List[str] = []

        for data in sorted(data_list, key=lambda x: x.avg_score, reverse=True):
            inst = data.data.instruction
            if not seen_instructions:
                unique_data.append(data)
                seen_instructions.append(inst)
                continue

            is_duplicate = any(
                similarity(inst, seen) > threshold for seen in seen_instructions
            )
            if not is_duplicate:
                unique_data.append(data)
                seen_instructions.append(inst)

        print(f"[DataFilter]（Jaccard）去重后保留 {len(unique_data)}/{len(data_list)} 条数据")
        return unique_data

    def _dedup_embedding(
        self,
        data_list: List[EvaluatedData],
        threshold: float,
    ) -> List[EvaluatedData]:
        """基于句向量余弦相似度的去重实现"""

        instructions = [d.data.instruction for d in data_list]
        embeddings = self.embed_fn(instructions)  # type: ignore[arg-type]

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(data_list):
            raise ValueError("embed_fn 返回的形状不合法，需为 [N, D]")

        def cos_sim(v, m):
            # v: [D], m: [K, D]
            v_norm = np.linalg.norm(v) + 1e-8
            m_norm = np.linalg.norm(m, axis=1) + 1e-8
            return (m @ v) / (m_norm * v_norm)

        selected_indices: List[int] = []
        selected_vecs: List[np.ndarray] = []

        sorted_indices = sorted(
            range(len(data_list)),
            key=lambda i: data_list[i].avg_score,
            reverse=True,
        )

        for idx in sorted_indices:
            v = embeddings[idx]
            if not selected_vecs:
                selected_indices.append(idx)
                selected_vecs.append(v)
                continue

            sims = cos_sim(v, np.stack(selected_vecs, axis=0))
            if sims.max() < threshold:
                selected_indices.append(idx)
                selected_vecs.append(v)

        unique_data = [data_list[i] for i in selected_indices]
        print(f"[DataFilter]（Embedding）去重后保留 {len(unique_data)}/{len(data_list)} 条数据")
        return unique_data

    def deduplicate_by_semantic_similarity(
        self,
        data_list: List[EvaluatedData],
        threshold: float = 0.85,
    ) -> List[EvaluatedData]:
        """基于指令语义相似度去重"""

        if not data_list:
            return []

        if self.embed_fn is None:
            return self._dedup_jaccard(data_list, threshold)
        else:
            return self._dedup_embedding(data_list, threshold)
