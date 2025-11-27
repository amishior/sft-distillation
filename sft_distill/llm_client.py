# sft_distill/llm_client.py

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp


class LLMAPIClient:
    """统一的 LLM API 客户端，支持 OpenAI 格式和自定义 API"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_base: str = config.get("api_base", "")
        self.api_key: str = config.get("api_key", "")
        self.model: str = config.get("model", "")
        self.max_concurrent: int = config.get("max_concurrent", 10)

        # 并发控制
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """统一的聊天完成接口"""
        if self.session is None:
            raise RuntimeError(
                "ClientSession 未初始化，请使用 'async with LLMAPIClient(...)' 进行调用"
            )

        async with self.semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                # 支持自定义 API 格式
                if "custom_format" in self.config:
                    payload = self._apply_custom_format(payload)

                async with self.session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                if "custom_parser" in self.config:
                    return self.config["custom_parser"](result)

                return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[LLMAPIClient] API 调用失败: {e}")
                raise

    def _apply_custom_format(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """应用自定义格式转换（可按需扩展）"""
        # TODO: 根据具体 API 标准化参数
        return payload
