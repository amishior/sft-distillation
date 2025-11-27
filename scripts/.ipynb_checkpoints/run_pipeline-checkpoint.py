# scripts/run_pipeline.py

import argparse
import asyncio
import json
from pathlib import Path

from sft_distill import CONFIG_TEMPLATE, DatasetDistillationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT dataset distillation pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to knowledge_slices.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for distilled dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file to override default CONFIG_TEMPLATE",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    config = dict(CONFIG_TEMPLATE)
    if config_path:
        user_conf = json.loads(Path(config_path).read_text(encoding="utf-8"))
        # 简单 shallow merge，必要时可以做深度 merge
        config.update(user_conf)
    return config


async def async_main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pipeline = DatasetDistillationPipeline(config)
    await pipeline.run(args.input, args.output)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
