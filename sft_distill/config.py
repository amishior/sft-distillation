# sft_distill/config.py

CONFIG_TEMPLATE = {
    # 生成器配置
    "generator": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "model": "qwen3-max",
        "max_concurrent": 5,
    },
    # 评估器配置（支持多个）
    "evaluators": {
        "primary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "Moonshot-Kimi-K2-Instruct",
            "max_concurrent": 3,
        },
        "secondary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "deepseek-v3.2-exp",
            "max_concurrent": 3,
        },
    },
    # 工作流配置
    "samples_per_slice": 3,       # 每个知识切片生成多少条数据
    "min_score_threshold": 7.0,   # 质量阈值
    "min_votes": 2,               # 至少需要几个评估器评分
    "train_val_split": 0.9,       # 训练集比例
    "enable_dedup": True,         # 是否启用去重
    "output_dir": "./distilled_dataset",
}
