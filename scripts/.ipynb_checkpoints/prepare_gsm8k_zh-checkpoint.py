from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 1. 加载 HuggingFace 数据集
ds = load_dataset("meta-math/GSM8K_zh")["train"]  # 只有一个 split，内部用列 split 标记 train/test

df = ds.to_pandas()

# 2. 按列 split 划分 train / test
train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# 可选：过滤掉没有中文的问题/答案
for part in (train_df, test_df):
    mask = part["question_zh"].str.len().fillna(0) > 0
    mask &= part["answer_zh"].str.len().fillna(0) > 0
    part.drop(part.index[~mask], inplace=True)

def build_extra_info(row):
    return {
        # 这里的 key 名必须叫 question / answer，
        # 才能被你命令里的 prompt_dict_keys / response_dict_keys 正确取到
        "question": row["question_zh"],
        "answer": row["answer_zh"],

        # 下面是附加信息，可要可不要
        "question_en": row["question"],
        "answer_en": row["answer"],
        "answer_only": row["answer_only"],
        "split": row["split"],
    }

train_df["extra_info"] = train_df.apply(build_extra_info, axis=1)
test_df["extra_info"] = test_df.apply(build_extra_info, axis=1)

# 3. 只保留 extra_info 一列，和 verl 的 gsm8k 预处理格式对齐
train_out = train_df[["extra_info"]]
test_out = test_df[["extra_info"]]

# 4. 保存到你习惯的目录，例如 $HOME/data/gsm8k_zh
save_dir = Path("~/data/gsm8k_zh").expanduser()
save_dir.mkdir(parents=True, exist_ok=True)

train_out.to_parquet(save_dir / "train.parquet", index=False)
test_out.to_parquet(save_dir / "test.parquet", index=False)

print("Saved to:", save_dir)
