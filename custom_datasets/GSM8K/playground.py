# %%
from datasets import load_dataset
import json

# 전체 GSM8K 데이터셋 로드
dataset = load_dataset("gsm8k", "main")["train"]

# 상위 100개 (train)와 그 다음 20개 (test) 선택
train_data = dataset.select(range(100))
test_data = dataset.select(range(100, 105))

train_data = [dict(row) for row in train_data]
test_data = [dict(row) for row in test_data]


# JSON 파일로 저장
with open("gsm8k_first_120.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("gsm8k_first_120_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)


print("✅ 저장 완료: gsm8k_first_100.json (train 100개 + test 20개)")

# %%
from datasets import load_dataset

# GSM8K 데이터셋 로드 및 일부 추출
dataset = load_dataset("gsm8k", "main")["train"].select(range(100))

# Hugging Face 포맷으로 저장
dataset.save_to_disk("gsm8k_first_100")
# %%
import json

with open("gsm8k_first_100.json", encoding="utf-8") as f:
    data = json.load(f)

print(data.keys())
# %%
print(len(data['train']), len(data['test']))  # 100, 20이 나와야 정상

# %%
from datasets import load_dataset

ds = load_dataset("json", data_files="gsm8k_first_100.json")

print(ds)             # DatasetDict({'train': ..., 'test': ...})
print(ds["test"][0])
# %%
