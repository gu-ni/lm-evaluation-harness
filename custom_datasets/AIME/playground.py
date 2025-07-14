# %%
import json
from datasets import load_dataset

dataset = load_dataset("AI-MO/aimo-validation-aime")["train"]

# %%
test_set = []
train_set = []

for x in dataset:
    x["question"] = x["problem"]
    x["answer"] = int(x["answer"])
    x.pop("problem")
    test_set.append(dict(x))
    train_set.append(dict(x))

# JSON으로 저장
with open("aime_train.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)

with open("aime_test.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2, ensure_ascii=False)

print(f"✅ 저장 완료: train {len(train_set)}개, test {len(test_set)}개")

# %%
len(dataset["train"])

# %%
dataset["train"][0]

# %%
print(dataset["train"][0].keys())
# dict_keys(['id', 'problem', 'solution', 'answer', 'url'])

# %%
for x in dataset["train"]:
    pass

# %%
print(dataset["train"][0]["problem"])
print()
print(dataset["train"][0]["solution"])
print()
print(dataset["train"][0]["answer"])

# %%
print(type(dataset["train"][0]["answer"]))
# %%
int('001')
# %%
