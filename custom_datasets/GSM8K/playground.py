# %%
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
import random

# 전체 GSM8K 데이터셋 로드
dataset = load_dataset("gsm8k", "main")["train"]

# %%
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




"""
{"doc_id": 0, 

"doc": {"question": "A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section?", "answer": "The craft store made 36 / 3 = <<36/3=12>>12 sales in the fabric section.\nIt made 36 / 4 = <<36/4=9>>9 sales in the jewelry section.\nThus, there were 36 - 12 - 9 = <<36-12-9=15>>15 sales in the stationery section.\n#### 15"}, 

"target": "The craft store made 36 / 3 = <<36/3=12>>12 sales in the fabric section.\nIt made 36 / 4 = <<36/4=9>>9 sales in the jewelry section.\nThus, there were 36 - 12 - 9 = <<36-12-9=15>>15 sales in the stationery section.\n#### 15", 

"arguments": {"gen_args_0": {

"arg_0": "Question: A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section?", 

"arg_1": {"until": ["Question:", "</s>", "<|im_end|>"], "do_sample": false, "temperature": 0.0, "max_gen_toks": 2000}}}, 

"resps": [[" \nA) 12\nB) 1/2\nC) 18\nD) 24\nE) 36\nThe best answer is D. ... ]]
"""

"""
{"doc_id": 1, 

"doc": {"question": "Marcy is a makeup artist and has agreed to do some makeup for her friend's wedding. The only makeup she has limited stock of is lip gloss so she counts how many tubes she needs. Each tube of lip gloss will hold enough lip gloss for 3 people's makeup. Marcy decides to bring 6 tubs of lip gloss, each of which holds 2 tubes of lip gloss, and this will be the exact amount she needs for everyone's makeup. How many people is Marcy painting with makeup?", "answer": "Marcy is bringing 6 tubs of lip gloss * 2 tubes of lip gloss per tub of lip gloss = <<6*2=12>>12 tubes of lip gloss.\nSo she must be applying makeup to 12 tubes of lip gloss * 3 people per tube of lip gloss = <<12*3=36>>36 people.\n#### 36"}, 

"target": "Marcy is bringing 6 tubs of lip gloss * 2 tubes of lip gloss per tub of lip gloss = <<6*2=12>>12 tubes of lip gloss.\nSo she must be applying makeup to 12 tubes of lip gloss * 3 people per tube of lip gloss = <<12*3=36>>36 people.\n#### 36", 

"arguments": {"gen_args_0": {

"arg_0": "Question: A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section? The craft store made 36 / 3 = <<36/3=12>>12 sales in the fabric section.\nIt made 36 / 4 = <<36/4=9>>9 sales in the jewelry section.\nThus, there were 36 - 12 - 9 = <<36-12-9=15>>15 sales in the stationery section.\n#### 15\n\nQuestion: During one day, there are 4 boat trips through the lake. The boat can take up to 12 people during one trip. How many people can the boat transport in 2 days? During each boat trip, there can be 12 people onboard, so during 4 boat trips, there can be 4 * 12 = <<4*12=48>>48 people in total.\nDuring two days the boat can transport a total of 48 * 2 = <<48*2=96>>96 people.\n#### 96\n\nQuestion: Marcy is a makeup artist and has agreed to do some makeup for her friend's wedding. The only makeup she has limited stock of is lip gloss so she counts how many tubes she needs. Each tube of lip gloss will hold enough lip gloss for 3 people's makeup. Marcy decides to bring 6 tubs of lip gloss, each of which holds 2 tubes of lip gloss, and this will be the exact amount she needs for everyone's makeup. How many people is Marcy painting with makeup?", 

"arg_1": {"until": ["Question:", "</s>", "<|im_end|>"], "do_sample": false, "temperature": 0.0}}}, 

"resps": [[" 6 tubs of lip gloss, each of which holds 2 tubes of lip gloss, will be 6 * 2 = <<6*2=12>>12 tubes of lip gloss.\nEach tube of lip gloss will hold enough lip gloss for 3 people's makeup, so 12 tubes will hold enough lip gloss for 12 * 3 = <<12*3=36>>36 people.\n#### 36\n\n"]], 

"filtered_resps": ["36"]
"""



# %%
question_len = []
for x in dataset:
    if x["question"]:
        question_len.append(len(x["question"]))

plt.figure(figsize=(8,5))
plt.hist(question_len, 
         bins=range(min(question_len), max(question_len), 10))
plt.title("Distribution of 'question' Text Lengths")
plt.show()

# %%
answer_len = []
for x in dataset:
    if x["answer"]:
        answer_len.append(len(x["answer"]))

plt.figure(figsize=(8,5))
plt.hist(answer_len, 
         bins=range(min(answer_len), max(answer_len), 10))
plt.title("Distribution of 'answer' Text Lengths")
plt.show()

# %%
from transformers import AutoTokenizer

# LLaMA tokenizer 로드 (예: LLaMA-2 7B)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

question_len = []
answer_len = []
for x in dataset:
    question_len.append(len(tokenizer.encode(x["question"], add_special_tokens=False)))
    answer_len.append(len(tokenizer.encode(x["answer"], add_special_tokens=False)))

# %%
plt.figure(figsize=(8,5))
plt.hist(question_len, 
         bins=range(min(question_len), max(question_len), 1))
plt.title("Distribution of 'question' Token Lengths")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(answer_len, 
         bins=range(min(answer_len), max(answer_len), 1))
plt.title("Distribution of 'answer' Token Lengths")
plt.show()

# %%
qa_ratio = []
for x in dataset:
    question_l = len(tokenizer.encode(x["question"], add_special_tokens=False))
    answer_l = len(tokenizer.encode(x["answer"], add_special_tokens=False))
    qa_ratio.append(answer_l / question_l)

# %%
sorted_qa_ratio = sorted(qa_ratio)
plt.figure(figsize=(8,5))
plt.plot(sorted_qa_ratio)
plt.title("len(answer) / len(question) in ascending order")
plt.show()

# %%
n = 0
for x in dataset:
    question_l = len(tokenizer.encode(x["question"], add_special_tokens=False))
    answer_l = len(tokenizer.encode(x["answer"], add_special_tokens=False))
    ratio = answer_l / question_l
    if ratio >= 3.76:
        n += 1
        # print("- question:", x["question"])
        # print("- answer:", x["answer"])
        # print()
print(n)
# %%
n = 0
for x in dataset:
    question_l = len(tokenizer.encode(x["question"], add_special_tokens=False))
    answer_l = len(tokenizer.encode(x["answer"], add_special_tokens=False))
    ratio = answer_l / question_l
    if ratio <= 0.7472:
        n += 1
        # print("- question:", x["question"])
        # print("- answer:", x["answer"])
        # print()
print(n)

# %%
for x in dataset:
    if x["answer"].count("\n#### ") == 0:
        print(x)
# %%
random.seed(42)

# 조건에 맞는 test set과 그 외 나머지 데이터 분리
test_set = []
rest = []

for x in dataset:
    question_l = len(tokenizer.encode(x["question"], add_special_tokens=False))
    answer_l = len(tokenizer.encode(x["answer"], add_special_tokens=False))
    ratio = answer_l / question_l if question_l > 0 else 0  # ZeroDivision 방지
    if ratio >= 3.76:
        test_set.append(dict(x))
    else:
        rest.append(dict(x))

# rest 중 랜덤하게 10개 선택 (재현성을 원하면 seed 지정)
train_set = random.sample(rest, 10)

# JSON으로 저장
with open("gsm8k_filtered_over_376e-2_train.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)

with open("gsm8k_filtered_over_376e-2_test.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2, ensure_ascii=False)

print(f"✅ 저장 완료: train {len(train_set)}개, test {len(test_set)}개")

# %%
test_set = []
rest = []

for x in dataset:
    question_l = len(tokenizer.encode(x["question"], add_special_tokens=False))
    answer_l = len(tokenizer.encode(x["answer"], add_special_tokens=False))
    ratio = answer_l / question_l if question_l > 0 else 0  # ZeroDivision 방지
    if ratio <= 0.7472:
        test_set.append(dict(x))
    else:
        rest.append(dict(x))

# rest 중 랜덤하게 10개 선택 (재현성을 원하면 seed 지정)
train_set = random.sample(rest, 10)

# JSON으로 저장
with open("gsm8k_filtered_under_7472e-4_train.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)

with open("gsm8k_filtered_under_7472e-4_test.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2, ensure_ascii=False)

print(f"✅ 저장 완료: train {len(train_set)}개, test {len(test_set)}개")
# %%
import json

# 파일 경로
input_path = "gsm8k_filtered_over_376e-2_test.json"
output_path = "gsm8k_filtered_over_376e-2_test.json"

# 파일 읽기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# id 추가
for i, item in enumerate(data):
    item["id"] = i  # 또는 item["question_id"] = i

# 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ {len(data)}개 항목에 id가 추가되어 저장되었습니다.")
# %%
print("**Post-Apocalyptic Survival Log**\n\nIn the remnants of the old world, where entertainment is scarce and joy is a precious commodity, the settlement of New Haven just announced a rare event—a traveling band performing under the wounded sky. Tickets for this concert, a beacon of hope amid the ruins, are traded at 40 credits each, a considerable sum earned through barter and toil.\n\nMr. Benson, a seasoned survivor known for his resourcefulness, decided to gather enough tickets not just for himself but for his small community. He acquired twelve of these coveted passes. In an effort to encourage solidarity and generosity among survivors, the ticket barons offer a discount: for every ticket purchased beyond the tenth one, buyers receive a five percent reduction in the ticket’s price, as a reward for fostering collective morale.\n\nThus, Mr. Benson’s purchase involved a straightforward but vital calculation within the harsh economy of New Haven. The first ten tickets carried the full weight of 40 credits each, while the two additional tickets slipped under the mercy of the discount—each costing less by that five percent margin. This tiered pricing system embodies the barter economy’s subtle complexities, balancing incentive and fairness among the dwindling population.\n\nFaced with this structure, Mr. Benson must now tally the total cost of his purchase so he can negotiate the final transfer of credits with the ticket barons, ensuring that no extra burdens fall upon his group’s fragile resources. Every credit spent must be justified in the ledger of survival.\n\nGiven these conditions—the full price for ten tickets and a five percent discount applied to the two tickets beyond that—what is the total amount of credits Mr. Benson has to transfer to claim all twelve concert passes?")
# %%
