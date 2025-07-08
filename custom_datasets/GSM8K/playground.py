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