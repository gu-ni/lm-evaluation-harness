import os
import json
import time
from pathlib import Path
from openai import OpenAI
from instruction_template import INSTRUCTION_GSM8K, genres

instruction_dict = {
    "GSM8K": INSTRUCTION_GSM8K
}

# 기존 JSON 파일에서 이미 처리한 id 들을 읽음
def load_existing_json(path):
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            existing_ids = {item["id"] for item in data if "id" in item}
            return data, existing_ids
    except Exception as e:
        print(f"Error loading existing JSON: {e}")
        return [], set()

# GPT 호출
def call_gpt(client, prompt):
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14",
        input=[
            {
                "role": "developer", 
                "content": "You are an imaginative storyteller who follows instructions well."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=1.0,
    )
    return response.output_text.strip()

# 메인 실행
if __name__ == "__main__":
    dataset = "GSM8K"
    filtered_name = "gsm8k_filtered_over_376e-2_test.json"

    output_path = f"/home/work/users/PIL_ghj/LLM/math/lm-evaluation-harness/custom_datasets/{dataset}"
    input_path = os.path.join(output_path, filtered_name)

    # 저장 파일 이름 변경
    file_name = Path(filtered_name).with_stem(Path(filtered_name).stem + "_narrative_by_gpt")
    output_json_path = os.path.join(output_path, file_name)

    # OpenAI 클라이언트
    client = OpenAI()

    # 기존 데이터 로드
    existing_data, existing_ids = load_existing_json(output_json_path)

    # 입력 파일 처리
    with open(input_path, "r", encoding="utf-8") as infile:
        problems = json.load(infile)  # input도 json 리스트라고 가정

        for i, problem in enumerate(problems):
            try:
                qid = problem.get("id")
                print(f"\n[Logging] Starting {i}-th Problem (ID: {qid})...")

                if qid in existing_ids:
                    print(f"[Logging] Skipping already processed id: {qid}")
                    continue

                genre = genres[i % len(genres)]
                problem["genre"] = genre
                instruction = instruction_dict[dataset].replace("{GENRE}", genre)
                input_prompt = instruction + problem["question"]

                new_question = call_gpt(client, input_prompt)
                print("\n------------------------- GPT Response -------------------------\n")
                print(new_question)
                print("\n----------------------------------------------------------------\n")

                problem["question"] = new_question
                existing_data.append(problem)
                existing_ids.add(qid)

                # 💾 매 반복마다 저장
                with open(output_json_path, "w", encoding="utf-8") as out:
                    json.dump(existing_data, out, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[Error] Skipping problem due to error: {e}")
                continue

            time.sleep(1)  # 속도 제한