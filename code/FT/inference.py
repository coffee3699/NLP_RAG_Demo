import json
from tqdm import tqdm
from model_response import MyLocalLLM

source_file = '/home/ldy/NLP_RAG_Demo/data/demo_test_data.jsonl'
target_file = '/FT/BASIC_FT10_results.json'


def main():
    my_local_llm = MyLocalLLM("/home/ldy/NLP_RAG_Demo/code/FT/weight/lora_3.10")

    result = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Generating responses"):
            data = json.loads(line)
            pred = my_local_llm.complete(data['query'])
            result.append({'query': data["query"], 'predicted_answer': pred.text, 'ground_truth': data["answer"]})

    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
