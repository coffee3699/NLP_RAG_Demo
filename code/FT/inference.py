import json
from tqdm import tqdm

from model_response import MyLocalLLM

source_file = '/home/ldy/NLP_RAG_Demo/data/crag_data_200.jsonl'
target_file = '/home/ldy/NLP_RAG_Demo/data/crag_200_result.jsonl'


def main():
    my_local_llm = MyLocalLLM("/home/ldy/NLP_RAG_Demo/code/FT/weight/lora_2")

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in tqdm(lines, desc="Generating responses"):
        data = json.loads(line)

        pred = my_local_llm.complete(data['query'])

        result.append(json.dumps({'query': data["query"], 'answer': data["answer"], 'pred': pred.text},
                                 ensure_ascii=False) + '\n')

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(''.join(result))


if __name__ == '__main__':
    main()
