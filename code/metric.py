# -*- coding: UTF-8 -*-
"""
@File    ：metric.py
@Author  ：zfk & czm
@Date    ：2024/5/9 16:15
"""
import json
import time

import nltk
import torch
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Setting the path to the model
checkpoint = ""
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.config.pad_token_id = model.config.eos_token_id

# Define the instruction for semantic comparison
instruction = ("Determine if the following two text snippets convey the same meaning. Answer 'Yes' if they are "
               "semantically similar, otherwise answer 'No':")


def metric(pred: str, answer_list: list[str]):
    # 计算BLEU
    # print(pred, answer_list)
    reference = [answer.split() for answer in answer_list]
    candidate = pred.split()
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # 计算ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred, ' '.join(answer_list))

    return bleu_score, rouge_scores[0]['rouge-l']


# Helper function to calculate average excluding -1
def calculate_average(scores):
    valid_scores = [score for score in scores if score != -1]
    return sum(valid_scores) / len(valid_scores) if valid_scores else -1


if __name__ == '__main__':
    start = time.time()
    with open('crag_200_result.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bleu_scores = []
    rouge_l_p = []
    rouge_l_r = []
    rouge_l_f = []
    lamini_scores = []
    lamini_scores_per_query = []
    output_file = "crag_200_evaluate_lamini.jsonl"
    with open(output_file, 'w', encoding='utf-8') as out:
        for line in tqdm(lines):
            data = json.loads(line)
            query = data['query']
            answer = data['answer']
            pred = data['pred']
            if pred == "":
                pred = "invalid question"
            bleu, rouge_l = metric(pred.lower().strip(), [answer.lower().strip()])
            bleu_scores.append(bleu)
            rouge_l_p.append(rouge_l['p'])
            rouge_l_r.append(rouge_l['r'])
            rouge_l_f.append(rouge_l['f'])

            # # Formulate the input prompt
            input_prompt = (f"### Instruction:\n{instruction}\n\n### Text 1:\n{answer}\n\n### Text 2:\n{pred}\n\n### "
                            f"Response:")

            # 构造8个相同的输入以进行并行处理
            input_prompts = [input_prompt] * 8
            # Tokenize inputs in batch
            inputs = tokenizer(input_prompts, return_tensors="pt", padding=True).to(device)
            # 使用模型进行批处理
            outputs = model.generate(**inputs, max_new_tokens=2, do_sample=True)
            # 解码并提取生成的文本
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [text[len(input_prompt):] for text in generated_texts]
            # 打印生成的文本
            lamini_scores = []
            for i, text in enumerate(generated_texts):
                # print(f"Response {i + 1}: {text}")
                if "Yes" in text:
                    lamini_scores.append(1)
                elif "No" in text:
                    lamini_scores.append(0)
                else:
                    lamini_scores.append(-1)
            lamini_scores_per_query.append(lamini_scores)

            output_data = {
                "query": query,
                "answer": answer,
                "pred": pred,
                "lamini_result": lamini_scores,
            }
            json.dump(output_data, out)
            out.write('\n')
            out.flush()

        # 计算8组平均分，过滤掉-1
        average_scores = [calculate_average(scores) for scores in zip(*lamini_scores_per_query)]
        # 计算8组平均分的平均分，并过滤掉-1的组平均分
        valid_average_scores = [score for score in average_scores if score != -1]
        final_average_score = sum(valid_average_scores) / len(valid_average_scores) if valid_average_scores else 0
        # print("Average Scores for each batch of 8 responses:", average_scores)
        print("lamini_scores:", final_average_score)

        # print(f"lamini_scores: {sum(lamini_scores) / len(lamini_scores)}")
        print(f"BLEU: {sum(bleu_scores) / len(bleu_scores)}")
        print(f"ROUGE-L P: {sum(rouge_l_p) / len(rouge_l_p)}")
        print(f"ROUGE-L R: {sum(rouge_l_r) / len(rouge_l_r)}")
        print(f"ROUGE-L F: {sum(rouge_l_f) / len(rouge_l_f)}")
        end = time.time()
        print('程序运行时间为: %s Seconds' % (end - start))
