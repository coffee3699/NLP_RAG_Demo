import json

import nltk
import torch
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_file = "/home/ldy/NLP_RAG_Demo/data/crag_200_result.jsonl"
output_file = "crag_200_evaluate.jsonl"


def metric(pred: str, answer: str):
    print(pred, answer)

    # Check if prediction meets a minimum requirement before scoring
    if pred in ["invalid question", "", None]:
        return 0, {'p': 0, 'r': 0, 'f': 0}

    # Calculate BLEU
    reference = answer.split()
    candidate = pred.split()
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate ROUGE
    rouge = Rouge()

    if pred.strip():
        try:
            rouge_scores = rouge.get_scores(pred, answer)
            return bleu_score, rouge_scores[0]['rouge-l']
        except ValueError as e:  # Catching ValueError explicitly
            print(f"Error computing ROUGE for '{pred}': {e}")
            return bleu_score, {'p': 0, 'r': 0, 'f': 0}  # Return zeros for ROUGE in case of error
    else:
        return bleu_score, {'p': 0, 'r': 0, 'f': 0}


# Helper function to calculate average excluding -1
def calculate_average(scores):
    valid_scores = [score for score in scores if score != -1]
    return sum(valid_scores) / len(valid_scores) if valid_scores else -1


def main():
    # Setting the path to the model
    checkpoint = "/data/zzy/Models/LaMini-GPT-774M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    with open(eval_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bleu_scores = []
    rouge_l_p = []
    rouge_l_r = []
    rouge_l_f = []
    lamini_scores_per_query = []

    with open(output_file, 'w', encoding='utf-8') as out:
        for line in tqdm(lines, desc="Evaluating"):
            data = json.loads(line)
            query = data['query']
            answer = data['answer']
            pred = data['pred']
            if pred == "":
                pred = "invalid question"
            bleu, rouge_l = metric(pred.lower().strip(), answer.lower().strip())
            bleu_scores.append(bleu)
            rouge_l_p.append(rouge_l['p'])
            rouge_l_r.append(rouge_l['r'])
            rouge_l_f.append(rouge_l['f'])

            # Formulate the input prompt
            instruction = (
                "Determine if the following two text snippets convey the same meaning. Answer 'Yes' if they are "
                "semantically similar, otherwise answer 'No':")
            input_prompt = (f"### Instruction:\n{instruction}\n\n### Text 1:\n{answer}\n\n### Text 2:\n{pred}\n\n### "
                            f"Response:")

            # 构造8个相同的输入以进行并行处理
            input_prompts = [input_prompt] * 8
            # Tokenize inputs in batch
            inputs = tokenizer(input_prompts, return_tensors="pt", padding=True).to(device)
            # 使用模型进行批处理
            outputs = model.generate(**inputs, max_new_tokens=2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            # 解码并提取生成的文本
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [text[len(input_prompt):] for text in generated_texts]

            lamini_scores = []
            for text in generated_texts:
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

        print("lamini_scores:", final_average_score)
        print(f"BLEU: {sum(bleu_scores) / len(bleu_scores)}")
        print(f"ROUGE-L P: {sum(rouge_l_p) / len(rouge_l_p)}")
        print(f"ROUGE-L R: {sum(rouge_l_r) / len(rouge_l_r)}")
        print(f"ROUGE-L F: {sum(rouge_l_f) / len(rouge_l_f)}")


if __name__ == '__main__':
    main()