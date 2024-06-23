import json

from llama_index.core import Document


def load_documents(file_path):
    """Load documents from jsonl eval dataset file, return a list of llama_index Document objects"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_objects = [json.loads(line) for line in f]

    contents = []
    for obj in json_objects:
        for result in obj.get('search_results', []):
            page_result = result.get('page_result', '').strip()
            if page_result:
                contents.append(page_result)
            else:
                print(f"Empty page_result found in {result['page_url']}")

    documents = [Document(text=content) for content in contents]
    if not documents:
        raise ValueError("No valid documents found. Please check the input file for valid content.")

    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_question_and_answer(file_path):
    """load question and answer from jsonl file, return a tuple of two lists, one for questions and one for answers"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_objects = [json.loads(line) for line in f]

    questions = [obj['query'] for obj in json_objects]
    answers = [obj['answer'] for obj in json_objects]

    return questions, answers


def eval_on_dataset(query_engine, data_path, results_path):
    """Evaluate the llama_index on the given dataset file"""
    # Load the questions and answers
    questions, answers = load_question_and_answer(data_path)

    # Query each question and store results
    results = []
    for question, answer in zip(questions, answers):
        response = query_engine.query(question)
        result = {
            'query': question,
            'predicted_answer': str(response),
            'ground_truth': answer,
            'retrieved_chunk': [node.text for node in response.source_nodes],
        }
        results.append(result)

    # Save results to a JSON file
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print('Results saved to', results_path)

if __name__ == '__main__':
    documents = load_documents('/home/ldy/NLP_RAG_Demo/data/test_data.jsonl')
    print(documents[:2])
    print(len(documents))

    # queries = load_question_and_answer('/home/ldy/NLP_RAG_Demo/data/test_data.jsonl')
    # print(queries[:2])
    # print(len(queries))
