"""
No special indexing, retrieval, or postprocessing (rerank) techniques are used in this script.
"""
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from model_response import MyApiLLM, MyLocalLLM
from utils import eval_on_dataset, load_documents

load_dotenv()

# Global configuration variables
MODEL_TYPE = 'local'  # 'local' or 'api'
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH')
DATA_PATH = os.getenv('EVAL_DATASET_PATH')
RESULTS_PATH = 'BASIC_results.json'


def main():
    # Load the documents
    documents = load_documents(DATA_PATH)

    # Parse the documents into nodes
    node_parser = SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)

    # Create a vector store index from the nodes
    index = VectorStoreIndex(nodes)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Evaluate the query engine on the dataset
    eval_on_dataset(query_engine, DATA_PATH, RESULTS_PATH)


if __name__ == '__main__':
    if MODEL_TYPE == 'api':
        assert API_KEY and SECRET_KEY, "API_KEY and SECRET_KEY must be provided"
        llm = MyApiLLM(API_KEY, SECRET_KEY)
    else:
        llm = MyLocalLLM(LLM_MODEL_PATH)

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH)

    main()
