"""
Hierarchical Indexing, Auto-Merging Retriever, BGE Reranker, FT ChatGLM3-6B
"""
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from model_response import MyLocalLLM, MyApiLLM
from utils import load_documents, eval_on_dataset

load_dotenv()

# Global configuration variables
MODEL_TYPE = 'local'  # 'local' or 'api'
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
LLM_MODEL_FT_PATH = os.getenv('LLM_MODEL_FT10_PATH')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH')
DATA_PATH = os.getenv('EVAL_DATASET_PATH')
SAVE_PATH = 'index/merge_index'
RESULTS_PATH = 'HI_AM_BGE_FT_results.json'
SIMILARITY_TOP_K = 12
RERANK_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH')
RERANK_TOP_N = 5


def build_hierarchical_index(documents, save_dir=SAVE_PATH, chunk_sizes=None):
    if chunk_sizes is None:
        chunk_sizes = [1024, 256, 64]

    # Parse the documents into hierarchical nodes
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    leaf_nodes = get_leaf_nodes(nodes)

    # Create a storage context and add the documents to the docstore
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # Create the hierarchical index or load it from disk
    if not os.path.exists(save_dir):
        hierarchical_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)
        hierarchical_index.storage_context.persist(persist_dir=save_dir)
    else:
        hierarchical_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return hierarchical_index


def main():
    # Load the documents and build the hierarchical index
    documents = load_documents(DATA_PATH)
    index = build_hierarchical_index(documents=documents, save_dir=SAVE_PATH)

    # Create the auto-merging retriever and query engine
    base_retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[
            Settings.rerank_model
        ]
    )

    # Evaluate the query engine on the dataset
    eval_on_dataset(query_engine, DATA_PATH, RESULTS_PATH)


if __name__ == '__main__':
    if MODEL_TYPE == 'api':
        assert API_KEY and SECRET_KEY, "API_KEY and SECRET_KEY must be provided"
        llm = MyApiLLM(API_KEY, SECRET_KEY)
    else:
        llm = MyLocalLLM(LLM_MODEL_FT_PATH)

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH)
    Settings.rerank_model = SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL_PATH)

    main()
