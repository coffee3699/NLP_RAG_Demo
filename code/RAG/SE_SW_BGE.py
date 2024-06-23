import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from RAG.utils import load_documents, eval_on_dataset
from model_response import MyApiLLM, MyLocalLLM

load_dotenv()

# Global configuration variables
MODEL_TYPE = 'local'  # 'local' or 'api'
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH')
DATA_PATH = os.getenv('EVAL_DATASET_PATH')
SAVE_PATH = 'index/sentence_index'
RESULTS_PATH = 'SE_SW_BGE_results.json'
SIMILARITY_TOP_K = 12
RERANK_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH')
RERANK_TOP_N = 2


def build_sentence_window_index(documents, sentence_window_size=3, save_dir=SAVE_PATH):
    # Create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.node_parser = node_parser
    # Create or load the sentence index
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(documents, show_progress=True)
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return sentence_index


def main():
    # Load the documents and build the sentence window index
    documents = load_documents(DATA_PATH)
    index = build_sentence_window_index(documents, save_dir=SAVE_PATH)

    # Post-processing document
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # Create the query engine
    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K, node_postprocessors=[postproc, Settings.rerank_model]
    )

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
    Settings.rerank_model = SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL_PATH)

    main()
