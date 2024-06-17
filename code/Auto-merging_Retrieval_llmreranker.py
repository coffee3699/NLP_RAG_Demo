# -*- coding: UTF-8 -*-
"""
@Project ：rag 
@File    ：Auto-merging_Retrieval.py
@Author  ：zfk
@Date    ：2024/5/8 11:25
"""
import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from model_response import MyApiLLM, MyLocalLLM
from llama_index.core.node_parser import SemanticSplitterNodeParser


from llama_index.llms.anthropic import Anthropic

from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank


def build_automerging_index(documents, save_dir="merge_index", chunk_sizes=None):
    chunk_sizes = chunk_sizes or [1024, 256, 64]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return automerging_index,leaf_nodes

def build_automerging_index_Semantic(documents, save_dir="merge_index", buffer_size=1024, embed_model=None):
    node_parser = SemanticSplitterNodeParser(buffer_size=buffer_size, embed_model=embed_model)
    nodes = node_parser.build_semantic_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return automerging_index,leaf_nodes

def main(args):
    documents = SimpleDirectoryReader(input_files=[args.data_path]).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    index ,leaf_nodes = build_automerging_index([document], save_dir=args.save_path)
    # index ,leaf_nodes= build_automerging_index_Semantic([document], save_dir=args.save_path, embed_model=Settings.embed_model)

    print("分块后的内容")
    for node in leaf_nodes:
        print(node.text)
    base_retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
    retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
    auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[Settings.rerank_model])
    auto_merging_response = auto_merging_engine.query("What did Elon Musk become in 2004?")
    print('auto_merging_response:', auto_merging_response)


if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='api', choices=['api', 'local'])
    parser.add_argument('--api_key', type=str, help='api_key', default='')
    parser.add_argument('--secret_key', type=str, help='secret_key', default='')
    parser.add_argument('--llm_model_path', type=str, help='local llm model path', default='../qwen1.5-0.5B')
    parser.add_argument('--embedding_model_path', type=str, help='local embedding model path',default='../BAAI/bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
    parser.add_argument('--similarity_top_k', type=int, default=12)
    parser.add_argument('--data_path', type=str, help='local data path', default='../data/Elon.txt')
    parser.add_argument('--save_path', type=str, help='chunk save path', default='./merge_index')
    parser.add_argument('--rerank_model_path', type=str, help='local rerank model path', default='../BAAI/bge-reranker-base/snapshots/02affbdf4485dbf4ae0f133ec3ca467884c00b99')
    parser.add_argument('--rerank_top_n', type=int, default=2)
    args = parser.parse_args()

    if args.model_type == 'api':
        assert args.api_key and args.secret_key, "api_key and secret_key must be provided"
        llm = MyApiLLM(args.api_key, args.secret_key)
    else:
        llm = MyLocalLLM(args.llm_model_path)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)
    reranker = RankGPTRerank(
    top_n = 3,
    llm = MyApiLLM(args.api_key, args.secret_key)
    # verbose=True,
)
    # Settings.rerank_model = SentenceTransformerRerank(top_n=args.rerank_top_n, model=args.rerank_model_path)
    Settings.rerank_model = reranker

    main(args)
#在命令行查看openai库的版本
#pip show openai
