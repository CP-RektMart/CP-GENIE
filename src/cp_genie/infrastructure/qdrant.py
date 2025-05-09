from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from qdrant_client import QdrantClient
from cp_genie.core.config import Settings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

settings = Settings()
client_naive = QdrantClient(
    url=settings.qdrant_url_naive,
    api_key=settings.qdrant_api_key_naive,
)

client_contextual = QdrantClient(
    url=settings.qdrant_url_contextual,
    api_key=settings.qdrant_api_key_contextual,
)

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_embedding_model)
reranker_model = HuggingFaceCrossEncoder(model_name=settings.reranker_model)


def initialize_vectorstore():

    naive_qdrant = QdrantVectorStore(
        client=client_naive,
        collection_name="cp-genie-2",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    faculty_qdrant = QdrantVectorStore(
        client=client_contextual,
        collection_name="cp-genie-faculty-info",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    other_qdrant = QdrantVectorStore(
        client=client_contextual,
        collection_name="cp-genie-other-info",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    retriever_naive = naive_qdrant.as_retriever(search_type="similarity", k=10)
    retriever_fac = faculty_qdrant.as_retriever(search_type="similarity", k=10)
    retriever_oth = other_qdrant.as_retriever(search_type="similarity", k=10)

    reranker = CrossEncoderReranker(model=reranker_model, top_n=10)

    retriever_naive_reranked = ContextualCompressionRetriever(
        base_retriever=retriever_naive, base_compressor=reranker
    )

    retriever_fac_reranked = ContextualCompressionRetriever(
        base_retriever=retriever_fac, base_compressor=reranker
    )

    retriever_oth_reranked = ContextualCompressionRetriever(
        base_retriever=retriever_oth, base_compressor=reranker
    )
    return retriever_naive_reranked, retriever_fac_reranked, retriever_oth_reranked
