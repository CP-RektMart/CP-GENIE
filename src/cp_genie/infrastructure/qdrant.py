from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from qdrant_client import QdrantClient
from cp_genie.core.config import Settings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

settings = Settings()
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_embedding_model)
reranker_model = HuggingFaceCrossEncoder(model_name=settings.reranker_model)


def initialize_vectorstore():
    print(settings.qdrant_url)
    print(settings.qdrant_api_key)
    faculty_qdrant = QdrantVectorStore(
        client=client,
        collection_name="cp-genie-faculty-info",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    other_qdrant = QdrantVectorStore(
        client=client,
        collection_name="cp-genie-other-info",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    reranker = CrossEncoderReranker(model=reranker_model, top_n=5)

    retriever_fac = faculty_qdrant.as_retriever(search_type="similarity", k=10)
    retriever_oth = other_qdrant.as_retriever(search_type="similarity", k=10)
    retriever_fac_reranked = ContextualCompressionRetriever(
        base_retriever=retriever_fac, base_compressor=reranker
    )

    retriever_oth_reranked = ContextualCompressionRetriever(
        base_retriever=retriever_oth, base_compressor=reranker
    )
    return retriever_fac_reranked, retriever_oth_reranked
