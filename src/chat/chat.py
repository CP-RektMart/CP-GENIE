from model.retriever.retrieve import retrieve_docs
from model.generator.generate import generate_answer, get_context, get_llm

def main():
    query = input("Enter your question: ")

    docs = retrieve_docs(query, k=10)

    context = get_context(docs)
    llm = get_llm()
    answer = generate_answer(llm, query, context)

    print("\n=== Answer ===\n", answer)