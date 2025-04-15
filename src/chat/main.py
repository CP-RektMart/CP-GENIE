from src.model.retriever.retrieve import retrieve_docs
from src.model.generator.generate import generate_answer, get_context, get_llm

def main():
    query = input("Enter your question: ")

    docs = retrieve_docs(query, k=10)

    for doc in docs:
        print(doc.metadata.get("title"))
        print(doc.page_content[:300])
        print("-" * 50)

    context = get_context(docs)
    llm = get_llm()
    answer = generate_answer(llm, query, context)

    print("\n=== Answer ===\n", answer)


if __name__ == "__main__":
    main()
