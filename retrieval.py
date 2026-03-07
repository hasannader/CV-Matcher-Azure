from langchain_classic.retrievers import MultiQueryRetriever

from config import llm
from prompts import multi_query_prompt, prompt, guard_prompt


def build_retriever(vectorstore):
    """Build the MultiQueryRetriever from a given vectorstore."""
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt
    )

    return multi_query_retriever


def run_guard(query):
    """Run the guard prompt to validate the query. Returns the label string."""
    return llm.invoke(
        guard_prompt.format_messages(query=query)
    ).content.strip().upper()


def run_rag(query, multi_query_retriever):
    """
    Run the full RAG pipeline:
    - Retrieve docs via MultiQueryRetriever
    - Generate answer via main prompt
    - Extract generated multi-queries for display
    Returns (response_text, retrieved_docs, query_list)
    """
    retrieved_docs = multi_query_retriever.invoke(query)

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    formatted_prompt = prompt.format_messages(
        query=query,
        context=context_text
    )

    response = llm.invoke(formatted_prompt).content.strip()

    # Extract generated multi-queries for display
    generated = multi_query_retriever.llm_chain.invoke({"question": query})

    if isinstance(generated, list):
        query_list = [q.strip() for q in generated if q.strip()]
    else:
        raw_text = generated.content if hasattr(generated, "content") else str(generated)
        query_list = [q.strip() for q in raw_text.split("\n") if q.strip()]

    return response, retrieved_docs, query_list
