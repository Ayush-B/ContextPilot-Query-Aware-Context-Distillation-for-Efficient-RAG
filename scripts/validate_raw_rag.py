from contextpilot.retrieval.retriever import Retriever
from contextpilot.generation.prompt_builder import PromptBuilder


def main():

    retriever = Retriever()
    builder = PromptBuilder()

    query = "What is hybrid retrieval?"

    chunks = retriever.retrieve(query, k=5)

    prompt = builder.build_prompt(query, chunks)

    print("PROMPT PREVIEW\n")
    print("-" * 60)
    print(prompt)


if __name__ == "__main__":
    main()