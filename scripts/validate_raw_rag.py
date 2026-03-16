from contextpilot.retrieval.retriever import Retriever
from contextpilot.generation.prompt_builder import PromptBuilder
from contextpilot.generation.generator import Generator


def main():

    retriever = Retriever()
    builder = PromptBuilder()
    generator = Generator()

    query = "What is hybrid retrieval?"

    chunks = retriever.retrieve(query, k=5)

    prompt = builder.build_prompt(query, chunks)

    print("PROMPT\n")
    print("-" * 60)
    print(prompt)

    print("\nMODEL ANSWER\n")
    print("-" * 60)

    answer = generator.generate(prompt)

    print(answer)


if __name__ == "__main__":
    main()