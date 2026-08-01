"""CLI Q&A Tool — answer questions about pasted text with paragraph-level citations."""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = """You are a precise research assistant.

Rules:
1. Answer ONLY using information from the provided text.
2. After EVERY claim, add a citation in the format [Paragraph X].
3. If a sentence uses information from multiple paragraphs, cite all of them.
4. If the text does not contain the answer, reply:
   "The text does not provide this information."
5. Do NOT add any information beyond what is in the text.

Example:
If the text says:
[Paragraph 1] The sky is blue.
[Paragraph 2] Grass is green.

And the question is: 'What color is the sky?'
Your answer should be: 'The sky is blue [Paragraph 1].'
"""


def get_client() -> OpenAI:
    """Load the API key from .env and return an OpenRouter-compatible client."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def read_text_interactive() -> str:
    """Read multi-line text from stdin until a line containing only 'END'."""
    print("Paste your text below. Type END on a new line when done:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def read_text_from_file(filepath: str) -> str:
    """Read and strip text from a UTF-8 file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"Error: File not found - {filepath}")
    except OSError as exc:
        raise SystemExit(f"Error: Cannot read file — {exc}")


def number_paragraphs(raw_text: str) -> str:
    """Split text into paragraphs on blank lines and number them [1], [2], ..."""
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    return "\n\n".join(f"[{i}] {block}" for i, block in enumerate(blocks, start=1))


def ask(client: OpenAI, numbered_paragraphs: str, question: str) -> str:
    """Send the prompt to the LLM and return the answer text."""
    user_message = f"Text:\n{numbered_paragraphs}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about a text with paragraph-level citations."
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Read the text from a file instead of pasting it interactively.",
    )
    args = parser.parse_args()

    client = get_client()

    # 1. Read the text — from file or interactively
    if args.file:
        raw_text = read_text_from_file(args.file)
    else:
        raw_text = read_text_interactive()

    # 2. Guardrail — empty text means no API call
    if not raw_text:
        print("Error: No text provided.")
        raise SystemExit(1)

    # 3. Number the paragraphs
    numbered = number_paragraphs(raw_text)

    # 4. Q&A loop — ask questions until the user types 'exit' or 'quit'
    print("\nYou can now ask questions about the text. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            sys.exit(0)
        if not question:
            print("Error: No question provided.")
            raise SystemExit(1)

        # 5. Ask the LLM and print the answer
        answer = ask(client, numbered, question)
        print(f"\n{answer}")
        print()


if __name__ == "__main__":
    main()
