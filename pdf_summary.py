"""PDF Summary Tool — read a PDF and print a structured summary with page citations."""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

MODEL = "qwen/qwen3.5-flash-02-23"

# Cap extracted text so long slide decks stay within the model's context window.
MAX_TEXT_CHARS = 30_000

SYSTEM_PROMPT = """You are a precise summarization assistant.

Given the text of a PDF (tagged with [Page N] markers), produce a summary with
exactly three sections:

## Overview
A 3-5 sentence high-level summary of the document.

## Key Points
3-5 bullet points. After EVERY key point, add a citation in the format
[Page X] where X is the page number the information came from.

## Limitations
A short paragraph noting anything the text does not cover, or the fact that
this is a summary of the extracted text only.

Rules:
1. Use ONLY information present in the provided text.
2. Cite every key point with [Page X] — the word 'Page' followed by the number.
3. If a page has no extractable text, skip it silently.
4. Do NOT add information beyond what is in the text.
"""


def get_client() -> OpenAI:
    """Load the API key from .env and return an OpenRouter-compatible client."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def parse_page_range(spec: str) -> tuple[int, int] | None:
    """Parse a 'START-END' page range. Returns (start, end) 1-based inclusive, or None if empty."""
    if not spec:
        return None
    parts = spec.split("-")
    if len(parts) != 2 or not parts[0].strip().isdigit() or not parts[1].strip().isdigit():
        raise SystemExit(
            f"Error: Invalid page range '{spec}'. Expected format START-END, e.g. --pages 1-5"
        )
    start, end = int(parts[0].strip()), int(parts[1].strip())
    if start < 1 or start > end:
        raise SystemExit(
            f"Error: Invalid page range '{spec}'. START must be at least 1 and no greater than END."
        )
    return start, end


def extract_text(pdf_path: str, page_range: tuple[int, int] | None = None) -> str:
    """Extract per-page text from a PDF, tagging each page with [Page N] markers.

    page_range is a 1-based inclusive (start, end) tuple; None means all pages.
    """
    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)
    if page_range:
        start, end = page_range
        if start > page_count:
            raise SystemExit(
                f"Error: Page range starts at page {start}, but the PDF has only {page_count} page(s)."
            )
        end = min(end, page_count)
        indices = range(start - 1, end)  # PDF pages are 0-based internally
    else:
        indices = range(page_count)

    pages = []
    for i in indices:
        page_text = reader.pages[i].extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(f"[Page {i + 1}]\n{page_text}")
    return "\n\n".join(pages)


def summarize(client: OpenAI, extracted_text: str) -> str:
    """Send the extracted text to the LLM and return the structured summary."""
    user_message = (
        f"Here is the text extracted from a PDF, tagged with page numbers:\n\n"
        f"{extracted_text}"
    )
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
        description="Read a PDF and print a structured summary (Overview, Key Points, Limitations)."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to summarize.")
    parser.add_argument(
        "--pages",
        metavar="START-END",
        help="Only summarize pages START through END (1-based, inclusive), e.g. --pages 1-5.",
    )
    args = parser.parse_args()

    # Guardrail — missing or unreadable file gets a friendly message, no traceback.
    if not os.path.isfile(args.pdf_path):
        print(f"Error: File not found - {args.pdf_path}")
        print("Usage: python pdf_summary.py <path-to-pdf>")
        raise SystemExit(1)

    # Validate the page range before doing any heavier work.
    page_range = parse_page_range(args.pages) if args.pages else None

    client = get_client()

    try:
        extracted_text = extract_text(args.pdf_path, page_range)
    except Exception as exc:
        print(f"Error: Could not read PDF - {exc}")
        raise SystemExit(1)

    # Guardrail — scanned/image-only PDF: explain instead of calling the LLM with empty text.
    if not extracted_text.strip():
        print(
            "This PDF appears to be scanned or contains no extractable text. "
            "This tool only supports text-based PDFs."
        )
        raise SystemExit(1)

    # Keep long documents within the model's context window.
    if len(extracted_text) > MAX_TEXT_CHARS:
        extracted_text = extracted_text[:MAX_TEXT_CHARS]
        print(f"(Note: PDF exceeds {MAX_TEXT_CHARS} characters; summarizing the first portion.)")

    summary = summarize(client, extracted_text)
    print(summary)


if __name__ == "__main__":
    main()
