import re
import os
from typing import Callable, List, Dict
from dataclasses import dataclass

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import pdfplumber
import PyPDF4
from dotenv import load_dotenv

from common import fetch_args


@dataclass
class ParsedPDF:
    # raw_pages: List[Tuple[int, str]]
    raw_pages: List[str]
    metadata: Dict[str, str]
    cleaned_pages: List[str] = None


def extract_metadata_from_pdf(file_path: str) -> dict:
    """
    Extracts pdf metadata.
    :param file_path: The path to the PDF file.
    :return: A dictionary of metadata entries.
    """
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }


def extract_pages_from_pdf(file_path: str) -> List[str]:
    """
    Extracts the text from each page of the PDF.
    :param file_path: The path to the PDF file.
    :return: A list containing the extracted text indexed by original page.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append(text)
    return pages


def parse_pdf(file_path: str) -> ParsedPDF:
    """
    Extracts the title and text from each page of the PDF.
    :param file_path: The path to the PDF file.
    :return: A ParsedPDF object.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return ParsedPDF(raw_pages=pages, metadata=metadata)


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(ppdf: ParsedPDF,
               cleaning_functions: List[Callable[[str], str]]) -> None:
    """Clean ppdf.raw_pages and store in ppdf.cleaned_pages."""
    cleaned_pages = []
    for text in ppdf.raw_pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append(text)
    ppdf.cleaned_pages = cleaned_pages


def text_to_docs(ppdf: ParsedPDF) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in enumerate(ppdf.cleaned_pages):
        # TODO: tunable params...
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **ppdf.metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    # set OPENAI_API_KEY from .env file
    load_dotenv()

    # fetch cli args
    conf = fetch_args()

    # Parse PDF
    ppdf = parse_pdf(conf.pdf_path)

    # Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    clean_text(ppdf, cleaning_functions)
    document_chunks = text_to_docs(ppdf)

    # Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=conf.collection_name,
        persist_directory=conf.persist_dir,
    )

    exit()
    # Save DB locally
    vector_store.persist()
