import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file and returns it as a string.

    Parameters:
    file_path (str): The path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """

    # 1. Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # 2. Check if the file is a PDF by extension (basic check)
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"Unsupported file format: '{file_path}' is not a PDF file.")

    # 3. Initialize an empty string to accumulate text
    extracted_text = ""

    try:
        # 4. Open the PDF file using PdfReader
        reader = PdfReader(file_path)

        # 5. Iterate through each page in the PDF
        for page in reader.pages:
            # page.extract_text() can sometimes return None, so we handle that gracefully
            text = page.extract_text() or ""
            extracted_text += text

    except Exception as e:
        # 6. Catch any other exceptions during PDF reading/parsing
        raise RuntimeError(f"An error occurred while reading the PDF: {str(e)}")

    # 7. Return the combined text from all pages
    return extracted_text


def chunk_text_with_langchain(text, chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". "]):
    """
    Splits a given string of text into manageable chunks using LangChain's CharacterTextSplitter.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size for each chunk.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """

    # 1. Handle empty or invalid text input
    if not text:
        return []

    # 2. Validate chunk_size and chunk_overlap
    #    - chunk_size should be greater than zero to avoid an infinite loop.
    #    - chunk_overlap should not exceed chunk_size, otherwise you won't make progress.
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    # 3. Create a CharacterTextSplitter instance with specified chunk_size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    # 4. Use the splitter to split the text into chunks
    #    This will return a list of strings, each being a chunk of the original text.
    chunks = splitter.split_text(text)

    # 5. Return the list of chunks
    return chunks



def embed_text_with_openai(text, model_name="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using OpenAI's Embedding API.

    Args:
        text (str): The input string to embed.
        model_name (str): The name of the OpenAI model to use for embeddings.

    Returns:
        list: The embedding vector as a list of floats.

    Raises:
        ValueError: If the input text is empty.
        AuthenticationError: If there's an issue with the API key.
        OpenAIError: If any other API error occurs.
    """

    # 1. Check if the input text is empty
    if not text.strip():
        raise ValueError("Input text is empty. Please provide a valid string.")

    try:
        # 2. Call the OpenAI Embedding API
        response = openai.Embedding.create(
            input=text,
            model=model_name
        )
        # 3. Extract the embedding from the API response
        embedding = response["data"][0]["embedding"]

        # 4. Return the embedding as a list of floats
        return embedding
    except openai.error.AuthenticationError as auth_err:
        return auth_err;
    except openai.error.OpenAIError as api_err:
        return api_err;