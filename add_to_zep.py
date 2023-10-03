import argparse
import pdfplumber
from zep_python import ZepClient
import settings_private
from pydantic import BaseModel
import os

class Document(BaseModel):
    content: str
    document_id: str
    metadata: dict

    class Config:
        schema_extra = {
            "example": {
                "content": "example content",
                "document_id": "doc1",
                "metadata": {"example": "metadata"},
            }
        }

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Add documents to Zep collection")
parser.add_argument("-f", "--file", required=True, help="Path to the input PDF file")
args = parser.parse_args()

zep = ZepClient("http://localhost:8000")
client = ZepClient(base_url=settings_private.ZEP_API_URL)
collection_name = "LagrangeDocs"
collection = client.document.get_collection(collection_name)

def read_pdf(file_path):
    pdf = pdfplumber.open(file_path)
    return [page.extract_text() for page in pdf.pages]

# Use the file path provided as a command-line argument
file_path = args.file

contents = read_pdf(file_path)

# Retrieve the next available document ID from the environment variable DOCUMENT_ID
next_document_id = int(os.environ.get("DOCUMENT_ID", "1"))

documents = []
for i, content in enumerate(contents):
    document_id = f"{collection_name}-{next_document_id}"
    documents.append(
        Document(
            content=content,
            document_id=document_id,
            metadata={"page_number": i, "pdf_name": file_path},
        )
    )

# Add the documents to the collection
uuids = collection.add_documents(documents)

# Increment the environment variable DOCUMENT_ID for the next document
next_document_id += 1
os.environ["DOCUMENT_ID"] = str(next_document_id)
