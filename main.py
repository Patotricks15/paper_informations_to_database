from services.db_service import *
from services.rag_service import *
from utils import *
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

file_path = "files/TD440.pdf"

from typing import TypedDict, Annotated, Optional, List

class PaperInformations(TypedDict):
    title: Annotated[Optional[str], "The title of the paper"]
    authors: Annotated[Optional[List[str]], "The authors of the paper"]
    date: Annotated[Optional[str], "The date of the paper"]
    abstract: Annotated[Optional[str], "The abstract of the paper"]
    keywords: Annotated[Optional[List[str]], "The keywords of the paper"]
    techniques: Annotated[Optional[List[str]], "Analyse and return the statistical techniques applied on the paper"]
    dataset_source: Annotated[Optional[str], "Where the data used in the paper comes from? It isn't the file path, it's the origin of the dataset"]
    citation: Annotated[Optional[str], "The citation in ABNT format, basically how to cite this paper, containing the authors names, the title and the year"]


selected_queries = [
    "The title of the paper",
    "The authors of the paper",
    "The date of the paper",
    "The abstract of the paper",
    "The keywords of the paper",
    "The statistical techniques applied on the paper",
    "The data source of the paper",
    "The citation in ABNT format containing the authors names, the title and the year"
]

mongodb = MongoDBService("langchain_pdf_mongodb", "langchain_pdf_mongodb")

rag = RAGService(
    model=ChatOpenAI(model = "gpt-4o-mini", temperature=0),
    text_splitter=RecursiveCharacterTextSplitter(),
    vectorstore_service=Chroma,
    schema=PaperInformations,
    queries=selected_queries
)


result = rag.run(file_path)

result = fix_unidecode(result)

print(result)

mongodb.insert_document(result)


