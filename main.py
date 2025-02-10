from services.db_service import *
from services.rag_service import *
from utils import *


file_path = "/home/patrick/langchain_pdf_mongodb/files/TD440.pdf"


class PaperInformations(BaseModel):
    title: Optional[str] = Field(default=None, description="The title of the paper")
    authors: Optional[List] = Field(default=None, description="The authors of the paper")
    date: Optional[str] = Field(default=None, description="The date of the paper")
    abstract: Optional[str] = Field(default=None, description="The abstract of the paper")
    keywords: Optional[List] = Field(default=None, description="The keywords of the paper")
    techniques: Optional[List] = Field(default=None, description="Analyse and return the statistical techniques applied on the paper")
    data_source: Optional[str] = Field(default=None, description="The data source of the paper")
    citation: Optional[str] = Field(default=None, description="The citation in ABNT format, basically how to cite this paper, containing the authors names, the title and the year")

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
    model=ChatOpenAI(model = "gpt-3.5-turbo-1106", temperature=0),
    text_splitter=SemanticChunker(embeddings=OpenAIEmbeddings(model="text-embedding-3-small")),
    vectorstore_service=Chroma,
    schema=PaperInformations,
    queries=selected_queries
)


result = rag.run(file_path)

result = fix_unidecode(result)

mongodb.insert_document(result)


