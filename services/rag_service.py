from langchain_chroma import Chroma
#from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

class RAGService:
    def __init__(self, model, text_splitter, vectorstore_service, schema, queries=List[str]):
        """
        Initialize the RAG Service.

        Args:
            model: The model to use to generate text.
            text_splitter: The text splitter to use to split the text into chunks.
            vectorstore_service: The vector store service to use to store and retrieve chunks.
            schema: The schema to use to validate the extracted information.
            queries: The questions to ask the model to generate text.
        """
        self.model = model
        self.text_splitter = text_splitter
        self.vectorstore_service = vectorstore_service
        self.queries = queries
        self.schema = schema
        self.prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

    def get_relevant_chunks(self) -> List[str]:
        """
        Get all the relevant chunks from the text for the given queries.

        Returns:
            List[str]: A list of all the relevant chunks from the text.
        """
        retrieved_texts = []
        for query in self.queries:
            docs = self.retriever.get_relevant_documents(query)
            retrieved_texts.extend([doc.page_content for doc in docs])
        return list(set(retrieved_texts))

    def run(self, file_path):
        """
        Processes a PDF file to extract relevant information based on predefined queries.

        Args:
            file_path (str): The path to the PDF file to be processed.

        Returns:
            dict: A dictionary containing the extracted information structured according to the defined schema.
        """

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        self.vectorstore = self.vectorstore_service.from_documents(docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
        self.splits = self.text_splitter.split_documents(docs)
        self.retriever = self.vectorstore.as_retriever()

        # Creating the chain
        runnable = self.prompt | self.model.with_structured_output(self.schema)

        relevant_chunks = self.get_relevant_chunks()

                # Combine the chunks into a condensed text
        reduced_text = " ".join(relevant_chunks)

        # Execute the model on the condensed text
        result = runnable.invoke({"text": reduced_text})

        return dict(result)