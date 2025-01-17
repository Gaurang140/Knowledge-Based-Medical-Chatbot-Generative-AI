from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings





# Exract data from the pdf file 

def load_pdf_file(data):

    """
    load data using this function 
    """

    loader = DirectoryLoader(
        data , 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents



def text_splits(extracted_data):

    text_splitters = RecursiveCharacterTextSplitter(

        chunk_size = 500 , 
        chunk_overlap = 20
    )

    text_chunks = text_splitters.split_documents(extracted_data)

    return text_chunks




def download_hugging_face_embedding_model():
    emebeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return emebeddings
