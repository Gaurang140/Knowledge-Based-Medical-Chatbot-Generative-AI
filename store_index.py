from src.helper import load_pdf_file , text_splits , download_hugging_face_embedding_model
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv 
import os 


load_dotenv()

 

PINE_CONE_API= os.getenv("PINE_CONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINE_CONE_API



extracted_data = load_pdf_file(data="Data/")
text_chunks = text_splits(extracted_data=extracted_data)
embeddings = download_hugging_face_embedding_model()

pc = Pinecone(api_key=PINE_CONE_API)


index_name = "medibot"



pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)



docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks , 
    index_name=index_name, 
    embedding=embeddings
)