from flask import Flask, render_template, request
from src.helper import download_hugging_face_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.environ.get('PINE_CONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings and setup Pinecone
embeddings = download_hugging_face_embedding_model()
index_name = "medibot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM
llm = ChatGroq(model="Llama-3.3-70b-Versatile", api_key=GROQ_API_KEY)

# Define the system prompt
system_prompt = (
    "Your name is Medibot. You are an assistant for medical question-answering tasks. "
    "You will try to answer medical-related questions and will not answer general questions. "
    "Use the following pieces of retrieval to answer the question. If you don't know the answer, "
    "say 'I don't know.' Use three sentences maximum and keep answers concise. Ask, 'Would you like to know the solution for that?' if applicable."
    "\n\n{context}"
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

# Create question-answer and retrieval chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat history store
store = {}

# Function to get session-specific chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history tracking
conversational_reg_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Define config for session management
config = {"configurable": {"session_id": "chat1"}}

# Flask routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]  # Get user input
    print(f"User Input: {msg}")
    
    try:
        # Retrieve relevant documents for context
        retrieved_docs = retriever.get_relevant_documents(msg)
        context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine into a single string

        # Get session-specific chat history
        session_id = config["configurable"]["session_id"]
        chat_history = get_session_history(session_id)

        # Invoke the chain with all required variables
        response = conversational_reg_chain.invoke(
            {
                "input": msg,  # User's current query
                "context": context,  # Relevant information retrieved
                "chat_history": chat_history.messages  # Messages from session history
            },
            config=config  # Configuration for the chain
        )
        
        print("Response:", response["answer"])
        return str(response["answer"])  # Return chatbot's response
    except Exception as e:
        print("Error:", str(e))
        return "An error occurred while processing your request."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
