pythonimport os
import streamlit as st
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AI Book Concierge", page_icon="📚", layout="centered")
st.title("📚 Comprehensive 900-Page Book Assistant")
st.write("Query deep context across the entire book knowledge base instantly.")

groq_api_key = os.environ.get("GROQ_API_KEY")
chroma_api_key = os.environ.get("CHROMA_API_KEY")

if not groq_api_key or not chroma_api_key:
    st.error("Error: Missing production API environmental variables inside system keys.")
    st.stop()

@st.cache_resource
def load_cloud_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant="5a13dadc-ed22-41d1-9fd5-75981663a42c", # Your exact cloud UUID
        database="default-database"
    )
    return Chroma(
        client=chroma_client,
        collection_name="large-book-index",
        embedding_function=embeddings
    )

vectorstore = load_cloud_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.1, groq_api_key=groq_api_key)

system_prompt = (
    "You are a highly precise AI assistant trained on a 900-page comprehensive handbook volume.\n"
    "Analyze the provided retrieved context snippets carefully to format your final answer.\n"
    "If the exact answer cannot be extracted from the context, reply with: "
    "'I am sorry, but I cannot find that documentation across the text of the book.'\n\n"
    "Context extracts:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have indexed the entire 900-page textbook volume. Ask me anything!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about the book..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Searching deep index boundaries..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            response_placeholder.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
