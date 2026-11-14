import streamlit as st
import os
import base64
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain & google genai / huggingface
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(page_title="Nzelu AI Workspace", layout="wide")

st.title("Nzelu AI Study Workspace")

# -------------------------------------------------------------
# 1. Initialize Pinecone & Models
# -------------------------------------------------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "edu-app")

# Ensure standard vector index exists (768 dimensions, cosine metric)
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Use HuggingFace local embeddings (natively outputs 768 dimensions to match Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Keep Gemini for the AI Tutor chat responses
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# -------------------------------------------------------------
# 2. Sidebar: Document Upload & Selection
# -------------------------------------------------------------
st.sidebar.header("📁 Study Workspaces")

uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Track uploaded documents in session state
if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = {}

if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.uploaded_files_cache:
        with st.spinner(f"Processing and indexing {file_name}..."):
            # Read file bytes
            bytes_data = uploaded_file.read()
            st.session_state.uploaded_files_cache[file_name] = bytes_data

            # Extract text
            if file_name.endswith(".txt"):
                text = bytes_data.decode("utf-8")
            else:
                try:
                    from pypdf import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(bytes_data))
                    text = "".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    text = bytes_data.decode("latin-1", errors="ignore")

            # Split text and index to Pinecone with source metadata
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            docs = [Document(page_content=t, metadata={"source": file_name}) for t in text_splitter.split_text(text)]
            
            if docs:
                # Batch upsert safely
                batch_size = 5
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    vector_store.add_documents(batch)

                st.sidebar.success(f"Indexed {file_name} successfully!")

# File selector dropdown
file_list = list(st.session_state.uploaded_files_cache.keys())
selected_file = st.sidebar.selectbox("Active Study Document", file_list if file_list else ["No documents uploaded"])

# Initialize isolated chat history dictionary per file
if "file_messages" not in st.session_state:
    st.session_state.file_messages = {}

current_chat_key = selected_file if selected_file != "No documents uploaded" else "default"
if current_chat_key not in st.session_state.file_messages:
    st.session_state.file_messages[current_chat_key] = []

# -------------------------------------------------------------
# 3. Main Layout: Split View (PDF Viewer on Left, Chat on Right)
# -------------------------------------------------------------
col_pdf, col_chat = st.columns([1.2, 1])

with col_pdf:
    st.subheader(f"📄 {selected_file if selected_file != 'No documents uploaded' else 'None'}")
    if selected_file != "No documents uploaded" and selected_file in st.session_state.uploaded_files_cache:
        file_bytes = st.session_state.uploaded_files_cache[selected_file]
        
        # Check if it's a text file vs PDF
        if selected_file.endswith(".txt"):
            text_content = file_bytes.decode("utf-8", errors="ignore")
            st.text_area("Document Content", text_content, height=600)
        else:
            try:
                import fitz  # PyMuPDF
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                total_pages = len(pdf_document)
                
                if total_pages > 1:
                    page_num = st.slider("Select Page", 1, total_pages, 1) - 1
                else:
                    page_num = 0
                    
                # Render selected page as an image
                page = pdf_document[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                
                st.image(img_bytes, caption=f"Page {page_num + 1} of {total_pages}", use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not render PDF pages: {e}")
    else:
        st.info("Upload a document using the sidebar to view its pages here.")

with col_chat:
    st.subheader("💬 Nzelu AI Tutor")

    # Display chat history for the active document
    chat_container = st.container(height=450)
    with chat_container:
        messages = st.session_state.file_messages[current_chat_key]
        if not messages:
            st.markdown("*Hello! Ask any question about your active document.*")
        
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User chat input bar
    prompt = st.chat_input("Ask about your study material...")

    if prompt and selected_file != "No documents uploaded":
        # Append user message
        st.session_state.file_messages[current_chat_key].append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Retrieve relevant chunks strictly filtered by the active file name
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4, 
                "score_threshold": 0.4,
                "filter": {"source": selected_file}
            },
        )

        docs = retriever.invoke(prompt)
        docs_text = "\n\n".join(d.page_content for d in docs)

        system_prompt = f"""You are Nzelu ai, an expert AI tutor and study assistant created by Joel Musonda. 
        Answer the student's question based strictly on the provided document context below. 
        To help the student master the concept deeply, you MUST include:
        1. A clear explanation of the answer.
        2. A memorable, creative real-world analogy.
        3. A brief practical case study or scenario demonstrating how this concept applies.
        
        If the context does not contain enough information to answer, state clearly that the document does not contain the answer.

        Context:
        {docs_text}
        """

        cleaned_history = []
        last_role = None
        for m in st.session_state.file_messages[current_chat_key][:-1]:
            role = m["role"]
            if role != last_role:
                if role == "user":
                    cleaned_history.append(HumanMessage(content=m["content"]))
                else:
                    cleaned_history.append(AIMessage(content=m["content"]))
                last_role = role

        messages_for_llm = [SystemMessage(content=system_prompt)] + cleaned_history + [HumanMessage(content=prompt)]

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Generating insights, analogies, and case studies..."):
                    result = llm.invoke(messages_for_llm).content
                    st.markdown(result)
                    st.session_state.file_messages[current_chat_key].append({"role": "assistant", "content": result})
    elif prompt and selected_file == "No documents uploaded":
        st.warning("Please upload a document in the sidebar first before chatting.")