import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss
import numpy as np

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page config
st.set_page_config(page_title="AING FARIS RAG Agent", page_icon="💸")
st.write("💸 AING FARIS - Maneh siapa?")

# --- HELPER FUNCTIONS ---

@st.cache_data
def split_text(text, chunk_size=500, overlap=50):
    tokens = text.split()
    chunks = [
        " ".join(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), chunk_size - overlap)
    ]
    return chunks

def embed_texts(texts, batch_size=50):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        embeddings.extend(np.array(e.embedding, dtype=np.float32) for e in response.data)
    return embeddings

def embed_query(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=[text])
    return np.array(response.data[0].embedding, dtype=np.float32)

@st.cache_resource
def process_pdf(pdf_bytes):
    pdf_reader = PdfReader(pdf_bytes)
    full_text = "\n".join(
        page.extract_text() for page in pdf_reader.pages if page.extract_text()
    )

    chunks = split_text(full_text)
    embeddings = embed_texts(chunks)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return chunks, index, full_text

# --- CHAT HISTORY INIT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
               "role": "system",
                "content": (
                    "You are Faris, an AI persona inspired by Faris Abduraahim. "
                    "You are a young, educated millennial playboy, but extremely sensitive and serious when it comes to finances. "
                    "Your style is informal Indonesian, like a Jakarta youngster who often uses words such as 'wkwkwk', 'bege', 'si bangke', 'oke siap', 'nuhun', 'bray', and 'aman'. "
                    "You call yourself 'aing' and always call the user 'mane'. "
                    "You always pay close attention to balance sheets (neraca), profit and loss statements (laba rugi), and cash flows (arus kas). "
                    "Your tone is friendly, humorous, casual, and relaxed—like chatting with an old friend—yet you switch to serious and professional when discussing financial matters. "
                    "If relevant context is provided from user documents, you must use it thoroughly to help answer questions. "
                    "Remember: Always respond in an informal, Jakarta millennial style unless the topic requires a more serious tone."
                )

        }
    ]

# --- FILE UPLOAD ---
st.subheader("📂 Maneh upload pdf maneh kesini")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

chunks, index, full_text = [], None, ""

if uploaded_file:
    with st.spinner("Ngolah PDF maneh, jangan diganggu Aing lagi kerja..."):
        uploaded_bytes = uploaded_file.getvalue()
        chunks, index, full_text = process_pdf(uploaded_file)
        st.success(f"✅ PDF processed into {len(chunks)} chunks. Aing siap bantu!")

# --- DISPLAY MESSAGES ---
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HANDLE USER INPUT ---
if prompt := st.chat_input("Tanya apa aja soal finansial maneh di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Add context if available
    if index is not None and chunks:
        #  if index is not None and chunks:
        # st.session_state.messages.pop(-2)
        with st.spinner("Faris lagi cari data relevan..."):
            q_emb = embed_query(prompt).reshape(1, -1)
            D, I = index.search(q_emb, k=5)
            context_chunks = [chunks[i] for i in I[0]]

            # Add the retrieved context as a system message for the LLM to use
            context_text = "\n\n".join(context_chunks)
            retrieval_message = {
                "role": "system",
                "content": (
                    "Berikut adalah informasi relevan dari dokumen user yang harus kamu gunakan sebagai referensi jika membantu menjawab pertanyaan:\n\n"
                    + context_text
                )
            }
            st.session_state.messages.append(retrieval_message)

        # Show expanders below the chat
        with st.container():
            st.write("### 📖 Hasil pencarian relevan:")
            for idx, chunk in enumerate(context_chunks):
                with st.expander(f"💡 Relevan #{idx + 1}"):
                    st.markdown(chunk)

    else:
        st.session_state.messages.append({
            "role": "system",
            "content": "⚠️ Maneh belum upload dokumen jadi aing jawab sebisanya ya."
        })

    # Get LLM response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=st.session_state.messages,
            stream=True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response}) 

st.button("Reset Chat", on_click=lambda: st.session_state.clear(), type="primary")