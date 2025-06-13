import streamlit as st
import openai
import tempfile
import os
import streamlit_cytoscapejs as cytoscapejs

# === Config
# st.set_page_config(page_title="Voice AI: Transcribe & Analyze", layout="wide")
st.title("🎙️ Voice AI – Transcribe, Summarize & Map")

# === OpenAI Key


api_access = st.radio(
    "Use Dera openai key?",
    options=["Yes", "No"],
    )

if api_access == "Yes":
    api_approve = st.text_input("Passwordnya apa gan?", type="password", key="openai_key") 
    if api_approve == st.secrets["ADMIN_PASSWORD"]:
        st.success("Key approved!")
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    else : 
        st.error("Swiper jangan menipu, swiper jangan menipu awuuuuuuuuuuu") 
        openai.api_key = None
else:
    openai.api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="openai_key")

# === Upload or Record Audio ===
st.header("🔊 Input")
audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "m4a", "wav"])
audio_input = st.audio_input("Or record audio")

tags = st.text_input("Add tags (optional)", placeholder="meeting, idea, task")

# === Function: Transcribe with Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        response = openai.Audio.transcribe("gpt-4o-mini-transcribe", f)
    return response['text']

# === Function: Chat Completion
def ask_gpt(prompt, role="system", model="gpt-4.1-mini"):
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

# === Function: Parse indented list into Graphviz DOT
def indented_list_to_cytoscape_elements(indented_text):
    lines = indented_text.strip().splitlines()
    elements = []
    stack = []

    for idx, line in enumerate(lines):
        label = line.strip("•- ").strip()
        indent = len(line) - len(line.lstrip())

        node_id = f"node_{idx}"
        elements.append({"data": {"id": node_id, "label": label}})

        # Find parent
        while stack and stack[-1][1] >= indent:
            stack.pop()

        if stack:
            parent_id = stack[-1][0]
            elements.append({"data": {"source": parent_id, "target": node_id}})

        stack.append((node_id, indent))

    return elements


# === Trigger
if st.button("🔍 Transcribe and Analyze") and (audio_file or audio_input):
    with st.spinner("Processing audio..."):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            if audio_file:
                tmp_file.write(audio_file.read())
            elif audio_input:
                tmp_file.write(audio_input.read())
            tmp_path = tmp_file.name
            # os.remove(tmp_path)
        transcript = transcribe_audio(tmp_path)


    # === Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Summary", "🧠 Mind Map", "📚 Full Transcript"])

    with tab1:
        with st.spinner("Generating summary..."):
            summary = ask_gpt(
                f"Summarize the following text in bullet points:\n\n{transcript}",
                role="You are an assistant summarizing voice transcriptions."
            )
            st.subheader("📝 Summary")
            st.markdown(summary)

    with tab2:
        with st.spinner("Generating mind map..."):
            mind_map_text = ask_gpt(
                f"Convert this transcript into a structured bullet-point mind map:\n\n{transcript}",
                role="You are an expert at generating hierarchical mind map structures."
            )

            st.subheader("🧠 Mind Map (Text)")
            st.markdown(mind_map_text)

            try:
                st.subheader("🧩 Interactive Mind Map (CytoscapeJS)")
                elements = indented_list_to_cytoscape_elements(mind_map_text)
                cytoscapejs.st_cytoscapejs(
                    elements=elements,
                    # layout={"name": "breadthfirst"},
                    # style={"width": "100%", "height": "500px"},
                    stylesheet=[
                        {"selector": "node", "style": {"content": "data(label)", "font-size": "16px"}},
                        {"selector": "edge", "style": {"curve-style": "bezier", "target-arrow-shape": "triangle"}},
                    ],
                )
            except Exception as e:
                st.error(f"Error creating graph: {e}")

    with tab3:
        st.subheader("📚 Full Transcript")
        st.markdown(transcript)

    st.success("Done! Tags: " + tags if tags else "Done!")

else:
    st.info("Please upload or record an audio file to begin.")
