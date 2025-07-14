import streamlit as st
import pandas as pd
import plotly.express as px
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Streamlit setup
st.set_page_config(page_title="Indera's Multi-Agent AI", page_icon="🧠")
st.title("🧠 Indera's Multi-Agent Data Scientist")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

@st.cache_data
def read_data(file, file_type):
    if file_type == "csv":
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Upload your data and ask me anything!"}]

if uploaded_file:
    file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
    df = read_data(uploaded_file, file_type)
    st.success(f"✅ Data loaded! {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())

    llm = ChatOpenAI(temperature=0.3, model="gpt-4.1-mini")
    llmQuery = ChatOpenAI(model = "o4-mini")

    # Memory to keep the conversation context
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Manager Agent for intent detection
    manager_prompt = PromptTemplate(
        input_variables=["prompt"],
        template="""
                You are a classifier that decides the user's intent.
                Possible intents:
                - visualize
                - interpret
                - query

                User prompt: {prompt}

                Only output one word.
                """
    )
    manager_chain = LLMChain(llm=llm, prompt=manager_prompt)

    # Query Agent
    query_agent = create_pandas_dataframe_agent(
        llm, df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        allow_dangerous_code=True
    )

    # Interpreter Agent with memory
    interpreter_agent = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Visualizer prompt with fallback logic

    visualizer_prompt = PromptTemplate(
        input_variables=["prompt", "columns"],
        template="""
    You are a data visualization JSON generator.

    Your task is to output only a JSON object describing how to visualize the data.
    Your response **must be valid JSON**. Never explain, just return the JSON.

    Supported chart types:
    - "histogram"
    - "bar"
    - "scatter"
    - "line"
    - "pie"

    Supported filter operators:
    - "=="
    - "!="
    - ">"
    - "<"
    - ">="
    - "<="

    Instructions:
    - Use the `columns` list provided below to guide your selection.
    - If the user does not specify a column, choose a reasonable default from: [{columns}].
    - If the user specifies a filter (e.g., "hanya bulan juni", "khusus Jakarta"), include a `filter` block.
    - If unsure about the value or column, pick a probable match from the column list.

    Examples:

    User: "Buat histogram harga"
    Output:
    {{
    "chart_type": "histogram",
    "x": "harga"
    }}

    User: "Jumlah produk bulan Juni"
    Output:
    {{
    "chart_type": "bar",
    "x": "produk",
    "filter": {{
        "column": "bulan",
        "operator": "==",
        "value": "Juni"
    }}
    }}

    User prompt: {prompt}
    """
    )
    visualizer_chain = LLMChain(llm=llm, prompt=visualizer_prompt)

    import re
    def parse_visualization_json(js):
        js = js.strip()

        # Remove ```json or ``` wrappers if present
        js = re.sub(r"^```json", "", js)
        js = re.sub(r"```$", "", js)
        js = js.strip()

        if not js.startswith("{"):
            raise s("LLM did not return a valid JSON object.")

        return json.loads(js)

    def dynamic_plot(df, spec):
        chart = spec["chart_type"]
        x = spec.get("x") or df.columns[0]
        y = spec.get("y") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        filter = spec.get("filter")

        # 🧪 Apply filter logic
        if filter:
            # Example filter format: {"column": "region", "operator": "==", "value": "Jakarta"}
            try:
                col = filter["column"]
                op = filter["operator"]
                val = filter["value"]

                if op == "==":
                    df = df[df[col] == val]
                elif op == "!=":
                    df = df[df[col] != val]
                elif op == ">":
                    df = df[df[col] > val]
                elif op == "<":
                    df = df[df[col] < val]
                elif op == ">=":
                    df = df[df[col] >= val]
                elif op == "<=":
                    df = df[df[col] <= val]
                else:
                    raise ValueError("Unsupported operator")

            except Exception as e:
                st.warning(f"⚠️ Filter ignored: {e}")

        # 🎨 Generate chart
        if chart == "histogram":
            return px.histogram(df, x=x)
        if chart == "scatter":
            return px.scatter(df, x=x, y=y)
        if chart == "bar":
            vc = df[x].value_counts().reset_index()
            vc.columns = [x, "count"]
            return px.bar(vc, x=x, y="count")
        if chart == "pie":
            return px.pie(df, names=x)
        if chart == "line":
            return px.line(df, x=x, y=y)

        raise ValueError("Unknown chart type")


    # Display previous messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Tanya apa saja...")

    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # 1️⃣ Intent detection
                intent = manager_chain.run({"prompt": prompt}).strip().lower()
                st.caption(f"Intent detected: **{intent}**")

                # 2️⃣ Always query
                query_response = query_agent.run(prompt)
                st.markdown(query_response)
                st.session_state.messages.append({"role":"assistant","content":query_response})

                # 3️⃣ Visualization if intent is visualize or prompt mentions 'chart'
                if "visualize" in intent or "chart" in prompt.lower():
                    try:
                        json_spec_str = visualizer_chain.run({
                            "prompt": prompt,
                            "columns": ", ".join(df.columns)
                        }).strip()
                        st.code(json_spec_str, language="json")
                        spec = parse_visualization_json(json_spec_str)
                        fig = dynamic_plot(df, spec)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
                        st.session_state.messages.append({
                            "role":"assistant",
                            "content":f"⚠️ Error creating chart: {e}"
                        })

                # 4️⃣ Interpretation
                interp = interpreter_agent.run(f"Jelaskan hasil ini secara santai: {query_response}")
                with st.expander("See interpretation") : 
                    st.markdown(interp)
                st.session_state.messages.append({"role":"assistant","content":interp})
else:
    st.info("👆 Upload your data to get started.")
