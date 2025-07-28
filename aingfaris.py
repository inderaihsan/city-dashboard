import streamlit as st
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os

from data.prompt import general_prompt

# --- Load from secrets ---
db_user = st.secrets["database"]["user"]
db_pass = st.secrets["database"]["password"]
db_host = st.secrets["database"]["host"]
db_port = st.secrets["database"]["port"]
db_name = st.secrets["database"]["name"]
db_schema = st.secrets["database"]["schema"]

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# --- Build DB URI ---
db_uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# --- LangChain setup ---
llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini", openai_api_key=openai_api_key)
db = SQLDatabase.from_uri(db_uri, include_tables=None, schema=db_schema)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    prefix=general_prompt,
)

# --- Streamlit UI ---
st.title("🏡 Property Q&A with LangChain + PostgreSQL")
st.markdown("Ask anything about your property database!")

query = st.text_input("❓ Ask your question:", placeholder="e.g. Show the 5 cheapest properties in Jakarta")

if st.button("Run") or query:
    with st.spinner("Running query..."):
        try:
            result = agent_executor.run(query)
            st.success("✅ Result:")
            st.markdown(result)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
