import os
import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyADh8pKp_qFmoJxh53JHU7MfiFkBDwwiRU"  # Replace with your key

# Streamlit UI
st.set_page_config(page_title="📰 Current Affairs AI Agent", layout="centered")
st.title("🧠 AI Agent for Current Affairs")
st.markdown("Ask any question about current events. The agent will search the web and respond using Gemini.")

# User input
query = st.text_input("Enter your current affairs question:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        try:
            # Define the tool
            search_tool = DuckDuckGoSearchRun()
            tools = [search_tool]

            # Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

            # ✅ Fixed Prompt (MUST include agent_scratchpad)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a smart assistant that uses tools like web search to answer user queries."),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # Create the agent
            agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

            # Create agent executor
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

            # Execute the agent
            with st.spinner("Thinking..."):
                result = agent_executor.invoke({"input": query})

            # Output
            st.success("✅ Answer:")
            st.write(result["output"])

            # Show reasoning
            with st.expander("🧩 Agent Reasoning & Tool Steps"):
                for step in result["intermediate_steps"]:
                    st.markdown(f"**Tool:** {step[0].tool}")
                    st.markdown(f"**Input:** {step[0].tool_input}")
                    st.markdown(f"**Output:** {step[1]}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error: {e}")
