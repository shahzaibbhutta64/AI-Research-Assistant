# --- SQLite Fix for Streamlit Cloud (MUST BE AT TOP) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------------------------------------------
# Helper: Get API Key safely
# -------------------------------------------------------------------
def get_groq_api_key():
    """
    Retrieves the Groq API Key from Streamlit Secrets (Cloud Deployment)
    or Environment Variables (Local Development).
    """
    # Check Streamlit Secrets first (Streamlit Community Cloud)
    if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
        return st.secrets["GROQ_API_KEY"]
    
    # Check Environment Variables (.env)
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return env_key
    
    return None

# -------------------------------------------------------------------
# Core Research Function
# -------------------------------------------------------------------
def run_research_crew(topic: str, api_key: str) -> str:
    """
    Executes the single-agent CrewAI research workflow.
    """
    # Set environment variable for tools/dependencies requiring it
    os.environ["GROQ_API_KEY"] = api_key

    # Initialize DuckDuckGo Search Tool
    search_tool = DuckDuckGoSearchRun()

    # Initialize Groq LLM using CrewAI's Native LLM wrapper
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )

    # Define Single Research Agent
    researcher = Agent(
        role="Research Analyst",
        goal=f"Research '{topic}' thoroughly using current web search results and produce an accurate, detailed report.",
        backstory=(
            "You are a skilled research analyst. Your strength lies in taking a topic, "
            "searching for up-to-date facts using web search tools, analyzing facts objectively, "
            "and synthesizing reliable information into structured, easy-to-read reports with source citations."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Define Research Task
    research_task = Task(
        description=(
            f"Conduct comprehensive research on the topic: '{topic}'.\n\n"
            "Steps to follow:\n"
            "1. Use DuckDuckGo Search to find relevant, reliable information and recent news about the topic.\n"
            "2. Analyze the key facts, developments, applications, benefits, challenges, and statistics.\n"
            "3. Organize your research logically.\n"
            "4. Construct a comprehensive report using the strict report structure required."
        ),
        expected_output=(
            "A structured markdown research report containing the following exact sections:\n"
            "# Research Report: [Topic Name]\n"
            "1. Executive Summary\n"
            "2. Introduction\n"
            "3. Background\n"
            "4. Key Findings\n"
            "5. Detailed Analysis\n"
            "6. Applications / Practical Examples\n"
            "7. Benefits\n"
            "8. Challenges and Limitations\n"
            "9. Future Developments\n"
            "10. Conclusion\n"
            "11. Sources (List title, summary, and URL of used web sources)"
        ),
        agent=researcher
    )

    # Instantiate and execute Crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)

# -------------------------------------------------------------------
# Streamlit User Interface
# -------------------------------------------------------------------
st.title("🔍 AI Research Agent")
st.caption("Powered by CrewAI, Groq (Llama-3.3-70b), and DuckDuckGo Search")

st.markdown("""
Enter any topic below. The AI Research Agent will search the web, analyze current information, 
and generate a structured research report complete with sources.
""")

# Input Field
topic_input = st.text_input(
    label="Research Topic",
    placeholder="e.g., Artificial Intelligence applications in workplace safety",
    help="Enter a clear topic or question you want the agent to investigate."
)

# Start Research Action
if st.button("Start Research", type="primary"):
    # Input Validation
    if not topic_input.strip():
        st.warning("⚠️ Please enter a valid research topic before starting.")
    else:
        api_key = get_groq_api_key()
        
        if not api_key:
            st.error(
                "❌ **Groq API Key Missing!**\n\n"
                "- **Local Testing:** Add `GROQ_API_KEY=your_key` to your `.env` file.\n"
                "- **Streamlit Cloud:** Add `GROQ_API_KEY = \"your_key\"` in **Settings > Secrets**."
            )
        else:
            try:
                # Progress / Status Indicator
                with st.status("🚀 Agent is conducting web research...", expanded=True) as status:
                    st.write("🔎 Querying DuckDuckGo for sources...")
                    st.write("🧠 Analyzing data with Groq Llama-3.3-70b...")
                    
                    # Execute research
                    report = run_research_crew(topic_input, api_key)
                    
                    status.update(label="✅ Research Complete!", state="complete", expanded=False)

                st.subheader("📊 Generated Research Report")
                st.markdown(report)

                # Download Buttons
                st.divider()
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report,
                    file_name=f"{topic_input.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during research execution: {str(e)}")
