# --- SQLite Patch for Streamlit Cloud (MUST BE AT VERY TOP) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env file if available (for local execution)
load_dotenv()

# CrewAI Imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------------------------------------------
# Custom DuckDuckGo Search Tool
# -------------------------------------------------------------------
@tool("DuckDuckGo Web Search")
def ddgs_search_tool(query: str) -> str:
    """
    Searches the web using DuckDuckGo to gather current facts, news, and details.
    Input should be a clear search query string.
    """
    try:
        # timeout=10 keeps a rate-limited/slow DDG response from hanging the
        # whole crew run indefinitely -- Streamlit Cloud's shared IPs get
        # rate-limited by DuckDuckGo more often than a home/dev machine does.
        results = list(DDGS(timeout=10).text(keywords=query, max_results=5))
        if not results:
            return "No relevant search results found."
        
        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get('title', 'No Title')
            href = res.get('href', 'No Link')
            body = res.get('body', 'No Description')
            formatted_results.append(f"{i}. {title}\n   URL: {href}\n   Snippet: {body}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        # Returned as a normal string (not raised) so the agent sees the
        # failure as a tool result and can retry or move on, instead of the
        # whole crew.kickoff() call dying / hanging.
        return f"Search execution error: {str(e)}. Try a shorter or different query."

# -------------------------------------------------------------------
# Helper: Retrieve API Key Safely
# -------------------------------------------------------------------
def get_groq_api_key():
    """
    Retrieves the Groq API Key from Streamlit Secrets (Cloud Deployment)
    or Environment Variables (Local Development).
    """
    # 1. Check Streamlit Secrets (Cloud Deployment)
    if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
        return st.secrets["GROQ_API_KEY"]
    
    # 2. Check Local Environment (.env file)
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return env_key
    
    return None

# -------------------------------------------------------------------
# Core Research Workflow (CrewAI Execution)
# -------------------------------------------------------------------
def run_research_crew(topic: str, api_key: str) -> str:
    """
    Executes the single-agent CrewAI research workflow.
    """
    # Set environment variable required by internal libraries
    os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq LLM via CrewAI's NATIVE "openai" provider, pointed at
    # Groq's OpenAI-compatible endpoint. This avoids routing through LiteLLM
    # entirely (the "groq/" prefix requires LiteLLM, which isn't installed
    # and isn't needed here since Groq speaks the OpenAI API format).
    #
    # NOTE on the model string: the FIRST "openai/" tells CrewAI to use its
    # native OpenAI-compatible client. Everything after that is passed
    # through verbatim as the model name Groq expects. Groq deprecated
    # llama-3.3-70b-versatile on 2026-08-16 and recommends openai/gpt-oss-120b
    # as the replacement -- and "openai/gpt-oss-120b" IS the real Groq model
    # ID (OpenAI is the model's publisher), so the double prefix is correct.
    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.3,
        # gpt-oss-120b is a reasoning model -- it spends tokens "thinking"
        # before it answers, which is why it's slower than the old
        # non-reasoning Llama-3.3-70b. "low" trims that thinking budget
        # for a big latency win; bump to "medium"/"high" if report quality
        # matters more than speed for your use case.
        reasoning_effort="low"
    )

    # Define Single Research Agent
    researcher = Agent(
        role="Research Analyst",
        goal=f"Research '{topic}' thoroughly using web search results and produce an accurate, detailed report.",
        backstory=(
            "You are a skilled research analyst. Your strength lies in taking a topic, "
            "searching for up-to-date facts using web search tools, analyzing facts objectively, "
            "and synthesizing reliable information into structured, easy-to-read reports with source citations."
        ),
        tools=[ddgs_search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Without these, a confused agent can loop on tool calls (re-running
        # searches, re-reasoning) far longer than expected with no feedback
        # to the user. This caps worst-case run time.
        max_iter=8,
        max_execution_time=180
    )

    # Define Research Task
    research_task = Task(
        description=(
            f"Conduct comprehensive research on the topic: '{topic}'.\n\n"
            "Steps to follow:\n"
            "1. Use DuckDuckGo Web Search to find relevant, reliable information and recent news about the topic.\n"
            "2. Analyze key facts, developments, applications, benefits, challenges, and statistics.\n"
            "3. Organize your research logically.\n"
            "4. Construct a comprehensive report using the exact report structure required."
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
st.caption("Powered by CrewAI, Groq (GPT-OSS-120B), and DuckDuckGo Search")

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
                    st.write("🧠 Analyzing data with Groq GPT-OSS-120B...")
                    
                    # Execute research
                    report = run_research_crew(topic_input, api_key)
                    
                    status.update(label="✅ Research Complete!", state="complete", expanded=False)

                st.subheader("📊 Generated Research Report")
                st.markdown(report)

                # Download Button
                st.divider()
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report,
                    file_name=f"{topic_input.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during research execution: {str(e)}")
