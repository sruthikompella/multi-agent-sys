from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import os


llm = LLM(
    model="ollama/llama3",  
    base_url="http://localhost:11434",  
)


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web using DuckDuckGo."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            return "\n".join([f"{r['title']}: {r['href']}" for r in results])

web_search_tool = WebSearchTool()


researcher = Agent(
    role="Industry Researcher",
    goal="Research Apollo Pharmacy's industry and key offerings.",
    backstory="Expert in analyzing industries using web sources.",
    tools=[web_search_tool],
    llm=llm,
    verbose=True
)

use_case_generator = Agent(
    role="Use Case Generator",
    goal="Propose AI/GenAI use cases for Apollo Pharmacy.",
    backstory="Skilled in identifying AI-driven opportunities.",
    llm=llm,
    verbose=True
)

resource_collector = Agent(
    role="Resource Collector",
    goal="Collect datasets for proposed use cases.",
    backstory="Specialist in sourcing open datasets.",
    tools=[web_search_tool],
    llm=llm,
    verbose=True
)

proposal_writer = Agent(
    role="Proposal Writer",
    goal="Compile a final proposal for Apollo Pharmacy.",
    backstory="Experienced in creating actionable reports.",
    llm=llm,
    verbose=True
)


task1 = Task(
    description="Search the web for Apollo Pharmacy's industry (Retail Pharmacy) and key offerings.",
    agent=researcher,
    expected_output="A summary of Apollo Pharmacy's industry segment and strategic focus."
)

task2 = Task(
    description="Analyze AI/GenAI trends in retail pharmacy and propose 3 use cases for Apollo Pharmacy.",
    agent=use_case_generator,
    context=[task1],
    expected_output="A list of 3 AI/GenAI use cases with descriptions."
)

task3 = Task(
    description="Find datasets on Kaggle, HuggingFace, and GitHub for the proposed use cases.",
    agent=resource_collector,
    context=[task2],
    expected_output="A list of clickable resource links in markdown format."
)

task4 = Task(
    description="Compile a final proposal with market research, use cases, resources, feasibility, and references.",
    agent=proposal_writer,
    context=[task1, task2, task3],
    expected_output="A markdown report with all required sections."
)


crew = Crew(
    agents=[researcher, use_case_generator, resource_collector, proposal_writer],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":
    print("Starting Multi-Agent Workflow...")
    result = crew.kickoff()
    
    
    with open("apollo_pharmacy_proposal.md", "w") as f:
        f.write(str(result))  
    print("\n📄 Final Report saved to apollo_pharmacy_proposal.md")