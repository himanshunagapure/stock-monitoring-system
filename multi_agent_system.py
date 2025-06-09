import os
import glob
import uuid
from typing import Dict, List, Literal, TypedDict, Annotated, Optional, Any, Union
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Import agents
from agent.agent3 import StockMonitoringAgent
from agent.code_generator_agent2 import CodeGeneratorAgent

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=500,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True
)

# State definition for the multi-agent system
class AgentState(TypedDict):
    """State schema shared between all agents."""
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    session_id: str
    user_query: str
    required_tools: List[str]
    missing_tools: List[str]
    generated_tools: List[str]
    workflow_complete: bool
    next_agent: Optional[Literal["tool_creator", "stock_monitor", END]]

# Tool Selection Expert Node
def tool_selection_expert(state: AgentState) -> AgentState:
    """Analyzes user query and determines required tools."""
    print("\n🔍 Tool Selection Expert analyzing query...")
    
    # Get user query
    query = state["user_query"]
    
    # System message for tool analysis
    system_msg = SystemMessage(content="""You are a Tool Selection Expert.
Your job is to analyze user queries and determine what tools are needed to fulfill them.

Available Tool Categories:
1. Stock Price Tools:
   - get_stock_price.py: Get current stock price
   - check_condition.py: Check price conditions (above/below)

2. Company Info Tools:
   - get_company_financials.py: Get financial data
   - get_company_ceo.py: Get company CEO info

3. Notification Tools:
   - request_email_alert.py: Send email notifications

Analyze the query and list ALL tools needed to fulfill it completely.
Do NOT list tools that don't exist or aren't mentioned above.
List tools in a simple format, one per line, with just the filename (e.g. 'fetch_price.py').
Do not use numbers, backticks, or any other formatting.
""")
    
    # Get tool analysis from LLM
    messages = [
        system_msg,
        HumanMessage(content=f"Analyze this query and list all required tools: {query}")
    ]
    
    response = llm.invoke(messages)
    print("🔍🔍🔍tool_selection_expert response := ", response)
    
    # Get existing tools from both directories
    existing_tools = set()
    for tool_path in glob.glob("tools/*.py"):
        if "__" not in tool_path:
            existing_tools.add(os.path.basename(tool_path))
            
    for tool_path in glob.glob("dynamic_tools/*.py"):
        if "__" not in tool_path:
            existing_tools.add(os.path.basename(tool_path))
    
    # Extract required tools from LLM response
    required_tools = []
    for line in response.content.split("\n"):
        if ".py" in line:
            # Clean up the tool name by removing numbers, backticks, and extra spaces
            tool_name = line.split(".py")[0].strip()
            tool_name = ''.join(c for c in tool_name if not c.isdigit() and c != '`' and c != '.')
            tool_name = tool_name.strip() + ".py"
            required_tools.append(tool_name)
    
    # Determine missing tools
    missing_tools = [tool for tool in required_tools if tool not in existing_tools]
    
    print(f"📋 Required tools: {required_tools}")
    if missing_tools:
        print(f"❌ Missing tools: {missing_tools}")
        next_agent = "tool_creator"
    else:
        print("✅ All required tools are available")
        next_agent = "stock_monitor"
    
    # Add analysis result to messages
    analysis_msg = AIMessage(content=f"Tool Analysis:\nRequired tools: {required_tools}\nMissing tools: {missing_tools}")
    
    return {
        "messages": state["messages"] + [analysis_msg],
        "session_id": state["session_id"],
        "user_query": query,
        "required_tools": required_tools,
        "missing_tools": missing_tools,
        "generated_tools": [],
        "workflow_complete": False,
        "next_agent": next_agent
    }

# Tool Creator Node
def tool_creator(state: AgentState) -> AgentState:
    """Generates missing tools using code_generator_agent."""
    if not state["missing_tools"]:
        return {**state, "next_agent": "stock_monitor"}
    
    print("\n🛠️ Tool Creator generating missing tools...")
    
    # Initialize code generator agent
    code_gen = CodeGeneratorAgent()
    generated_tools = []
    generation_messages = []
    
    # Generate each missing tool
    for tool in state["missing_tools"]:
        print(f"\n📝 Generating tool: {tool}")
        result = code_gen.create_tool(
            f"Create a tool named {tool} that can handle this query: {state['user_query']}"
        )
        
        if result["status"] == "success":
            generated_tools.append(tool)
            generation_messages.append(AIMessage(content=f"✅ Generated tool: {tool}"))
            print(f"✅ Generated {tool} successfully")
        else:
            generation_messages.append(AIMessage(content=f"❌ Failed to generate {tool}: {result['message']}"))
            print(f"❌ Failed to generate {tool}: {result['message']}")
    
    return {
        **state,
        "messages": state["messages"] + generation_messages,
        "generated_tools": generated_tools,
        "workflow_complete": False,
        "next_agent": "stock_monitor"
    }

# Stock Monitor Node
def stock_monitor(state: AgentState) -> AgentState:
    """Executes the query using stock_monitoring_agent."""
    print("\n📊 Stock Monitor executing query...")
    
    print("🔍🔍🔍stock_monitor state(inside multi_agent_system) := ", state)
    # Initialize stock monitoring agent
    stock_agent = StockMonitoringAgent()
    
    # Create a session if needed
    session_id = state.get("session_id")
    if not session_id:
        session_id = stock_agent.create_session()
        print(f"📝 Created new session: {session_id}")
    
    # Build new user_query message
    tools_list = "', '".join(tool.replace(".py", "") for tool in state['required_tools'])
    user_query = f"{state['user_query']}. Use all these tools : '{tools_list}'"
    
    # Execute query
    result = stock_agent.run(user_query, session_id)
    print("🔍🔍🔍(Inside multi_agent_system) result from stock_monitoring_agent := ", result)
    
    # Add result to messages
    if isinstance(result, dict) and "messages" in result:
        messages = state["messages"] + result["messages"]
    else:
        messages = state["messages"] + [AIMessage(content=str(result))]
    
    print("🔍🔍🔍(Inside multi_agent_system) messages receieved from stock_monitoring_agent:= ", messages)
    return {
        **state,
        "messages": messages,
        "session_id": session_id,
        "workflow_complete": True,
        "next_agent": END
    }

def route_next(state: AgentState) -> Literal["tool_creator", "stock_monitor", END]:
    """Routes to the next agent based on state."""
    return state["next_agent"] or END

# Create the multi-agent workflow
def create_multi_agent_workflow() -> StateGraph:
    """Creates the LangGraph workflow for the multi-agent system."""
    
    # Create workflow with shared state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("tool_selection_expert", tool_selection_expert)
    workflow.add_node("tool_creator", tool_creator)
    workflow.add_node("stock_monitor", stock_monitor)
    
    # Add edges with conditional routing
    workflow.set_entry_point("tool_selection_expert")
    workflow.add_conditional_edges(
        "tool_selection_expert",
        route_next,
        {
            "tool_creator": "tool_creator",
            "stock_monitor": "stock_monitor"
        }
    )
    workflow.add_conditional_edges(
        "tool_creator",
        route_next,
        {
            "stock_monitor": "stock_monitor",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "stock_monitor",
        route_next,
        {END: END}
    )
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

class MultiAgentSystem:
    """Main class for the multi-agent system."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        # Create directories if they don't exist
        if not os.path.exists("dynamic_tools"):
            os.makedirs("dynamic_tools")
            with open("dynamic_tools/__init__.py", "w") as f:
                f.write("# Dynamic tools package\n")
        
        self.graph = create_multi_agent_workflow()
    
    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run a query through the multi-agent system."""
        try:
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "session_id": session_id or str(uuid.uuid4()),
                "user_query": query,
                "required_tools": [],
                "missing_tools": [],
                "generated_tools": [],
                "workflow_complete": False,
                "next_agent": None
            }
            print("⚒️⚒️⚒️initial_state", initial_state)
            # Configure checkpointer with required keys
            config = {
                "configurable": {
                    "thread_id": initial_state["session_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": None
                }
            }
            print("🌟🌟🌟config", config)
            # Run the workflow with configured checkpointer
            result = self.graph.invoke(initial_state, config=config)
            res = {
                "status": "success",
                "messages": result["messages"],
                "session_id": result["session_id"]
            }
            print("🚀🚀🚀res", res)
            return res
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }

# Create global instance
multi_agent_system = MultiAgentSystem()

def run_query(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run a query through the multi-agent system."""
    return multi_agent_system.run(query, session_id)

if __name__ == "__main__":
    # Test the system
    print("🤖 Testing Multi-Agent System")
    print("=" * 60)
    
    #test_queries = [
    #     "Get the current price of Apple stock",
    #     "Monitor Tesla stock and send me an email when it goes above $250",
    #     "Analyze sentiment from current news headlines for a stock INTC"
    # ]
    test_queries = [
        "Fetch Tesla stock price and send me an email to himunagapure114@gmail.com when it goes above $250"
    ]
    
    for query in test_queries:
        print(f"\n🚀 Testing query: {query}")
        print("-" * 50)
        
        result = run_query(query)
        
        if result["status"] == "success":
            print("\n✅ Query executed successfully")
            for msg in result["messages"]:
                if hasattr(msg, 'content'):
                    print(f"📋 {msg.content}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("=" * 60) 