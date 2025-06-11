# stock_monitoring_agent.py (LangGraph + HITL + Session Management)

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, asdict
import yfinance as yf

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

from tools.get_stock_price import fetch_stock_price
from tools.check_condition import check_condition
from tools.request_email_alert import send_email_alert
from tools.get_company_ceo import get_company_ceo
from tools.get_company_financials import get_company_financials

from tool_registry import load_dynamic_tools

# ✅ Configure Gemini
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Try stable model
        temperature=0.2,
        max_tokens=500,
        google_api_key=os.getenv("GOOGLE_API_KEY"),  # Explicit API key
        convert_system_message_to_human=True
    )
    print("✅ LLM configured successfully")
except Exception as e:
    print(f"❌ LLM configuration failed: {e}")
    raise

# ✅ Session Management Classes
@dataclass
class StockData:
    symbol: str
    price: float
    timestamp: datetime
    
@dataclass
class EmailRequest:
    id: str
    email: str
    subject: str
    message: str
    stock_data: Optional[StockData]
    created_at: datetime
    status: Literal["pending", "approved", "rejected", "sent"] = "pending"

@dataclass
class UserSession:
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    stock_data: Optional[StockData] = None
    pending_emails: List[EmailRequest] = None
    
    def __post_init__(self):
        if self.pending_emails is None:
            self.pending_emails = []

class SessionManager:
    """Thread-safe session management with automatic cleanup"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self._sessions: Dict[str, UserSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Create a new session (with optional custom session_id)"""
        if not session_id:
            session_id = str(uuid.uuid4()) # for testing of file individually
        now = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now
        )
        
        self._sessions[session_id] = session
        self._cleanup_expired_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session and update last activity. Create if not exists."""
        session = self._sessions.get(session_id)
        if not session:
            self.create_session(session_id=session_id)
            session = self._sessions[session_id]
        else:
            session.last_activity = datetime.now()
        return session
    
    def update_stock_data(self, session_id: str, symbol: str, price: float):
        """Update stock data for session"""
        session = self.get_session(session_id)
        if session:
            session.stock_data = StockData(
                symbol=symbol,
                price=price,
                timestamp=datetime.now()
            )
    
    def add_email_request(self, session_id: str, email: str, subject: str, message: str) -> str:
        """Add email request to session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        email_request = EmailRequest(
            id=str(uuid.uuid4()),
            email=email,
            subject=subject,
            message=message,
            stock_data=session.stock_data,
            created_at=datetime.now()
        )
        
        session.pending_emails.append(email_request)
        return email_request.id
    
    def get_pending_email_requests(self, session_id: str) -> List[EmailRequest]:
        """Get all pending email requests for session"""
        session = self.get_session(session_id)
        if not session:
            return []
        return [req for req in session.pending_emails if req.status == "pending"]
    
    def update_email_status(self, session_id: str, email_id: str, status: str) -> bool:
        """Update email request status"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        for email_req in session.pending_emails:
            if email_req.id == email_id:
                email_req.status = status
                return True
        return False
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        
        for sid in expired_sessions:
            del self._sessions[sid]
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        self._sessions.pop(session_id, None)

# ✅ Global Session Manager Instance
session_manager = SessionManager()

# ✅ LangGraph State Definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    current_stock: Optional[Dict[str, Any]]
    pending_confirmations: List[Dict[str, Any]]
    user_input: str
    workflow_complete: bool

# ✅ Enhanced Tool Functions
def get_stock_price_tool(symbol: str, session_id: str = None) -> str:
    """Get stock price and store in session"""
    try:
        symbol = symbol.strip().upper()
        print(f"🔍 Getting price for {symbol} (Session: {session_id})")
        
        stock = yf.Ticker(symbol)
        price = stock.info.get("regularMarketPrice")
        
        if not price:
            price = fetch_stock_price(symbol)
        
        if price:
            price = float(price)
            # Store in session if session_id provided
            if session_id:
                session_manager.update_stock_data(session_id, symbol, price)
            
            return f"✅ {symbol} current price: ${price:.2f}"
        else:
            return f"❌ Could not fetch price for {symbol}"
            
    except Exception as e:
        return f"❌ Error getting price for {symbol}: {str(e)}"

def check_condition_tool(condition: str, target_price: float, session_id: str = None) -> str:
    """Check price condition using session data"""
    try:
        if not session_id:
            return "❌ Session ID required for condition checking"
        
        session = session_manager.get_session(session_id)
        if not session or not session.stock_data:
            return "❌ No stock data found. Please fetch stock price first."
        
        stock_data = session.stock_data
        current_price = stock_data.price
        symbol = stock_data.symbol
        
        condition = condition.lower().strip()
        rule = {"condition": condition, "target_price": target_price}
        
        print(f"🔍 Checking: {symbol} ${current_price} {condition} ${target_price}")
        
        result = check_condition(current_price, rule)
        
        if result:
            return f"✅ CONDITION MET: {symbol} (${current_price:.2f}) IS {condition.upper()} ${target_price}"
        else:
            return f"❌ CONDITION NOT MET: {symbol} (${current_price:.2f}) is NOT {condition.upper()} ${target_price}"
            
    except Exception as e:
        return f"❌ Error checking condition: {str(e)}"

def request_email_confirmation_tool(email: str, session_id: str = None) -> str:
    """Request email confirmation - creates pending request"""
    try:
        if not session_id:
            return "❌ Session ID required"
        
        session = session_manager.get_session(session_id)
        if not session:
            return "❌ Invalid session"
        
        # Create email content
        if session.stock_data:
            subject = f"Stock Alert: {session.stock_data.symbol}"
            message = f"{session.stock_data.symbol} is currently at ${session.stock_data.price:.2f}"
        else:
            subject = "Stock Alert"
            message = "Stock monitoring alert"
        
        # Add to pending emails
        email_id = session_manager.add_email_request(session_id, email, subject, message)
        
        return f"📧 Email confirmation requested for {email}. Email ID: {email_id}. Waiting for user approval."
        
    except Exception as e:
        return f"❌ Error requesting email confirmation: {str(e)}"

def get_ceo_tool(symbol: str) -> str:
    """Get company CEO"""
    try:
        return get_company_ceo(symbol.strip().upper())
    except Exception as e:
        return f"❌ Error getting CEO for {symbol}: {str(e)}"

def get_financials_tool(symbol: str) -> str:
    """Get company financials"""
    try:
        return get_company_financials(symbol.strip().upper())
    except Exception as e:
        return f"❌ Error getting financials for {symbol}: {str(e)}"

# ✅ LangGraph Tools with Session Context
def create_session_aware_tools(session_id: str) -> List[Tool]:
    """Create tools that are aware of the current session"""
    # Static/built-in tools
    static_tools = [
        Tool(
            name="get_stock_price",
            func=lambda symbol: get_stock_price_tool(symbol, session_id),
            description="Get current stock price. Input: stock symbol (e.g., 'AAPL')"
        ),
        Tool(
            name="check_condition",
            func=lambda condition_and_price: check_condition_tool(
                condition_and_price.split(',')[0].strip(),
                float(condition_and_price.split(',')[1].strip()),
                session_id
            ),
            description="Check if stock meets condition. Input: 'condition,target_price' (e.g., 'above,300')"
        ),
        Tool(
            name="request_email_alert",
            func=lambda email: request_email_confirmation_tool(email, session_id),
            description="Request email alert (requires confirmation). Input: email address"
        ),
        Tool(
            name="get_company_ceo",
            func=get_ceo_tool,
            description="Get company CEO name. Input: stock symbol"
        ),
        Tool(
            name="get_company_financials",
            func=get_financials_tool,
            description="Get company financial data. Input: stock symbol"
        )
    ]
    
    # Load dynamic tools
    # dynamic_tools = load_dynamic_tools()
    # print("\n 🌟🚀Dynamic tools = ",dynamic_tools)
    # print("\n 🌟🚀Static tools = ",static_tools)
    
    # Load dynamic tools with enhanced error handling
    try:
        print("🔄 Loading dynamic tools...")
        dynamic_tools = load_dynamic_tools()
        print(f"✅ Successfully loaded {len(dynamic_tools)} dynamic tools")
        
        # Debug: Print loaded tool info
        for tool in dynamic_tools:
            print(f"   🧰 {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"❌ Error loading dynamic tools: {str(e)}")
        dynamic_tools = []
        
    # Combine static and dynamic tools
    all_tools = static_tools + dynamic_tools
    
    print(f"📊 Total tools loaded: {len(all_tools)} (Static: {len(static_tools)}, Dynamic: {len(dynamic_tools)})")
    
    return all_tools
    

# ✅ LangGraph Nodes
def agent_node(state: AgentState):
    """Main agent reasoning node"""
    session_id = state["session_id"]
    messages = state["messages"]
    
    # Create session-aware tools
    tools = create_session_aware_tools(session_id)
    
    if not tools:
        return {
            "messages": [AIMessage(content="❌ No tools available. Please check tool configuration.")],
            "workflow_complete": True
        }

    # Get tool names for system message
    tool_names = [tool.name for tool in tools]
    print("\n🚀All Tool Names (inside tool user):", tool_names)
    
    # System message with instructions
    system_msg = SystemMessage(content=f"""You are an intelligent stock monitoring assistant.

Available tools: {', '.join(tool_names)}

CAPABILITIES:
- Get real-time stock prices
- Check price conditions (above, below, equal)
- Request email alerts (requires user confirmation)
- Get company CEO information
- Get company financial data
- Access dynamically loaded tools for specialized tasks

AUTONOMOUS OPERATION GUIDELINES:
- Analyze the user's request carefully to understand their true intent
- Always use the most appropriate tool for the task
- Review all available tools and determine which ones are relevant
- If a dynamic tool is available for a specific request (like news analysis), prefer it over generic response
- Design your own workflow to accomplish the user's goals most effectively
- You have complete freedom to decide the sequence and combination of tools to us
- Be creative in combining tools to provide comprehensive answers
- Think creatively about how to combine tools for complex requests
Only use tools that are actually needed - don't use unnecessary tools

WORKFLOW GUIDELINES:
1. Understand the user's request completely
2. Use all required tools in correct order
3. Use tools strategically to gather needed information
4. For email alerts, use 'request_email_alert' tool - this will create a pending request
5. Be clear about what actions require confirmation
6. Provide comprehensive responses with all requested information

IMPORTANT: 
- When calling tools, use the exact parameter names as specified in the tool schema
- Do not use generic 'args' or 'kwargs' parameter names
- Pass parameters as direct key-value pairs
- For example, if a tool expects 'symbol', pass it as {{"symbol": "AAPL"}}

Current session: {session_id}
""")
    
    # Convert tools to Gemini format
    formatted_tools = []
    for tool in tools:
        # Get function signature if available
        if hasattr(tool, 'func'):
            import inspect
            sig = inspect.signature(tool.func)
            
            # Build parameter schema
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    # Get parameter type and default value
                    param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "string"
                    default_value = param.default if param.default != inspect.Parameter.empty else None
                    
                    # Get parameter description from docstring
                    param_desc = f"Parameter {param_name}"
                    if hasattr(tool.func, '__doc__') and tool.func.__doc__:
                        doc_lines = tool.func.__doc__.split('\n')
                        for line in doc_lines:
                            if param_name in line:
                                param_desc = line.strip()
                                break
                    
                    # Add parameter to schema
                    properties[param_name] = {
                        "type": "string",
                        "description": param_desc
                    }
                    
                    # Add to required if no default value
                    if default_value is None:
                        required.append(param_name)
            
            # Create tool schema
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            # Add example if available
            if hasattr(tool.func, '__doc__') and tool.func.__doc__:
                doc_lines = tool.func.__doc__.split('\n')
                for line in doc_lines:
                    if "Example" in line:
                        tool_schema["examples"] = [line.strip()]
                        break
            
            # Add explicit parameter example
            if len(properties) == 1:
                param_name = list(properties.keys())[0]
                tool_schema["examples"] = [f'{{"{param_name}": "value"}}']
            
            formatted_tools.append(tool_schema)
    
    print("\n🌟🌟🌟 formatted_tools = ", formatted_tools)
    
    # Bind tools to LLM with proper format
    llm_with_tools = llm.bind(tools=formatted_tools)
    
    # Get response from LLM
    response = llm_with_tools.invoke([system_msg] + messages)
    print("\n🌟🌟🌟 response = ", response)
    print("\n🌟🌟🌟 response.tool_calls = ", response.tool_calls)
    
    # Check if tools need to be called
    if response.tool_calls:
        # Execute tools
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            print(f'\n 🌟🌟🌟 Tool call = {tool_call}, tool name = {tool_name} and tool_args = {tool_args}')
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # Handle different argument formats
                        if isinstance(tool_args, dict):
                            # Remove any empty or None values
                            tool_args = {k: v for k, v in tool_args.items() if v not in (None, '', {})}
                            
                            if not tool_args:
                                # No arguments provided, call function without args
                                result = tool.func()
                            else:
                                # Get function signature
                                sig = inspect.signature(tool.func)
                                param_names = list(sig.parameters.keys())
                                
                                # If function expects a single parameter and we have a single value
                                if len(param_names) == 1 and len(tool_args) == 1:
                                    # Pass the single value directly
                                    result = tool.func(list(tool_args.values())[0])
                                else:
                                    # Pass arguments as keyword arguments
                                    result = tool.func(**tool_args)
                        else:
                            # Direct argument
                            result = tool.func(tool_args)
                        
                        print(f'\n 🌟 Tool Name = {tool_name}, tool args = {tool_args}, function result = {result}')
                        tool_results.append(f"Tool '{tool_name}' result: {result}")
                    except Exception as e:
                        error_msg = f"Tool '{tool_name}' error: {str(e)}"
                        print(f"❌ {error_msg}")
                        tool_results.append(error_msg)
                    break
        
        # Create a comprehensive response
        final_response = response.content or ""
        if tool_results:
            final_response += "\n\n" + "\n".join(tool_results)
        print('\n\n 🌟🌟🌟 Final_result = ', tool_results)
        
        return {
            "messages": [AIMessage(content=final_response)],
            "workflow_complete": True
        }
    
    return {
        "messages": [response],
        "workflow_complete": True
    }

def check_confirmations_node(state: AgentState):
    """Check for pending confirmations"""
    session_id = state["session_id"]
    
    # Get pending email requests
    pending_emails = session_manager.get_pending_email_requests(session_id)
    
    if pending_emails:
        return {
            "pending_confirmations": [asdict(req) for req in pending_emails],
            "workflow_complete": False  # Don't complete if confirmations pending
        }
    
    return {"pending_confirmations": []}

def should_continue(state: AgentState) -> Literal["check_confirmations", END]:
    """Decide whether to continue or end"""
    if state.get("workflow_complete", False):
        return "check_confirmations"
    return END

def confirmation_routing(state: AgentState) -> Literal["agent", END]:
    """Route based on confirmations"""
    if state.get("pending_confirmations"):
        return END  # Exit to handle confirmations in UI
    return END

# ✅ Build LangGraph
def create_stock_monitoring_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("check_confirmations", check_confirmations_node)
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue,
        {"check_confirmations": "check_confirmations", END: END}
    )
    workflow.add_conditional_edges("check_confirmations", confirmation_routing,
        {"agent": "agent", END: END}
    )
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory, interrupt_before=["check_confirmations"])

# ✅ Main Agent Class
class StockMonitoringAgent:
    """Main agent class with session management"""
    
    def __init__(self):
        self.graph = create_stock_monitoring_graph()
    
    def create_session(self, session_id: Optional[str] = None, user_id: str = None) -> str:
        """Create a new user session"""
        return session_manager.create_session(session_id, user_id)
    
    def run(self, user_input: str, session_id: str, thread_id: str = None):
        """Run the agent with session context"""
        if not thread_id:
            thread_id = session_id
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
            "user_input": user_input,
            "current_stock": None,
            "pending_confirmations": [],
            "workflow_complete": False
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state, config=config)
            return result
        except Exception as e:
            print(f"Detailed error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Agent execution failed: {str(e)}"}
    
    def get_pending_confirmations(self, session_id: str) -> List[EmailRequest]:
        """Get pending email confirmations for session"""
        return session_manager.get_pending_email_requests(session_id)
    
    def handle_email_confirmation(self, session_id: str, email_id: str, approved: bool) -> Dict[str, Any]:
        """Handle email confirmation response"""
        try:
            if approved:
                # Update status to approved
                session_manager.update_email_status(session_id, email_id, "approved")
                
                # Get the email request details
                session = session_manager.get_session(session_id)
                if session:
                    email_req = next((req for req in session.pending_emails if req.id == email_id), None)
                    if email_req:
                        # Send the email
                        send_email_alert(email_req.email, email_req.subject, email_req.message)
                        session_manager.update_email_status(session_id, email_id, "sent")
                        
                        return {
                            "success": True,
                            "message": f"✅ Email sent successfully to {email_req.email}",
                            "email_id": email_id
                        }
            else:
                # Update status to rejected
                session_manager.update_email_status(session_id, email_id, "rejected")
                return {
                    "success": True,
                    "message": "❌ Email sending cancelled by user",
                    "email_id": email_id
                }
            
            return {"success": False, "message": "Email request not found"}
            
        except Exception as e:
            return {"success": False, "message": f"Error handling confirmation: {str(e)}"}
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "stock_data": asdict(session.stock_data) if session.stock_data else None,
                "pending_emails_count": len([req for req in session.pending_emails if req.status == "pending"])
            }
        return None
    
    def cleanup_session(self, session_id: str):
        """Clean up a session"""
        session_manager.delete_session(session_id)

# ✅ Create Global Agent Instance
stock_agent = StockMonitoringAgent()

# ✅ Convenience Functions for External Usage
def create_new_session(session_id: Optional[str] = None, user_id: str = None) -> str:
    """Create a new session - convenience function"""
    return stock_agent.create_session(session_id, user_id)

def run_stock_query(user_input: str, session_id: str) -> Dict[str, Any]:
    """Run a stock query - convenience function"""
    return stock_agent.run(user_input, session_id)

def get_pending_confirmations(session_id: str) -> List[EmailRequest]:
    """Get pending confirmations - convenience function"""
    return stock_agent.get_pending_confirmations(session_id)

def handle_confirmation(session_id: str, email_id: str, approved: bool) -> Dict[str, Any]:
    """Handle confirmation - convenience function"""
    return stock_agent.handle_email_confirmation(session_id, email_id, approved)

# ✅ CLI Testing Function     
def test_agent():
    """Enhanced test function"""
    print("🤖 TESTING ENHANCED STOCK MONITORING AGENT")
    print("=" * 60)
    
    session_id = create_new_session("test_user")
    print(f"📝 Session: {session_id}")
    
    # test_queries = [
    #     "Get Apple stock price",
    #     "Check if Google stock is below $120", 
    #     "Monitor Tesla stock and send me an email at himunagapure114@gmail.com, when it goes above $250. Use these tools = 'fetch_price.py', 'check_condition.py', 'notify.py'"
    # ]
    
    #test_queries = ["Monitor Tesla stock and send me an email at himunagapure114@gmail.com, when it goes above $250. Use these tools = 'fetch_price.py', 'check_condition.py', 'notify.py' "]
    test_queries = ["Analyze sentiment from recent news headlines for Intel (INTC)"]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🚀 Test {i}: {query}")
        print("-" * 50)
        
        result = run_stock_query(query, session_id)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            for msg in result.get("messages", []):
                if hasattr(msg, 'content'):
                    print(f"📋 {msg.content}")
        
        print("🌟🌟Checking for pending confirmations...")
        confirmations = get_pending_confirmations(session_id)
        if confirmations:
            print(f"\n📧 Pending confirmations: {len(confirmations)}")
            for conf in confirmations:
                print(f"  - To: {conf.email}")
                print(f"  - Subject: {conf.subject}")
                print(f"  - ID: {conf.id}")

if __name__ == "__main__":
    test_agent()