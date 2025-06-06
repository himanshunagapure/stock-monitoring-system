# app.py - LangGraph Stock Monitoring with Streamlit

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import asdict

# Import the new LangGraph agent
from agents.stock_monitoring_agent import (
    create_new_session,
    run_stock_query,
    get_pending_confirmations,
    handle_confirmation,
    stock_agent
)

# Configure Streamlit
st.set_page_config(
    page_title="📈 Stock Alert AI - LangGraph Edition",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'agent_session_id' not in st.session_state:
        # Create a new agent session for this user
        st.session_state.agent_session_id = create_new_session(f"streamlit_user_{int(time.time())}")
        st.session_state.session_created_at = datetime.now()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'pending_confirmations' not in st.session_state:
        st.session_state.pending_confirmations = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False

init_session_state()

# Custom CSS for better UI
st.markdown("""
<style>
    .confirmation-box {
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
        margin: 20px 0;
    }
    
    .success-box {
        border: 2px solid #51cf66;
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
        margin: 20px 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4c6ef5;
        margin: 10px 0;
    }
    
    .example-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 10px 15px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid #4c6ef5;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left-color: #2196f3;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left-color: #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Smart Stock Monitoring System")
st.markdown("*Powered by LangGraph & Gemini AI with Human-in-the-Loop Email Confirmation*")

# Session info in sidebar
with st.sidebar:
    st.markdown("### 📊 Session Info")
    session_info = stock_agent.get_session_info(st.session_state.agent_session_id)
    if session_info:
        st.markdown(f"""
        **Session ID:** `{st.session_state.agent_session_id[:8]}...`
        
        **Created:** {st.session_state.session_created_at.strftime('%H:%M:%S')}
        
        **Stock Data:** {'✅ Available' if session_info.get('stock_data') else '❌ None'}
        
        **Pending Emails:** {session_info.get('pending_emails_count', 0)}
        """)
    
    if st.button("🔄 New Session"):
        # Create new session
        st.session_state.agent_session_id = create_new_session(f"streamlit_user_{int(time.time())}")
        st.session_state.session_created_at = datetime.now()
        st.session_state.chat_history = []
        st.session_state.pending_confirmations = []
        st.session_state.current_query = ""
        st.rerun()

# Check for pending email confirmations
def check_pending_confirmations():
    """Check and update pending confirmations"""
    confirmations = get_pending_confirmations(st.session_state.agent_session_id)
    st.session_state.pending_confirmations = confirmations

# Email Confirmation Interface
def show_email_confirmations():
    """Display email confirmation interface"""
    check_pending_confirmations()
    
    if st.session_state.pending_confirmations:
        st.markdown("---")
        st.markdown('<div class="confirmation-box">', unsafe_allow_html=True)
        st.markdown("### 📧 Email Confirmation Required")
        st.warning("🚨 The agent wants to send email alert(s). Please review and approve:")
        
        for i, email_req in enumerate(st.session_state.pending_confirmations):
            with st.expander(f"📧 Email Request #{i+1}", expanded=True):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **📧 Email:** `{email_req.email}`
                    
                    **📊 Subject:** {email_req.subject}
                    
                    **📝 Message:** {email_req.message}
                    
                    **🕒 Requested:** {email_req.created_at}
                    """)
                    
                    if email_req.stock_data:
                        try:
                            # Try different ways to access the stock data
                            if hasattr(email_req.stock_data, 'symbol'):
                                # If it's an object with attributes
                                st.markdown(f"""
                                **📈 Stock Info:**
                                - Symbol: {email_req.stock_data.symbol}
                                - Price: ${email_req.stock_data.price:.2f}
                                - Time: {email_req.stock_data.timestamp}
                                """)
                            elif hasattr(email_req.stock_data, '__dict__'):
                                # If it's an object that can be converted to dict
                                stock_dict = email_req.stock_data.__dict__
                                st.markdown(f"""
                                **📈 Stock Info:**
                                - Symbol: {stock_dict.get('symbol', 'N/A')}
                                - Price: ${stock_dict.get('price', 0):.2f}
                                - Time: {stock_dict.get('timestamp', 'N/A')}
                                """)
                            elif callable(getattr(email_req.stock_data, 'dict', None)):
                                # If it's a pydantic model or similar with .dict() method
                                stock_dict = email_req.stock_data.dict()
                                st.markdown(f"""
                                **📈 Stock Info:**
                                - Symbol: {stock_dict.get('symbol', 'N/A')}
                                - Price: ${stock_dict.get('price', 0):.2f}
                                - Time: {stock_dict.get('timestamp', 'N/A')}
                                """)
                            else:
                                # Fallback: display as string
                                st.markdown(f"""
                                **📈 Stock Info:** {str(email_req.stock_data)}
                                """)
                        except Exception as e:
                            st.markdown(f"""
                            **📈 Stock Info:** Error displaying stock data: {str(e)}
                            """)
                
                with col2:
                    st.markdown("**Actions:**")
                    
                    col_yes, col_no = st.columns(2)
                    
                    with col_yes:
                        if st.button("✅ Approve", key=f"approve_{email_req.id}", type="primary"):
                            handle_email_response(email_req.id, True)
                    
                    with col_no:
                        if st.button("❌ Decline", key=f"decline_{email_req.id}", type="secondary"):
                            handle_email_response(email_req.id, False)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return True
    
    return False

def handle_email_response(email_id: str, approved: bool):
    """Handle email confirmation response"""
    result = handle_confirmation(st.session_state.agent_session_id, email_id, approved)
    
    if result.get('success'):
        if approved:
            st.success(f"✅ {result['message']}")
        else:
            st.info(f"ℹ️ {result['message']}")
        
        # Remove from pending confirmations
        st.session_state.pending_confirmations = [
            req for req in st.session_state.pending_confirmations 
            if req.id != email_id
        ]
        
        # Add to chat history
        action = "approved and sent" if approved else "declined"
        st.session_state.chat_history.append({
            "type": "system",
            "content": f"📧 Email confirmation {action}: {result['message']}",
            "timestamp": datetime.now()
        })
        
        time.sleep(1)
        st.rerun()
    else:
        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

# Main Query Interface
def show_main_interface():
    """Show main query interface"""
    
    # Check for pending confirmations first
    has_pending = show_email_confirmations()
    
    if not has_pending:  # Only show main interface if no pending confirmations
        st.markdown("### 🧠 Ask About Stocks")
        
        # Query input
        user_query = st.text_area(
            "Enter your stock-related question:",
            value=st.session_state.current_query,
            height=100,
            placeholder="e.g., 'Get Apple stock price, check if above $150, and email me at user@example.com'"
        )
        
        # Buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Run Query", type="primary", disabled=st.session_state.processing):
                if user_query.strip():
                    run_query(user_query.strip())
                else:
                    st.error("Please enter a query first!")
        
        with col2:
            if st.button("🔄 Clear", disabled=st.session_state.processing):
                st.session_state.current_query = ""
                st.rerun()
        
        with col3:
            if st.button("📜 Clear History"):
                st.session_state.chat_history = []
                st.rerun()

def run_query(query: str):
    """Execute a query using the LangGraph agent"""
    st.session_state.current_query = query
    st.session_state.processing = True
    
    # Add user message to chat
    st.session_state.chat_history.append({
        "type": "user",
        "content": query,
        "timestamp": datetime.now()
    })
    
    with st.spinner("🤖 AI Agent is working..."):
        try:
            # Run the LangGraph agent
            result = run_stock_query(query, st.session_state.agent_session_id)
            
            # Process result
            if result and not result.get('error'):
                # Extract the agent's response
                messages = result.get('messages', [])
                if messages:
                    # Get the last AI message
                    ai_response = None
                    for msg in reversed(messages):
                        if hasattr(msg, 'content') and msg.content:
                            ai_response = msg.content
                            break
                    
                    if ai_response:
                        st.session_state.chat_history.append({
                            "type": "agent",
                            "content": ai_response,
                            "timestamp": datetime.now()
                        })
                    else:
                        st.session_state.chat_history.append({
                            "type": "agent",
                            "content": "Query completed successfully!",
                            "timestamp": datetime.now()
                        })
                else:
                    st.session_state.chat_history.append({
                        "type": "agent",
                        "content": "Query completed successfully!",
                        "timestamp": datetime.now()
                    })
                
                st.success("✅ Query completed!")
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                st.session_state.chat_history.append({
                    "type": "error",
                    "content": f"❌ Error: {error_msg}",
                    "timestamp": datetime.now()
                })
                st.error(f"❌ Error: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.chat_history.append({
                "type": "error",
                "content": f"❌ Error: {error_msg}",
                "timestamp": datetime.now()
            })
            st.error(f"❌ Error: {error_msg}")
    
    st.session_state.processing = False
    st.rerun()

# Chat History Display
def show_chat_history():
    """Display chat history"""
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        
        for i, msg in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
            timestamp = msg['timestamp'].strftime('%H:%M:%S')
            
            if msg['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You</strong> <small>({timestamp})</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'agent':
                st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>🤖 Agent</strong> <small>({timestamp})</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'system':
                st.markdown(f"""
                <div class="info-card">
                    <strong>ℹ️ System</strong> <small>({timestamp})</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'error':
                st.markdown(f"""
                <div class="chat-message" style="background: #ffebee; border-left-color: #f44336;">
                    <strong>⚠️ Error</strong> <small>({timestamp})</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)

# Example Queries
def show_examples():
    """Show example queries"""
    if not st.session_state.pending_confirmations:  # Only show if no pending confirmations
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        
        examples = [
            {
                "title": "📊 Basic Price Check",
                "query": "Get the current price of Apple stock (AAPL)"
            },
            {
                "title": "👨‍💼 CEO Information",
                "query": "Who is the CEO of Microsoft?"
            },
            {
                "title": "📧 Price Alert with Email",
                "query": "Monitor Tesla stock, check if it's above $200, and send alert to himunagapure114@gmail.com"
            },
            {
                "title": "📈 Comprehensive Analysis",
                "query": "Get Apple's current price, financial data, and tell me who the CEO is"
            },
            {
                "title": "🔔 Conditional Alert",
                "query": "Check if Google stock is below $120 and notify himunagapure114@gmail.com if true"
            }
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(
                    f"{example['title']}", 
                    key=f"example_{i}",
                    help=example['query'],
                    use_container_width=True
                ):
                    st.session_state.current_query = example['query']
                    st.rerun()

# Sidebar Instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Type your query** in the text area
    2. **Click "Run Query"** to execute
    3. **Review email confirmations** if prompted
    4. **View results** in the conversation history
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - ✅ Real-time stock prices
    - ✅ Price condition monitoring  
    - ✅ Email alerts (with confirmation)
    - ✅ Company CEO information
    - ✅ Financial data retrieval
    - ✅ Session management
    - ✅ Chat history
    """)
    
    st.markdown("### 🛡️ Email Safety")
    st.markdown("""
    **Human-in-the-Loop Protection:**
    - All emails require your approval
    - Review details before sending
    - Approve or decline each request
    - Complete transparency
    """)
    
    st.markdown("### 🔄 Session Management")
    st.markdown("""
    - **Isolated sessions** per user
    - **Persistent state** during session
    - **Automatic cleanup** of old data
    - **Multi-user support**
    """)

# Main Application Flow
def main():
    """Main application flow"""
    show_main_interface()
    show_chat_history()
    show_examples()

# Run the app
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🤖 <strong>LangGraph Stock Monitoring Agent</strong><br>
    <small>🤖 Powered by Gemini AI </small><br>
    <small>Made by Himanshu Nagapure</small>
</div>
""", unsafe_allow_html=True)