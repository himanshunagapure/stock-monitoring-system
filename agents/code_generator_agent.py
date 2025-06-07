#code-generator-agent.py
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from api_key_config import API_KEY_MAPPING

# Load environment variables
load_dotenv()

class RequestValidator:
    """Validates user requests for security, feasibility, and API requirements."""
    
    def __init__(self):
        # Security-related keywords that should be rejected
        self.security_blocklist = [
            'hack', 'hacking', 'exploit', 'vulnerability', 'breach', 'crack', 'break into',
            'unauthorized access', 'bypass security', 'sql injection', 'xss', 'malware',
            'virus', 'trojan', 'backdoor', 'keylogger', 'phishing', 'ddos', 'botnet',
            'social engineering', 'password cracking', 'brute force', 'penetration test',
            'reverse engineer', 'decompile', 'insider trading', 'market manipulation',
            'pump and dump', 'front running', 'wash trading', 'spoofing'
        ]
        
        # Impossible/unrealistic requests - enhanced patterns
        self.impossible_keywords = [
            'predict with 100% accuracy', 'guaranteed profit', 'risk-free trading',
            'money printing machine', 'get rich quick', 'never lose money',
            'perfect prediction', 'crystal ball', 'time travel', 'future prices exactly',
            'beat the market always', 'infinite money', 'free money generator',
            'magic formula', 'secret algorithm that always wins', 'reads my mind',
            'mind reading', 'telepathic', 'psychic', '100% accurate', 'never wrong',
            'always profitable', 'guarantee success', 'foolproof strategy'
        ]
        
        # Premium/restricted data sources - enhanced list
        self.premium_sources = [
            'bloomberg terminal', 'bloomberg professional', 'bloomberg api', 'bloomberg data',
            'refinitiv', 'factset', 'morningstar direct', 'reuters eikon', 'reuters api',
            'quandl premium', 'iex cloud premium', 'insider information', 'private data',
            'confidential reports', 'exclusive access', 'proprietary data', 'paid subscription only',
            'enterprise data', 'institutional data', 'level 2 data', 'real-time options',
            'options chain', 'level ii data', 'tick data', 'microsecond data',
            'high frequency data', 'professional data feed'
        ]
        
        # Data-heavy operations that require premium APIs
        self.heavy_operations = [
            'analyze 1000 stocks', 'analyze hundreds of stocks', 'scan entire market',
            'backtest', 'backtesting', 'historical backtesting', 'strategy backtesting',
            'portfolio backtesting', 'analyze 10 years', '10 year analysis',
            'bulk analysis', 'mass screening', 'market screening', 'entire stock market',
            'all stocks', 'every stock', 'thousands of tickers', 'full market scan'
        ]
        
        # Social media APIs that require keys
        self.social_media_apis = [
            'twitter', 'reddit', 'social media', 'sentiment analysis', 'social sentiment',
            'twitter mentions', 'reddit mentions', 'social media mentions', 'tweet analysis',
            'social buzz', 'social media data', 'twitter api', 'reddit api'
        ]
    
    def validate_request(self, request: str) -> Tuple[bool, str, List[str]]:
        """
        Validate a user request comprehensively.
        
        Returns:
            Tuple[bool, str, List[str]]: (is_valid, reason_if_invalid, required_apis)
        """
        request_lower = request.lower()
        
        # Check for security violations
        for keyword in self.security_blocklist:
            if keyword in request_lower:
                return False, f"🚫 Security violation: Request contains '{keyword}'. Cannot create tools for hacking, unauthorized access, or illegal trading activities.", []
        
        # Check for impossible requests
        for keyword in self.impossible_keywords:
            if keyword in request_lower:
                return False, f"🚫 Impossible request: '{keyword}' is not technically feasible. Financial markets are inherently unpredictable and no system can guarantee perfect accuracy.", []
        
        # Check for premium/restricted data requests
        for source in self.premium_sources:
            if source in request_lower:
                return False, f"🚫 Premium data request: '{source}' requires expensive paid subscriptions or special access that we don't support. Consider using free alternatives like yfinance for basic market data.", []
        
        # Check for data-heavy operations
        for operation in self.heavy_operations:
            if operation in request_lower:
                return False, f"🚫 Data-heavy operation: '{operation}' requires premium API subscriptions with higher rate limits. Free APIs cannot handle such large-scale operations efficiently.", []
        
        # Check for social media requirements
        for social_api in self.social_media_apis:
            if social_api in request_lower:
                return False, f"🚫 Social media data: '{social_api}' requires specific API keys (Twitter API, Reddit API, etc.) that are not configured. We cannot create placeholder implementations for missing APIs.", []
        
        # Check if request is too vague
        if len(request.strip().split()) < 3:
            return False, "🚫 Request too vague: Please provide a more detailed description of what you want to build.", []
        
        return True, "", []

class APIRequirementAnalyzer:
    """Analyzes API requirements using LLM to determine if request is feasible."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_request_feasibility(self, user_request: str, available_apis: Dict) -> Tuple[bool, str, List[str]]:
        """
        Use LLM to analyze if the request can be fulfilled with available APIs.
        
        Returns:
            Tuple[bool, str, List[str]]: (is_feasible, reason_if_not, required_apis)
        """
        
        # Create a prompt to analyze the request
        analysis_prompt = f"""
Analyze this user request for a financial tool: "{user_request}"

Available APIs and their capabilities:
{json.dumps(available_apis, indent=2)}

Determine if this request can be fulfilled with the available APIs. Consider:

1. Does the request require data sources we don't have access to?
2. Does it require premium/paid APIs that aren't available?
3. Does it require social media APIs (Twitter, Reddit) that need special keys?
4. Does it require real-time or high-frequency data beyond free API limits?
5. Does it require data-heavy operations (analyzing thousands of stocks) that would exceed free API limits?
6. Does it make impossible claims (100% accuracy, guaranteed profits)?

Respond in this exact JSON format:
{{
    "feasible": true/false,
    "reason": "explanation if not feasible",
    "required_apis": ["list", "of", "api", "names", "needed"],
    "missing_capabilities": ["list", "of", "missing", "features"],
    "alternative_suggestion": "suggest free alternatives if applicable"
}}

Be strict - if the request cannot be properly fulfilled with available free APIs, mark it as not feasible.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a financial API analysis expert. Analyze the request and respond in the exact JSON format specified."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.model.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
            else:
                json_part = response_text
            
            # Parse the JSON response
            analysis = json.loads(json_part)
            
            return (
                analysis.get("feasible", False),
                analysis.get("reason", "Analysis failed"),
                analysis.get("required_apis", [])
            )
            
        except Exception as e:
            # If LLM analysis fails, be conservative and reject complex requests
            return False, f"Unable to analyze request complexity: {str(e)}", []

class CodeGeneratorAgent:
    def __init__(self):
        """Initialize the Code Generator Agent with comprehensive validation."""
        # Configure Gemini API
        if not os.getenv('GOOGLE_API_KEY'):
            raise Exception("GOOGLE_API_KEY is required for the Code Generator Agent to function.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            convert_system_message_to_human=True
        )
        
        # Initialize validators
        self.validator = RequestValidator()
        self.api_analyzer = APIRequirementAnalyzer(self.model)
        
        # API Key Mapping
        self.api_key_mapping = API_KEY_MAPPING
        
        # Available APIs catalog with strict requirements
        self.available_apis = {
            'yfinance': {
                'description': 'Free stock data and financial information',
                'requires_key': False,
                'import_statement': 'import yfinance as yf',
                'limitations': 'Rate limited to ~2000 requests/hour per IP. No real-time data. Delayed quotes.',
                'capabilities': ['stock prices', 'historical data', 'company info', 'financial statements'],
                'cannot_do': ['real-time options', 'tick data', 'level 2 data', 'social sentiment']
            },
            'newsapi': {
                'description': 'News headlines and articles (free tier: 100 requests/day)',
                'requires_key': True,
                'key_name': 'NEWSAPI_KEY',
                'import_statement': 'import requests',
                'limitations': 'Free tier: 100 requests/day, 1 month history max',
                'capabilities': ['news headlines', 'article snippets', 'source filtering'],
                'cannot_do': ['full article content', 'real-time news', 'social media data']
            },
            'alphavantage': {
                'description': 'Financial data and stock information (free tier: 5 requests/minute)',
                'requires_key': True,
                'key_name': 'ALPHAVANTAGE_KEY',
                'import_statement': 'import requests',
                'limitations': 'Free tier: 5 requests/minute, 500 requests/day',
                'capabilities': ['stock data', 'forex', 'crypto', 'technical indicators'],
                'cannot_do': ['bulk operations', 'real-time options', 'social data']
            },
            'email_smtp': {
                'description': 'Send emails via SMTP (Gmail)',
                'requires_key': True,
                'key_name': ['EMAIL_APP', 'EMAIL_APP_PASSWORD'],
                'import_statement': 'import smtplib\nfrom email.mime.text import MIMEText\nfrom email.mime.multipart import MIMEMultipart',
                'limitations': 'Gmail app passwords required, rate limits apply',
                'capabilities': ['send notifications', 'email reports', 'alerts'],
                'cannot_do': ['receive emails', 'email parsing', 'bulk emailing']
            }
        }
        
        self.tools_directory = "generated_tools"
        self.registry_file = "tool_registry.json"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories and files."""
        if not os.path.exists(self.tools_directory):
            os.makedirs(self.tools_directory)
        
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                json.dump({"tools": []}, f)
    
    def detect_api_requirements(self, user_request: str) -> List[str]:
        """Detect which APIs are needed and validate they're sufficient."""
        request_lower = user_request.lower()
        needed_apis = []
        
        # More precise API detection
        api_patterns = {
            'newsapi': ['news', 'headlines', 'articles', 'latest news', 'financial news',
                'trending news', 'top news', 'recent news', 'news headlines',
                'news articles', 'breaking news', 'market news', 'company news'],
            'alphavantage': ['technical indicators', 'forex', 'cryptocurrency', 'crypto', 'moving average','ma'
                'rsi', 'macd', 'bollinger bands', 'ema', 'sma'],
            'yfinance': ['stock price', 'stock prices', 'current price', 'share price', 'share prices',
                'stock data', 'ticker', 'market data', 'financial info', 'earnings', 
                'balance sheet', 'stock information', 'price of', 'get price', 'fetch price',
                'stock value', 'current stock', 'stock quote', 'quote for', 'price for',
                'stock current price', 'current price of', 'price data', 'market price'],
            'email_smtp': [
                'send email', 'send mail', 'email me', 'mail me', 'send notification email',
                'sends email', 'sends mail', 'emails me', 'mails me', 'sends notification email',
                'email notification', 'send via email', 'email alert', 'mail alert',
                'emails notification', 'sends via email', 'emails alert', 'mails alert',
                'email the result', 'mail the result', 'send me an email', 'send me a mail',
                'sends me an email', 'sends me a mail',
                'email report', 'mail report', 'send report via email', 'email summary',
                'emails report', 'mails report', 'sends report via email', 'emails summary',
                'mail summary', 'send summary via email', 'sends summary via email','via email'
            ]
        }
        
        # Check ALL patterns for ALL APIs (don't return early)
        for api, patterns in api_patterns.items():
            for pattern in patterns:
                if pattern in request_lower:
                    if api not in needed_apis:  # Avoid duplicates
                        needed_apis.append(api)
                    break  # Found a match for this API, check next API
        
        # Default to yfinance for stock-related requests
        if not needed_apis and any(word in request_lower for word in ['stock', 'ticker', 'price', 'market', 'company']):
            needed_apis.append('yfinance')
            
        return needed_apis
    
    def _comprehensive_validation(self, user_request: str) -> Tuple[bool, str, List[str]]:
        """Perform comprehensive validation of the user request."""
        
        # Step 1: Basic keyword validation
        is_valid, reason, _ = self.validator.validate_request(user_request)
        if not is_valid:
            return False, reason, []
        
        # Step 2: LLM-based feasibility analysis
        is_feasible, feasibility_reason, required_apis = self.api_analyzer.analyze_request_feasibility(
            user_request, self.available_apis
        )
        
        if not is_feasible:
            return False, f"🤖 AI Analysis: {feasibility_reason}", required_apis
        
        # Step 3: Detect and validate required APIs
        detected_apis = self.detect_api_requirements(user_request)
        
        # Step 4: Check if all required API keys are available
        missing_keys = self._check_api_keys(detected_apis)
        if missing_keys:
            missing_keys_str = ', '.join(missing_keys)
            return False, f"🔑 Missing required API keys: {missing_keys_str}. All required APIs must be properly configured before code generation.", detected_apis
        
        return True, "", detected_apis
    
    def create_tool(self, user_request: str) -> Dict:
        """Main method with comprehensive validation before code generation."""
        try:
            print(f"🔧 Processing request: {user_request}")
            print("🔍 Running comprehensive validation...")
            
            # Comprehensive validation
            is_valid, error_reason, detected_apis = self._comprehensive_validation(user_request)
            
            if not is_valid:
                return {
                    "status": "rejected",
                    "message": f"❌ Request rejected: {error_reason}",
                    "detected_apis": detected_apis
                }
            
            print(f"✅ Validation passed. APIs required: {', '.join(detected_apis) if detected_apis else 'None'}")
            
            # Show API limitations before proceeding
            if detected_apis:
                print("\n⚠️  API LIMITATIONS:")
                for api in detected_apis:
                    if api in self.available_apis:
                        print(f"- {api}: {self.available_apis[api]['limitations']}")
                
                # Confirm user wants to proceed
                proceed = input("\nDo you want to proceed with these API limitations? (y/n): ").strip().lower()
                if proceed != 'y':
                    return {
                        "status": "cancelled",
                        "message": "Tool creation cancelled due to API limitations."
                    }
            
            # Generate code only after all validations pass
            generated_code, detected_apis = self.generate_tool_code(user_request, detected_apis)
            
            # Extract function name
            function_name = self.extract_function_name(generated_code)
            
            # Show code to user for final approval
            print("\n" + "="*50)
            print("GENERATED CODE:")
            print("="*50)
            print(generated_code)
            print("="*50)
            print(f"Function Name: {function_name}")
            print(f"APIs Used: {', '.join(detected_apis)}")
            print("="*50)
            
            # Final approval
            approval = input("\nApprove this code? (y/n): ").strip().lower()
            
            if approval == 'y':
                filepath = self.save_tool(generated_code, function_name, user_request, detected_apis)
                
                result = {
                    "status": "success",
                    "function_name": function_name,
                    "filepath": filepath,
                    "apis_used": detected_apis,
                    "message": f"Tool '{function_name}' created successfully!"
                }
                
                print(f"\n✅ {result['message']}")
                print(f"📁 Saved to: {filepath}")
                self._show_usage_reminders()
                
                return result
            else:
                return {
                    "status": "cancelled",
                    "message": "Tool creation cancelled by user."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating tool: {str(e)}"
            }
    
    def generate_tool_code(self, user_request: str, detected_apis: List[str]) -> Tuple[str, List[str]]:
        """Generate code only for validated requests with confirmed APIs."""
        
        # Create enhanced prompt with strict API usage
        api_info = ""
        for api in detected_apis:
            if api in self.available_apis:
                api_data = self.available_apis[api]
                api_info += f"\n- {api}: {api_data['description']}"
                api_info += f"\n  Capabilities: {', '.join(api_data['capabilities'])}"
                api_info += f"\n  Limitations: {api_data['limitations']}"
                api_info += f"\n  Import: {api_data['import_statement']}"
                if api_data['requires_key']:
                    if isinstance(api_data['key_name'], list):
                        keys = ", ".join([f"os.getenv('{key}')" for key in api_data['key_name']])
                        api_info += f"\n  Keys: {keys}"
                    else:
                        api_info += f"\n  Key: os.getenv('{api_data['key_name']}')"
        
        prompt = f"""
You are creating a Python function for: "{user_request}"

STRICT REQUIREMENTS:
1. Use ONLY these verified APIs: {', '.join(detected_apis)}
2. Do NOT use any other APIs or create placeholder implementations
3. Include comprehensive error handling for API failures
4. Add rate limiting considerations for API calls
5. Return structured data with clear success/error status
6. Include appropriate financial disclaimers

CRITICAL ANTI-SIMULATION REQUIREMENTS:
1. NEVER create simulated/fake/placeholder data
2. NEVER use hardcoded values as substitutes for real API data
3. NEVER calculate fake prices like "100.0 + len(news_headlines) * 0.5"
4. NEVER return mock data when real API calls fail
5. If you cannot get real data, return an error message - DO NOT simulate

Available APIs (ONLY use these):
{api_info}

Create a single, complete function that:
- Handles all specified requirements within API limitations
- Includes proper input validation
- Has comprehensive error handling
- Returns meaningful Real results or clear error messages
- Includes rate limiting awareness
- Has detailed docstring with limitations
- NEVER simulates or creates fake data

Do NOT:
- Create placeholder implementations for missing APIs
- Promise capabilities beyond available APIs
- Use external APIs not listed above
- Make unrealistic accuracy claims
- Simulate ANY data - always use real API responses

Format as complete Python code with imports and example usage.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a financial tool code generator. Generate code that strictly follows the requirements and never simulates data."),
                HumanMessage(content=prompt)
            ]
            
            response = self.model.invoke(messages)
            generated_code = response.content
            
            # Clean up the code
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].strip()
            
            # POST-GENERATION VALIDATION: Check for simulation patterns
            simulation_patterns = [
                'simulated_price', 'fake_price', 'mock_price', 'placeholder',
                '100.0 +', '* 0.5', 'mock_data', 'fake_data', 'dummy_data'
            ]
            
            for pattern in simulation_patterns:
                if pattern in generated_code.lower():
                    raise Exception(f"Generated code contains simulation pattern: '{pattern}'. Real API implementation required.")
                    
            return generated_code, detected_apis
            
        except Exception as e:
            raise Exception(f"Error generating code: {str(e)}")
    
    def _check_api_keys(self, apis: List[str]) -> List[str]:
        """Check which required API keys are missing."""
        missing_keys = []
        
        for api in apis:
            if api in self.available_apis and self.available_apis[api]['requires_key']:
                key_name = self.available_apis[api]['key_name']
                
                if isinstance(key_name, list):
                    for key in key_name:
                        if not os.getenv(key):
                            missing_keys.append(key)
                else:
                    if not os.getenv(key_name):
                        missing_keys.append(key_name)
        
        return missing_keys
    
    def _show_usage_reminders(self):
        """Show important usage reminders."""
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("- This tool is for educational purposes only")
        print("- Financial data may have delays or inaccuracies")
        print("- Always verify data from official sources")
        print("- Respect API rate limits to avoid being blocked")
        print("- Past performance does not guarantee future results")
    
    def extract_function_name(self, code: str) -> str:
        """Extract the main function name from generated code."""
        function_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if function_matches:
            return function_matches[0]
        return "unknown_function"
    
    def standardize_api_keys(self, code: str) -> str:
        """Replace API key variable names with standardized ones."""
        for standard_name, actual_name in self.api_key_mapping.items():
            pattern = rf"os\.getenv\(['\"]({standard_name})['\"]"
            replacement = f"os.getenv('{actual_name}'"
            code = re.sub(pattern, replacement, code)
        return code
    
    def save_tool(self, code: str, function_name: str, description: str, apis_used: List[str]) -> str:
        """Save the generated tool with enhanced headers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{function_name}_{timestamp}.py"
        filepath = os.path.join(self.tools_directory, filename)
        
        standardized_code = self.standardize_api_keys(code)
        
        header = f'''"""
Generated Tool: {function_name}
Description: {description}
APIs Used: {', '.join(apis_used)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DISCLAIMER: This tool is for educational and informational purposes only.
- Financial data may be delayed or inaccurate
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Always consult with financial professionals before making investment decisions
- Respect API rate limits and terms of service

API Limitations:
{chr(10).join([f"- {api}: {self.available_apis[api]['limitations']}" for api in apis_used if api in self.available_apis])}
"""

import os
from dotenv import load_dotenv
load_dotenv()

'''
        
        with open(filepath, 'w') as f:
            f.write(header + standardized_code)
        
        self._update_registry(function_name, filename, description, apis_used)
        return filepath
    
    def _update_registry(self, function_name: str, filename: str, description: str, apis_used: List[str]):
        """Update the tool registry."""
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
        
        tool_info = {
            "name": function_name,
            "filename": filename,
            "description": description,
            "apis_used": apis_used,
            "created_at": datetime.now().isoformat(),
            "filepath": os.path.join(self.tools_directory, filename)
        }
        
        registry["tools"].append(tool_info)
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def list_available_apis(self):
        """Display comprehensive API information."""
        print("\n📡 Available APIs:")
        print("-" * 70)
        for api_name, api_info in self.available_apis.items():
            status = "✅" if not api_info['requires_key'] else "🔑"
            
            print(f"{status} {api_name}: {api_info['description']}")
            
            if api_info['requires_key']:
                if isinstance(api_info['key_name'], list):
                    all_keys_available = all(os.getenv(key) for key in api_info['key_name'])
                    key_available = "✅" if all_keys_available else "❌"
                    key_names = ', '.join(api_info['key_name'])
                    print(f"   Keys: {key_names} - {key_available}")
                else:
                    key_available = "✅" if os.getenv(api_info['key_name']) else "❌"
                    print(f"   Key: {api_info['key_name']} - {key_available}")
            
            print(f"   Capabilities: {', '.join(api_info['capabilities'])}")
            print(f"   Cannot do: {', '.join(api_info['cannot_do'])}")
            print(f"   Limitations: {api_info['limitations']}")
            print()

def main():
    """Main function with enhanced user guidance."""
    try:
        agent = CodeGeneratorAgent()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("🤖 Robust Code Generator Agent Started!")
    print("=" * 70)
    print("IMPORTANT DISCLAIMERS:")
    print("- This tool creates educational financial tools only")
    print("- All required API keys must be configured before code generation")
    print("- Free APIs have strict limitations - no premium data access")
    print("- Generated tools are for learning purposes, not actual trading")
    print("=" * 70)
    print("Commands: 'exit' to quit, 'apis' to see available APIs, 'help' for guidelines")
    
    while True:
        user_input = input("\n💬 Describe the tool you want to create: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'apis':
            agent.list_available_apis()
            continue
        elif user_input.lower() == 'help':
            print("\n📖 USAGE GUIDELINES:")
            print("\n✅ REQUESTS THAT WILL BE ACCEPTED:")
            print("- 'Get current stock price for Apple' (uses yfinance)")
            print("- 'Fetch financial news for a company' (uses newsapi)")
            print("- 'Calculate moving averages for a stock' (uses yfinance)")
            print("- 'Send email alert when stock hits price target' (uses yfinance + email)")
            print("- 'Create a tool that gets stock price for a stock and latest news for it and sends summary via email'") #Multiple API Requirements
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Unclear request")
            print("- 'Create a simple portfolio tracker'") #doesn't work
            print("- 'Get information about Apple'") #doesn't work
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Security Violation")
            print("- Build a function that hacks into trading systems")
            print("- Access insider trading information")
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Impossible Request")
            print("- Create a tool that predicts tomorrow's stock prices with 100% accuracy") 
            print("- 'Generate code that creates money printing machine")
            print("- 'Make a tool that generates guaranteed profits'")
            print(" - 'Create a tool that reads my mind and picks stocks'") 
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: free vs paid confusion")
            print("- Create a tool that fetches from Premium data sources (Bloomberg, Reuters, etc.)") 
            print("- 'Fetch real-time options chain data'") 
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: free vs paid confusion: Data heavy")
            print("- 'Create a tool that analyzes 1000 stocks and ranks them by volatility'") 
            print("- 'Build a function that backtests trading strategies over 10 years'") 
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: API key not found")
            print("- 'Create dashboard showing stock price, news, and social media mentions'")
            
            print("\n🔑 API KEY REQUIREMENTS:")
            print("- ALL required API keys must be configured before code generation")
            print("- No placeholder implementations for missing APIs")
            print("- Check 'apis' command to see which keys are missing")
            continue
        elif not user_input:
            continue
        
        result = agent.create_tool(user_input)
        print(f"\n📋 Result: {result['message']}")
        
        if result['status'] == 'rejected' and 'detected_apis' in result:
            detected = result['detected_apis']
            if detected:
                print(f"💡 APIs that would be needed: {', '.join(detected)}")

if __name__ == "__main__":
    main()