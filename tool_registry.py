# tool_registry.py - IMPROVED VERSION
import json
import os
import sys
import importlib.util
from typing import List, Dict, Any, Callable
from pathlib import Path
from langchain_core.tools import Tool

class DynamicToolLoader:
    """Enhanced dynamic tool loader with better error handling and validation"""
    
    def __init__(self, registry_path: str = "tool_registry.json", tools_dir: str = "dynamic_tools"):
        self.registry_path = Path(registry_path)
        self.tools_dir = Path(tools_dir)
        self._loaded_modules = {}  # Cache for loaded modules
        
    def load_dynamic_tools(self) -> List[Tool]:
        """Load dynamic tools from tool_registry.json with improved error handling"""
        dynamic_tools = []
        
        try:
            # Ensure tools directory exists
            self.tools_dir.mkdir(exist_ok=True)
            
            # Load tool registry
            if not self.registry_path.exists():
                print(f"⚠️ {self.registry_path} not found, skipping dynamic tools")
                return dynamic_tools
            
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            print(f"📋 Found {len(registry.get('tools', []))} tools in registry")
            
            for tool_info in registry.get('tools', []):
                try:
                    tool = self._load_single_tool(tool_info)
                    if tool:
                        dynamic_tools.append(tool)
                        print(f"✅ Successfully loaded: {tool_info['name']}")
                    else:
                        print(f"⚠️ Failed to load: {tool_info['name']}")
                        
                except Exception as e:
                    print(f"❌ Error loading tool {tool_info.get('name', 'unknown')}: {str(e)}")
                    continue
        
        except FileNotFoundError:
            print(f"❌ Registry file not found: {self.registry_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in registry: {str(e)}")
        except Exception as e:
            print(f"❌ Error loading tool registry: {str(e)}")
        
        print(f"📊 Loaded {len(dynamic_tools)} dynamic tools successfully")
        return dynamic_tools
    
    def _load_single_tool(self, tool_info: Dict[str, Any]) -> Tool:
        """Load a single tool with comprehensive validation"""
        
        # Validate required fields
        required_fields = ['name', 'filepath', 'description']
        for field in required_fields:
            if field not in tool_info:
                raise ValueError(f"Missing required field: {field}")
        
        tool_name = tool_info['name']
        tool_path = Path(tool_info['filepath'])
        
        # Check if file exists
        if not tool_path.exists():
            # Try relative to tools directory
            alt_path = self.tools_dir / tool_path.name
            if alt_path.exists():
                tool_path = alt_path
            else:
                raise FileNotFoundError(f"Tool file not found at tool path: {tool_path}")
        
        # Load or get cached module
        module_key = str(tool_path.absolute())
        print("\n Module Key = ", module_key)
        if module_key in self._loaded_modules:
            module = self._loaded_modules[module_key]
        else:
            module = self._load_module(tool_name, tool_path)
            self._loaded_modules[module_key] = module
        
        # Find the main function
        main_func = self._find_main_function(module, tool_name)
        print("\n Main function name = ",main_func)
        print("\n Main function name already present = ",tool_name )
        if not main_func:
            raise ValueError(f"No callable function found in {tool_name}")
        
        # Validate function is callable
        if not callable(main_func):
            raise ValueError(f"Main function in {tool_name} is not callable")
        
        # Create enhanced tool wrapper
        wrapped_func = self._create_tool_wrapper(main_func, tool_name)
        
        # Create Tool instance
        tool = Tool(
            name=tool_name,
            func=wrapped_func,
            description=tool_info['description']
        )
        print("💡💡\nTool Creation :",tool)
        return tool
    
    def _load_module(self, tool_name: str, tool_path: Path):
        """Load Python module from file path"""
        try:
            # Create unique module name to avoid conflicts
            module_name = f"dynamic_tool_{tool_name}_{hash(str(tool_path))}"
            
            # Load module spec
            spec = importlib.util.spec_from_file_location(module_name, tool_path)
            if spec is None:
                raise ImportError(f"Could not load module specification for {tool_path}")
            
            # Create module
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to make imports work
            sys.modules[module_name] = module
            
            # Execute module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            raise ImportError(f"Failed to load module {tool_name}: {str(e)}")
    
    def _find_main_function(self, module, tool_name: str) -> Callable:
        """Find the main callable function in the module"""
        
        # Priority order for function names
        possible_names = [
            tool_name,           # Exact tool name
            'main',              # Standard main function
            'execute',           # Common execution function
            'run',               # Common run function
            f"{tool_name}_main", # Tool name + main
            'handler',           # Handler function
            'process'            # Process function
        ]
        
        # Try each possible name
        for func_name in possible_names:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    return func
        
        # If no named function found, look for any callable
        for attr_name in dir(module):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr = getattr(module, attr_name)
                if callable(attr) and not isinstance(attr, type):  # Skip classes
                    print(f"⚠️ Using fallback function '{attr_name}' for tool {tool_name}")
                    return attr
        
        return None
    
    def _create_tool_wrapper(self, original_func: Callable, tool_name: str) -> Callable:
        """Create a wrapper function with error handling and logging"""
        
        def wrapped_tool_function(*args, **kwargs):
            try:
                print(f"🔧 Executing dynamic tool: {tool_name}")
                print(f"   Args: {args}")
                print(f"   Kwargs: {kwargs}")
                
                # Call the original function
                result = original_func(*args, **kwargs)
                
                print(f"✅ Tool {tool_name} completed successfully")
                return result
                
            except Exception as e:
                error_msg = f"❌ Error in dynamic tool '{tool_name}': {str(e)}"
                print(error_msg)
                return error_msg
        
        # Preserve function metadata
        wrapped_tool_function.__name__ = f"dynamic_{tool_name}"
        wrapped_tool_function.__doc__ = getattr(original_func, '__doc__', f"Dynamic tool: {tool_name}")
        
        return wrapped_tool_function
    
    def validate_tool_file(self, tool_path: Path) -> Dict[str, Any]:
        """Validate a tool file and return information about it"""
        validation_result = {
            'valid': False,
            'functions': [],
            'errors': [],
            'main_function': None
        }
        
        try:
            # Try to load the module
            temp_name = f"validation_{hash(str(tool_path))}"
            spec = importlib.util.spec_from_file_location(temp_name, tool_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all callable functions
            functions = []
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    attr = getattr(module, attr_name)
                    if callable(attr) and not isinstance(attr, type):
                        functions.append(attr_name)
            
            validation_result['functions'] = functions
            validation_result['main_function'] = self._find_main_function(module, tool_path.stem)
            validation_result['valid'] = validation_result['main_function'] is not None
            
        except Exception as e:
            validation_result['errors'].append(str(e))
        
        return validation_result

# Global instance
tool_loader = DynamicToolLoader()

def load_dynamic_tools() -> List[Tool]:
    """Convenience function for backward compatibility"""
    return tool_loader.load_dynamic_tools()

# Example usage and testing functions
def test_dynamic_tool_loading():
    """Test the dynamic tool loading system"""
    print("🧪 Testing Dynamic Tool Loading System")
    print("=" * 50)
    
    # Load tools
    tools = load_dynamic_tools()
    
    print(f"\n📊 Results:")
    print(f"   Total tools loaded: {len(tools)}")
    
    for tool in tools:
        print(f"   ✅ {tool.name}: {tool.description}")
        
        # Test tool execution with dummy data
        try:
            # This is just a test - adjust based on your actual tool requirements
            result = tool.func("test_input")
            print(f"      Test result: {result}")
        except Exception as e:
            print(f"      Test failed: {str(e)}")
    
    return tools

def validate_all_tools():
    """Validate all tools in the registry"""
    print("🔍 Validating All Tools")
    print("=" * 30)
    
    loader = DynamicToolLoader()
    
    try:
        with open(loader.registry_path, 'r') as f:
            registry = json.load(f)
        
        for tool_info in registry.get('tools', []):
            tool_path = Path(tool_info['filepath'])
            print(f"\n📋 Validating: {tool_info['name']}")
            
            validation = loader.validate_tool_file(tool_path)
            
            if validation['valid']:
                print(f"   ✅ Valid - Main function: {validation['main_function'].__name__}")
                print(f"   📝 Available functions: {validation['functions']}")
            else:
                print(f"   ❌ Invalid")
                for error in validation['errors']:
                    print(f"      Error: {error}")
                print(f"   📝 Found functions: {validation['functions']}")
    
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")

if __name__ == "__main__":
    # Run tests
    test_dynamic_tool_loading()
    print("\n" + "="*50 + "\n")
    validate_all_tools()