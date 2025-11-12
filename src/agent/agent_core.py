"""
Core logic for the agentic AI loop and tool orchestration.
Uses LangChain for agent orchestration and tool use.
"""

from typing import Dict, List, Optional, Any
import pandas as pd

try:
    # LangChain 1.0+ uses LangGraph for agents
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import Tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    Tool = None
    create_react_agent = None

from ..genai.llm_interface import LLMInterface
from ..agent.memory import SessionMemory
from ..data_loader import load_cpg_data, get_data_summary
from ..tools.trend_analysis import extract_trends, detect_seasonality, compare_stores
from ..tools.anomaly_detection import detect_anomalies, detect_sales_drop
from ..tools.scenario_simulation import simulate_promotion, simulate_price_change, simulate_supply_shortage


class CPGDecisionAgent:
    """
    Main agent class that orchestrates tools, memory, and LLM for CPG decision support.
    """
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        memory: Optional[SessionMemory] = None,
        data_path: Optional[str] = None,
        use_spark: Optional[bool] = None
    ):
        """
        Initialize the CPG Decision Agent.
        
        Args:
            llm: LLMInterface instance (if None, creates default)
            memory: SessionMemory instance (if None, creates new)
            data_path: Path to CPG sales data parquet file
            use_spark: Whether to use PySpark (None for auto-detect)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required. Install with: pip install langchain langchain-openai")
        
        self.llm = llm or LLMInterface()
        self.memory = memory or SessionMemory()
        self.data_path = data_path
        self.use_spark = use_spark
        self.data: Optional[pd.DataFrame] = None
        
        # Load data if path provided
        if data_path:
            self.load_data(data_path)
        
        # Initialize tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
    
    def load_data(self, data_path: str):
        """Load CPG sales data."""
        self.data = load_cpg_data(data_path, use_spark=self.use_spark)
        if hasattr(self.data, 'toPandas'):  # PySpark DataFrame
            self.data = self.data.toPandas()
        self.memory.update_context('data_path', data_path)
        self.memory.update_context('data_loaded', True)
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from analysis functions."""
        tools = [
            Tool(
                name="extract_trends",
                func=lambda query: str(self._extract_trends_tool(query)),
                description="Extract trends from sales data. Input: JSON string with date_col, value_col, group_by (optional)."
            ),
            Tool(
                name="detect_seasonality",
                func=lambda query: str(self._detect_seasonality_tool(query)),
                description="Detect seasonality patterns (weekly, monthly, quarterly). Input: JSON string with date_col, value_col, period."
            ),
            Tool(
                name="detect_anomalies",
                func=lambda query: str(self._detect_anomalies_tool(query)),
                description="Detect anomalies in sales data. Input: JSON string with date_col, value_col, method (zscore/iqr/isolation_forest)."
            ),
            Tool(
                name="compare_stores",
                func=lambda query: str(self._compare_stores_tool(query)),
                description="Compare performance across stores. Input: JSON string with date_col, value_col, store_col."
            ),
            Tool(
                name="simulate_promotion",
                func=lambda query: str(self._simulate_promotion_tool(query)),
                description="Simulate promotional campaign impact. Input: JSON string with product_id, store_id, discount_pct, duration_days."
            ),
            Tool(
                name="simulate_price_change",
                func=lambda query: str(self._simulate_price_change_tool(query)),
                description="Simulate price increase impact. Input: JSON string with product_id, store_id, price_increase_pct."
            ),
            Tool(
                name="get_data_summary",
                func=lambda _: str(self._get_data_summary_tool()),
                description="Get summary statistics of the loaded dataset."
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create LangChain agent executor using LangGraph."""
        if not hasattr(self.llm, 'llm') or not self.llm.llm:
            raise ValueError("LLM not properly initialized")
        
        # Check if using Hugging Face (which may not support tool binding)
        use_hf = getattr(self.llm, 'use_huggingface', False)
        
        # Create system prompt
        system_prompt = """You are a smart CPG Decision Support Agent. You help business stakeholders make data-driven decisions.

You have access to tools for:
- Trend analysis and seasonality detection
- Anomaly detection
- Store comparison
- Scenario simulation (promotions, price changes)

When answering questions:
1. Use the appropriate tools to gather data
2. Analyze the results
3. Provide clear, actionable insights
4. Reference specific numbers and trends
5. Suggest next steps or recommendations

Always be concise but thorough. Use natural language to explain technical findings."""
        
        # Check if LLM supports tool binding (OpenAI models do, Hugging Face typically doesn't)
        try:
            # Try to create ReAct agent (works with OpenAI and compatible models)
            agent_graph = create_react_agent(
                self.llm.llm,
                self.tools,
                prompt=system_prompt
            )
        except (AttributeError, TypeError) as e:
            # If tool binding fails (e.g., Hugging Face), create a simple wrapper
            # that manually handles tool calls
            if use_hf:
                # For Hugging Face, we'll use a simpler approach
                # Return a callable that processes queries manually
                class SimpleAgent:
                    def __init__(self, llm, tools, system_prompt):
                        self.llm = llm
                        self.tools = {tool.name: tool for tool in tools}
                        self.system_prompt = system_prompt
                    
                    def invoke(self, input_dict):
                        question = input_dict.get('messages', [])[-1].content if input_dict.get('messages') else str(input_dict)
                        if hasattr(question, 'content'):
                            question = question.content
                        
                        # Create a prompt that includes available tools
                        tools_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
                        prompt = f"""{self.system_prompt}

Available tools:
{tools_desc}

Question: {question}

Think step by step:
1. Which tool(s) should I use to answer this question?
2. Call the tool(s) with appropriate parameters
3. Analyze the results
4. Provide a clear answer

Let me help you answer this question."""
                        
                        # Get LLM response
                        response = self.llm.invoke(prompt)
                        response_text = str(response) if not isinstance(response, str) else response
                        
                        # Try to detect tool calls in the response and execute them
                        # For now, we'll do simple keyword matching
                        tool_results = []
                        for tool_name, tool in self.tools.items():
                            if tool_name.lower() in question.lower() or any(keyword in question.lower() for keyword in tool_name.split('_')):
                                try:
                                    # Try to extract parameters from question or use defaults
                                    tool_input = '{}'  # Default empty JSON
                                    result = tool.func(tool_input)
                                    tool_results.append(f"{tool_name} result: {result}")
                                except Exception as e:
                                    tool_results.append(f"{tool_name} error: {str(e)}")
                        
                        # Combine tool results with LLM response
                        if tool_results:
                            final_response = f"{response_text}\n\nTool Results:\n" + "\n".join(tool_results)
                        else:
                            final_response = response_text
                        
                        # Return in LangGraph format
                        from langchain_core.messages import AIMessage
                        return {"messages": [AIMessage(content=final_response)]}
                
                return SimpleAgent(self.llm.llm, self.tools, system_prompt)
            else:
                raise e
        
        return agent_graph
    
    def _check_data_loaded(self):
        """Check if data is loaded."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first or provide data_path in constructor.")
    
    def _extract_trends_tool(self, query: str) -> Dict:
        """Tool wrapper for trend extraction."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = extract_trends(
            self.data,
            date_col=params.get('date_col', 'date'),
            value_col=params.get('value_col', 'revenue'),
            group_by=params.get('group_by'),
            method=params.get('method', 'linear')
        )
        self.memory.add_tool_call('extract_trends', params, result)
        return result
    
    def _detect_seasonality_tool(self, query: str) -> Dict:
        """Tool wrapper for seasonality detection."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = detect_seasonality(
            self.data,
            date_col=params.get('date_col', 'date'),
            value_col=params.get('value_col', 'revenue'),
            period=params.get('period', 'weekly'),
            group_by=params.get('group_by')
        )
        self.memory.add_tool_call('detect_seasonality', params, result)
        return result
    
    def _detect_anomalies_tool(self, query: str) -> Dict:
        """Tool wrapper for anomaly detection."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = detect_anomalies(
            self.data,
            date_col=params.get('date_col', 'date'),
            value_col=params.get('value_col', 'revenue'),
            method=params.get('method', 'zscore'),
            threshold=params.get('threshold', 3.0),
            group_by=params.get('group_by')
        )
        self.memory.add_tool_call('detect_anomalies', params, result)
        return result
    
    def _compare_stores_tool(self, query: str) -> Dict:
        """Tool wrapper for store comparison."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = compare_stores(
            self.data,
            date_col=params.get('date_col', 'date'),
            value_col=params.get('value_col', 'revenue'),
            store_col=params.get('store_col', 'store_id')
        )
        self.memory.add_tool_call('compare_stores', params, result)
        return result
    
    def _simulate_promotion_tool(self, query: str) -> Dict:
        """Tool wrapper for promotion simulation."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = simulate_promotion(
            self.data,
            product_id=params.get('product_id'),
            store_id=params.get('store_id'),
            discount_pct=params.get('discount_pct', 0.15),
            duration_days=params.get('duration_days', 7)
        )
        self.memory.add_tool_call('simulate_promotion', params, result)
        return result
    
    def _simulate_price_change_tool(self, query: str) -> Dict:
        """Tool wrapper for price change simulation."""
        self._check_data_loaded()
        import json
        params = json.loads(query) if isinstance(query, str) else query
        result = simulate_price_change(
            self.data,
            product_id=params.get('product_id'),
            store_id=params.get('store_id'),
            price_increase_pct=params.get('price_increase_pct', 0.10)
        )
        self.memory.add_tool_call('simulate_price_change', params, result)
        return result
    
    def _get_data_summary_tool(self) -> Dict:
        """Tool wrapper for data summary."""
        self._check_data_loaded()
        result = get_data_summary(self.data)
        self.memory.add_tool_call('get_data_summary', {}, result)
        return result
    
    def run(self, question: str, generate_memo: bool = True) -> Dict[str, Any]:
        """
        Run the agent to answer a business question.
        
        Args:
            question: Business question to answer
            generate_memo: Whether to generate a strategy memo
        
        Returns:
            Dictionary with response, analysis results, and memo
        """
        # Add user question to memory
        self.memory.add_message('user', question)
        
        # Get recent history for context
        recent_history = self.memory.get_recent_history(n=10)
        chat_history = [
            (msg['role'], msg['content']) for msg in recent_history
        ]
        
        try:
            # Run agent (LangGraph agents use invoke directly)
            # Format input for LangGraph - it expects messages
            from langchain_core.messages import HumanMessage
            response = self.agent_executor.invoke({
                "messages": [HumanMessage(content=question)]
            })
            
            # Extract response from LangGraph format
            if isinstance(response, dict):
                if 'messages' in response:
                    # Get last message content
                    messages = response['messages']
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content'):
                            agent_response = last_msg.content
                        elif isinstance(last_msg, dict):
                            agent_response = last_msg.get('content', str(last_msg))
                        else:
                            agent_response = str(last_msg)
                    else:
                        agent_response = str(response)
                else:
                    agent_response = response.get('output', str(response))
            else:
                agent_response = str(response)
            
            # Add agent response to memory
            self.memory.add_message('assistant', agent_response)
            
            # Get tool results from recent calls
            recent_tool_calls = self.memory.tool_calls[-5:]  # Last 5 tool calls
            analysis_results = {
                tc['tool_name']: tc['result'] for tc in recent_tool_calls
            }
            
            # Generate strategy memo if requested
            memo = None
            if generate_memo and analysis_results:
                try:
                    memo = self.llm.generate_strategy_memo(
                        analysis_results,
                        question,
                        context=self.memory.get_conversation_summary()
                    )
                except Exception as e:
                    print(f"Warning: Could not generate memo: {e}")
            
            return {
                'response': agent_response,
                'analysis_results': analysis_results,
                'strategy_memo': memo,
                'tool_calls': [tc['tool_name'] for tc in recent_tool_calls]
            }
        
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.memory.add_message('assistant', error_msg)
            return {
                'response': error_msg,
                'error': str(e),
                'analysis_results': {},
                'strategy_memo': None
            }
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear()
