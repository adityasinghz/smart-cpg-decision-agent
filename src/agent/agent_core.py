"""
Agent Layer - Core Agent Implementation
========================================
Implements the agentic AI loop for CPG decision support.
Orchestrates tools, memory, and LLM to answer business questions intelligently.

Author: CPG Analytics Team
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json
import re
import pandas as pd

from ..genai.llm_interface import LLMInterface
from ..agent.memory import SessionMemory
from ..data_loader import load_cpg_data, get_data_summary
from ..tools.trend_analysis import extract_trends, detect_seasonality, compare_stores
from ..tools.anomaly_detection import detect_anomalies, detect_sales_drop, get_anomaly_summary
from ..tools.scenario_simulation import simulate_promotion, simulate_price_change, simulate_supply_shortage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPGDecisionAgent:
    """
    Intelligent agent for CPG decision support.
    
    Features:
        - Natural language query understanding with schema awareness
        - Automatic tool selection and orchestration
        - Context-aware responses using memory
        - Multi-step reasoning for complex queries
        - Business memo generation
        - Direct data Q&A with validation
    """
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        memory: Optional[SessionMemory] = None,
        data_path: Optional[str] = None,
        use_spark: Optional[bool] = None
    ):
        """
        Initialize the CPG agent.
        
        Args:
            llm: LLMInterface instance (if None, creates default)
            memory: SessionMemory instance (if None, creates new)
            data_path: Path to CPG sales data parquet file
            use_spark: Whether to use PySpark (None for auto-detect)
        """
        logger.info("Initializing CPG Decision Support Agent...")
        
        # Initialize LLM
        self.llm = llm or LLMInterface()
        
        # Initialize memory
        self.memory = memory or SessionMemory()
        
        # Data loading
        self.data_path = data_path
        self.use_spark = use_spark
        self.data: Optional[pd.DataFrame] = None
        
        # Load data if path provided
        if data_path:
            self.load_data(data_path)
        
        # System prompt for the agent
        self.system_prompt = self._build_system_prompt()
        
        # Available tools registry
        self.tools = {
            'load_data': self._tool_load_data,
            'trend_analysis': self._tool_trend_analysis,
            'anomaly_detection': self._tool_anomaly_detection,
            'scenario_simulation': self._tool_scenario_simulation,
            'get_summary': self._tool_get_summary,
            'data_qa': self._tool_data_qa
        }
        
        logger.info("✓ Agent initialized successfully")
    
    def load_data(self, data_path: str):
        """Load CPG sales data."""
        logger.info(f"Loading data from {data_path}")
        self.data = load_cpg_data(data_path, use_spark=self.use_spark)
        if hasattr(self.data, 'toPandas'):  # PySpark DataFrame
            self.data = self.data.toPandas()
        
        # Store dataset profile in memory
        self._profile_dataset()
        
        self.memory.update_context('data_path', data_path)
        self.memory.update_context('data_loaded', True)
        logger.info(f"✓ Data loaded: {len(self.data)} rows")
    
    def _profile_dataset(self):
        """Profile the dataset and store facts in memory."""
        if self.data is None:
            return
        
        # Store schema
        schema_info = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        self.memory.update_context('schema', schema_info)
        
        # Store categorical distinct values
        categorical_cols = ['category', 'store_region', 'promo_type', 'store_size', 'store_id', 'sku_id']
        distinct_values = {}
        for col in categorical_cols:
            if col in self.data.columns:
                distinct_values[col] = sorted(self.data[col].dropna().unique().tolist())
        self.memory.update_context('distinct_values', distinct_values)
        
        # Store numeric ranges
        numeric_ranges = {}
        for col in ['revenue', 'units_sold', 'price']:
            if col in self.data.columns:
                numeric_ranges[col] = {
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'mean': float(self.data[col].mean())
                }
        self.memory.update_context('numeric_ranges', numeric_ranges)
        
        # Store summary stats
        if 'date' in self.data.columns:
            self.memory.update_context('date_range', (
                str(self.data['date'].min()),
                str(self.data['date'].max())
            ))
        if 'revenue' in self.data.columns:
            self.memory.update_context('total_revenue', float(self.data['revenue'].sum()))
        if 'store_id' in self.data.columns:
            self.memory.update_context('num_stores', int(self.data['store_id'].nunique()))
        if 'sku_id' in self.data.columns:
            self.memory.update_context('num_skus', int(self.data['sku_id'].nunique()))
    
    def chat(self, user_query: str) -> str:
        """
        Main chat interface for the agent.
        
        Args:
            user_query: User's natural language query.
            
        Returns:
            Agent's response as a string.
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Add user message to memory
        self.memory.add_message('user', user_query)
        
        try:
            # Step 1: Understand the query and determine actions
            plan = self._plan_actions(user_query)
            
            # Step 2: Execute planned actions
            results = self._execute_plan(plan)
            
            # Step 3: Generate response using LLM
            response = self._generate_response(user_query, plan, results)
            
            # Add assistant response to memory
            self.memory.add_message('assistant', response, metadata={'plan': plan, 'results': results})
            
            return response
            
        except Exception as e:
            error_msg = f"I encountered an error processing your request: {str(e)}"
            logger.error(f"Error in chat: {e}", exc_info=True)
            self.memory.add_message('assistant', error_msg, metadata={'error': str(e)})
            return error_msg
    
    def _plan_actions(self, query: str) -> Dict[str, Any]:
        """
        Determine what actions to take based on the query with improved classification.
        
        Args:
            query: User query.
            
        Returns:
            Plan dictionary with actions to execute.
        """
        query_lower = query.lower()
        
        plan = {
            'actions': [],
            'query_type': 'general',
            'requires_data': True,
            'needs_visualization': False
        }
        
        # Ensure data is loaded
        if self.data is None:
            plan['actions'].append({'tool': 'load_data', 'params': {}})
        
        # ========== Priority 1: Data Q&A (must be first to avoid false trend/anomaly matches) ==========
        if self._is_data_qa(query_lower):
            plan['query_type'] = 'data_qa'
            plan['actions'].append({
                'tool': 'data_qa',
                'params': self._extract_data_qa_params(query)
            })
            logger.info("Classified as: DATA_QA")
            return plan
        
        # ========== Priority 2: Scenario Simulation ==========
        if self._is_scenario_query(query_lower):
            plan['query_type'] = 'scenario'
            scenario_type, params = self._extract_scenario_params(query)
            plan['actions'].append({
                'tool': 'scenario_simulation',
                'params': {'scenario_type': scenario_type, **params}
            })
            logger.info(f"Classified as: SCENARIO ({scenario_type})")
            return plan
        
        # ========== Priority 3: Anomaly Detection ==========
        if self._is_anomaly_query(query_lower):
            plan['query_type'] = 'anomaly'
            plan['actions'].append({
                'tool': 'anomaly_detection',
                'params': {'metric': self._extract_metric(query)}
            })
            logger.info("Classified as: ANOMALY")
            return plan
        
        # ========== Priority 4: Trend Analysis ==========
        if self._is_trend_query(query_lower):
            plan['query_type'] = 'trend'
            plan['actions'].append({
                'tool': 'trend_analysis',
                'params': {'metric': self._extract_metric(query)}
            })
            logger.info("Classified as: TREND")
            return plan
        
        # ========== Priority 5: Summary/Overview ==========
        if any(word in query_lower for word in ['summary', 'overview', 'report', 'status', 'performance', 'dashboard']):
            plan['query_type'] = 'summary'
            plan['actions'].append({
                'tool': 'get_summary',
                'params': {}
            })
            logger.info("Classified as: SUMMARY")
            return plan
        
        # ========== Default: General query → Trend ==========
        plan['query_type'] = 'general'
        plan['actions'].append({
            'tool': 'trend_analysis',
            'params': {'metric': 'revenue'}
        })
        logger.info("Classified as: GENERAL (default to trend)")
        return plan
    
    # ========== Intent Detection Methods ==========
    
    def _is_data_qa(self, query_lower: str) -> bool:
        """Detect if query is a direct data question."""
        strong_keywords = [
            'how many', 'how much', 'total', 'count of', 'number of',
            'date range', 'time span', 'time period', 'dataset', 'data set',
            'show me', 'list', 'display', 'what are the', 'which',
            'top', 'bottom', 'highest', 'lowest', 'best', 'worst',
            'average', 'mean', 'median', 'maximum', 'minimum',
            'sum of', 'unique', 'distinct'
        ]
        
        exclude_keywords = ['trend', 'growing', 'pattern', 'seasonality', 'anomaly', 'what if', 'simulate']
        
        has_strong = any(kw in query_lower for kw in strong_keywords)
        has_exclude = any(kw in query_lower for kw in exclude_keywords)
        
        return has_strong and not has_exclude
    
    def _is_scenario_query(self, query_lower: str) -> bool:
        """Detect if query is a scenario simulation."""
        scenario_indicators = [
            'what if', 'suppose', 'simulate', 'scenario', 'imagine',
            'promotion', 'promo', 'discount', 'price change', 'price increase', 'price decrease',
            'stockout', 'out of stock', 'inventory shortage', 'demand change', 'demand increase'
        ]
        return any(ind in query_lower for ind in scenario_indicators)
    
    def _is_anomaly_query(self, query_lower: str) -> bool:
        """Detect if query is about anomalies."""
        anomaly_keywords = [
            'anomaly', 'anomalies', 'outlier', 'outliers', 'unusual', 'strange',
            'spike', 'spikes', 'drop', 'drops', 'abnormal', 'irregular',
            'data quality', 'issues', 'problems', 'errors'
        ]
        return any(kw in query_lower for kw in anomaly_keywords) and \
               'what if' not in query_lower and \
               not self._is_data_qa(query_lower)
    
    def _is_trend_query(self, query_lower: str) -> bool:
        """Detect if query is about trends."""
        trend_keywords = [
            'trend', 'trending', 'growth', 'growing', 'increasing', 'decreasing',
            'pattern', 'patterns', 'seasonality', 'seasonal', 'cyclical',
            'over time', 'time series', 'progression', 'trajectory'
        ]
        return any(kw in query_lower for kw in trend_keywords) and \
               'what if' not in query_lower and \
               not self._is_data_qa(query_lower)
    
    def _extract_scenario_params(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Extract scenario type and parameters."""
        query_lower = query.lower()
        
        # Promotion/Discount
        if any(word in query_lower for word in ['promotion', 'promo', 'discount']):
            return 'promotion', self._extract_promotion_params(query)
        
        # Price Change
        elif any(word in query_lower for word in ['price change', 'price increase', 'price decrease', 'pricing']):
            return 'price_change', self._extract_price_params(query)
        
        # Stockout
        elif any(word in query_lower for word in ['stockout', 'out of stock', 'inventory shortage', 'shortage']):
            return 'stockout', self._extract_stockout_params(query)
        
        # Default to promotion
        return 'promotion', {}
    
    def _extract_data_qa_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters for data QA."""
        query_lower = query.lower()
        params = {
            'original_query': query,
            'operation': 'describe',
            'metric': None,
            'filters': {},
            'limit': 10
        }
        
        # Detect operation type
        if any(w in query_lower for w in ['top', 'highest', 'best', 'largest']):
            params['operation'] = 'top_n'
        elif any(w in query_lower for w in ['bottom', 'lowest', 'worst', 'smallest']):
            params['operation'] = 'bottom_n'
        elif any(w in query_lower for w in ['total', 'sum']):
            params['operation'] = 'aggregate'
        elif any(w in query_lower for w in ['average', 'avg', 'mean']):
            params['operation'] = 'average'
        elif any(w in query_lower for w in ['count', 'how many', 'number of']):
            params['operation'] = 'count'
        elif any(w in query_lower for w in ['date range', 'time span', 'time period']):
            params['operation'] = 'date_range'
        elif any(w in query_lower for w in ['show', 'list', 'display']):
            params['operation'] = 'sample'
        
        # Extract metric
        params['metric'] = self._extract_metric(query)
        
        # Extract limit
        limit_match = re.search(r'(?:top|bottom|first|last)\s+(\d+)', query_lower)
        if limit_match:
            params['limit'] = int(limit_match.group(1))
        
        # Extract grouping
        if 'categor' in query_lower:
            params['group_by'] = 'category'
        elif 'region' in query_lower:
            params['group_by'] = 'store_region'
        elif 'sku' in query_lower:
            params['group_by'] = 'sku_id'
        elif 'store' in query_lower and 'region' not in query_lower:
            params['group_by'] = 'store_id'
        elif 'promo' in query_lower:
            params['group_by'] = 'promo_type'
        
        return params
    
    def _tool_data_qa(self, **params) -> Dict[str, Any]:
        """Answer direct data questions with validation and grounding."""
        if self.data is None:
            self._tool_load_data()
        
        operation = params.get('operation', 'describe')
        metric = params.get('metric', 'revenue')
        limit = params.get('limit', 10)
        group_by = params.get('group_by')
        query_text = params.get('original_query', '').lower()
        
        result = {
            'operation': operation,
            'query': params.get('original_query', ''),
            'data': {}
        }
        
        # Validate group_by column
        if group_by and group_by not in self.data.columns:
            distinct_vals = self.memory.get_context('distinct_values', {})
            result['data'] = {
                'error': f"Column '{group_by}' does not exist in the dataset.",
                'available_columns': list(self.data.columns),
                'suggestion': f"Try grouping by: {', '.join(['category', 'store_region', 'sku_id', 'store_id', 'promo_type'])}"
            }
            return result
        
        # Extract and validate filter values
        filtered_data = self.data.copy()
        filter_col = None
        filter_value = None
        
        # Try to extract specific filter values from query
        distinct_vals = self.memory.get_context('distinct_values', {})
        
        if 'region' in query_text:
            filter_col = 'store_region'
            distinct_regions = distinct_vals.get('store_region', [])
            for region in distinct_regions:
                if str(region).lower() in query_text:
                    filter_value = region
                    break
        
        elif 'category' in query_text or 'categor' in query_text:
            filter_col = 'category'
            distinct_cats = distinct_vals.get('category', [])
            for cat in distinct_cats:
                if str(cat).lower() in query_text:
                    filter_value = cat
                    break
        
        elif 'promo' in query_text:
            filter_col = 'promo_type'
            distinct_promos = distinct_vals.get('promo_type', [])
            for promo in distinct_promos:
                if str(promo).lower() in query_text:
                    filter_value = promo
                    break
        
        # Apply filter if found
        if filter_col and filter_value:
            filtered_data = filtered_data[filtered_data[filter_col] == filter_value]
            if len(filtered_data) == 0:
                available = distinct_vals.get(filter_col, [])
                result['data'] = {
                    'error': f"No data found for {filter_col}='{filter_value}'",
                    'available_values': available,
                    'suggestion': f"Available {filter_col}: {', '.join(map(str, available))}"
                }
                return result
        
        # Execute Query
        try:
            if operation == 'date_range':
                result['data'] = {
                    'start_date': str(filtered_data['date'].min()),
                    'end_date': str(filtered_data['date'].max()),
                    'total_days': (filtered_data['date'].max() - filtered_data['date'].min()).days,
                    'total_rows': len(filtered_data)
                }
            
            elif operation == 'aggregate':
                if group_by:
                    agg_result = filtered_data.groupby(group_by)[metric].sum().sort_values(ascending=False).head(limit)
                    result['data'] = {
                        'metric': metric,
                        'operation': 'sum',
                        'group_by': group_by,
                        'results': {str(k): float(v) for k, v in agg_result.items()}
                    }
                else:
                    result['data'] = {
                        'metric': metric,
                        'total': float(filtered_data[metric].sum())
                    }
            
            elif operation == 'average':
                if group_by:
                    avg_result = filtered_data.groupby(group_by)[metric].mean().sort_values(ascending=False).head(limit)
                    result['data'] = {
                        'metric': metric,
                        'operation': 'average',
                        'group_by': group_by,
                        'results': {str(k): float(v) for k, v in avg_result.items()}
                    }
                else:
                    result['data'] = {
                        'metric': metric,
                        'average': float(filtered_data[metric].mean())
                    }
            
            elif operation == 'count':
                if group_by:
                    count_result = filtered_data[group_by].value_counts().head(limit)
                    result['data'] = {
                        'group_by': group_by,
                        'results': {str(k): int(v) for k, v in count_result.items()}
                    }
                else:
                    result['data'] = {
                        'total_rows': len(filtered_data),
                        'unique_stores': int(filtered_data['store_id'].nunique()) if 'store_id' in filtered_data.columns else 0,
                        'unique_skus': int(filtered_data['sku_id'].nunique()) if 'sku_id' in filtered_data.columns else 0,
                        'unique_categories': int(filtered_data['category'].nunique()) if 'category' in filtered_data.columns else 0
                    }
            
            elif operation == 'top_n':
                if group_by:
                    top_result = filtered_data.groupby(group_by)[metric].sum().nlargest(limit)
                    result['data'] = {
                        'metric': metric,
                        'operation': 'top',
                        'limit': limit,
                        'group_by': group_by,
                        'results': {str(k): float(v) for k, v in top_result.items()}
                    }
            
            elif operation == 'bottom_n':
                if group_by:
                    bottom_result = filtered_data.groupby(group_by)[metric].sum().nsmallest(limit)
                    result['data'] = {
                        'metric': metric,
                        'operation': 'bottom',
                        'limit': limit,
                        'group_by': group_by,
                        'results': {str(k): float(v) for k, v in bottom_result.items()}
                    }
            
            elif operation == 'sample':
                sample_cols = ['date', 'category', 'store_region', 'revenue', 'units_sold']
                available_cols = [c for c in sample_cols if c in filtered_data.columns]
                sample_data = filtered_data[available_cols].head(limit)
                result['data'] = {
                    'rows': sample_data.to_dict('records')
                }
            
            else:  # describe
                result['data'] = {
                    'total_rows': len(filtered_data),
                    'columns': list(filtered_data.columns),
                    'distinct_values': distinct_vals,
                    'numeric_ranges': self.memory.get_context('numeric_ranges', {})
                }
            
        except Exception as e:
            logger.error(f"Error in data_qa: {e}")
            result['data'] = {'error': str(e)}
        
        return result
    
    # ========== Tool Execution ==========
    
    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned actions."""
        results = {}
        
        for action in plan['actions']:
            tool_name = action['tool']
            params = action['params']
            
            logger.info(f"Executing tool: {tool_name}")
            
            try:
                tool_func = self.tools.get(tool_name)
                if tool_func:
                    result = tool_func(**params)
                    results[tool_name] = result
                    self.memory.add_tool_call(tool_name, params, result)
                else:
                    logger.warning(f"Unknown tool: {tool_name}")
                    results[tool_name] = {'error': f'Unknown tool: {tool_name}'}
                    
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                results[tool_name] = {'error': str(e)}
        
        return results
    
    def _generate_response(
        self,
        query: str,
        plan: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Generate natural language response with dataset grounding."""
        # Add dataset profile to context
        schema = self.memory.get_context('schema', {})
        distinct_vals = self.memory.get_context('distinct_values', {})
        
        dataset_info = "\nDATASET SCHEMA:\n"
        dataset_info += f"Columns: {', '.join(schema.keys())}\n\n"
        if distinct_vals:
            dataset_info += "Available categorical values:\n"
            for col, vals in distinct_vals.items():
                dataset_info += f"  • {col}: {', '.join(map(str, vals))}\n"
        
        # Format results for LLM - extract key insights
        results_text = self._format_results_for_llm(results, plan['query_type'])
        
        # Build prompt based on query type
        if plan['query_type'] == 'data_qa':
            prompt = f"""
User Query: {query}

{dataset_info}

Data Query Results:
{results_text}

CRITICAL INSTRUCTIONS:
1. Answer using ONLY the exact data provided above
2. If results contain an error about missing values, explain clearly what's NOT in the dataset
3. Always suggest the actual available values from "Available categorical values"
4. NEVER invent column names, regions, categories, or other values
5. Be specific with exact numbers from results
6. If unsure, say "The dataset does not contain..." and list what IS available

Answer the user's question clearly and concisely. Do NOT mention tool calls or technical details.
"""
        elif plan['query_type'] == 'anomaly':
            prompt = f"""
User Query: {query}

{dataset_info}

Anomaly Detection Results:
{results_text}

INSTRUCTIONS:
1. Start with a direct, concise answer (e.g., "There are X anomalies...")
2. Provide specific numbers: count, percentage, high/low breakdown
3. Identify patterns: regions, categories, dates, promotions associated with anomalies
4. Explain business context: what these anomalies mean (promotions, data quality issues, etc.)
5. Provide actionable recommendations (3-4 bullet points)
6. Be professional and business-focused
7. Do NOT mention tool calls, methods, or technical implementation details
8. Keep response concise (3-4 paragraphs maximum)

Generate a clear, business-focused response.
"""
        elif plan['query_type'] == 'trend':
            prompt = f"""
User Query: {query}

{dataset_info}

Trend Analysis Results:
{results_text}

INSTRUCTIONS:
1. Start with a direct answer about the trend direction
2. Provide specific numbers: growth rate, percentage change, R², significance
3. Mention seasonality if detected
4. Explain business implications
5. Provide actionable recommendations
6. Do NOT mention tool calls or technical details
7. Keep response concise (3-4 paragraphs)

Generate a clear, business-focused response.
"""
        elif plan['query_type'] == 'scenario':
            prompt = f"""
User Query: {query}

{dataset_info}

Scenario Simulation Results:
{results_text}

INSTRUCTIONS:
1. Start with a direct answer about the scenario impact
2. Provide specific numbers: revenue change, units change, percentages
3. Compare baseline vs simulated results
4. Explain business implications
5. Provide clear recommendation (proceed or not, with rationale)
6. Do NOT mention tool calls or technical details
7. Keep response concise (3-4 paragraphs)

Generate a clear, business-focused response.
"""
        else:
            prompt = f"""
User Query: {query}

{dataset_info}

Analysis Results:
{results_text}

INSTRUCTIONS:
1. Provide a clear, direct answer to the user's question
2. Use specific numbers from the results
3. Highlight key insights
4. Provide actionable recommendations when appropriate
5. Do NOT mention tool calls, methods, or technical implementation details
6. Keep response concise and business-focused (3-4 paragraphs)

Generate a clear, business-focused response.
"""
        
        # Generate response - adjust temperature based on query type
        if plan['query_type'] == 'data_qa':
            original_temp = self.llm.temperature
            self.llm.temperature = 0.5
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=600
        )
        
        # Restore original temperature
        if plan['query_type'] == 'data_qa':
            self.llm.temperature = original_temp
        
        return response
    
    # ========== Tool Implementations ==========
    
    def _tool_load_data(self) -> Dict[str, Any]:
        """Load CPG sales data and profile it completely."""
        try:
            if self.data_path:
                self.data = load_cpg_data(self.data_path, use_spark=self.use_spark)
                if hasattr(self.data, 'toPandas'):
                    self.data = self.data.toPandas()
                
                self._profile_dataset()
                
                return {
                    'status': 'success',
                    'rows': len(self.data),
                    'columns': list(self.data.columns)
                }
            else:
                return {'status': 'error', 'message': 'No data path provided'}
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _tool_trend_analysis(self, metric: str = 'revenue') -> Dict[str, Any]:
        """Perform trend analysis."""
        if self.data is None:
            self._tool_load_data()
        
        if self.data is None:
            return {'error': 'Data not loaded'}
        
        try:
            result = extract_trends(
                self.data,
                date_col='date',
                value_col=metric,
                method='linear',
                period='daily'  # Default to daily, matching reference
            )
            
            # Also get seasonality
            seasonality_result = detect_seasonality(
                self.data,
                date_col='date',
                value_col=metric,
                period='monthly'
            )
            
            combined_result = {
                'linear_trend': result,
                'seasonality': seasonality_result
            }
            
            self.memory.add_tool_call('trend_analysis', {'metric': metric}, combined_result)
            return combined_result
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'error': str(e)}
    
    def _tool_anomaly_detection(self, metric: str = 'revenue') -> Dict[str, Any]:
        """Perform anomaly detection matching reference implementation."""
        if self.data is None:
            self._tool_load_data()
        
        if self.data is None:
            return {'error': 'Data not loaded'}
        
        try:
            # Use get_anomaly_summary to match reference implementation format
            result = get_anomaly_summary(
                self.data,
                metric=metric,
                include_multivariate=True
            )
            
            # Store insight about anomaly rate
            anomaly_rate = result.get('overall_assessment', {}).get('anomaly_rate', 0)
            data_quality = result.get('overall_assessment', {}).get('data_quality', 'unknown')
            self.memory.add_context('anomaly_rate', anomaly_rate)
            self.memory.add_context('data_quality', data_quality)
            
            self.memory.add_tool_call('anomaly_detection', {'metric': metric}, result)
            return result
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'error': str(e)}
    
    def _tool_scenario_simulation(
        self,
        scenario_type: str = 'promotion',
        **kwargs
    ) -> Dict[str, Any]:
        """Perform scenario simulation."""
        if self.data is None:
            self._tool_load_data()
        
        if self.data is None:
            return {'error': 'Data not loaded'}
        
        try:
            if scenario_type == 'promotion':
                discount_pct = kwargs.get('discount_pct', 20)
                duration = kwargs.get('duration_days', 7)
                result = simulate_promotion(
                    self.data,
                    discount_pct=discount_pct,
                    duration_days=duration
                )
            elif scenario_type == 'price_change':
                price_change = kwargs.get('price_change_pct', 10.0)
                result = simulate_price_change(
                    self.data,
                    price_change_pct=price_change  # Pass as percentage matching reference
                )
            elif scenario_type == 'stockout':
                probability = kwargs.get('stockout_probability', 0.1)
                result = simulate_supply_shortage(
                    self.data,
                    shortage_probability=probability
                )
            else:
                result = {'error': f'Unknown scenario type: {scenario_type}'}
            
            self.memory.add_tool_call('scenario_simulation', {'scenario_type': scenario_type, **kwargs}, result)
            return result
        except Exception as e:
            logger.error(f"Error in scenario simulation: {e}")
            return {'error': str(e)}
    
    def _tool_get_summary(self) -> Dict[str, Any]:
        """Get overall data summary."""
        if self.data is None:
            self._tool_load_data()
        
        if self.data is None:
            return {'error': 'Data not loaded'}
        
        try:
            summary = get_data_summary(self.data)
            self.memory.add_tool_call('get_summary', {}, summary)
            return summary
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}
    
    # ========== Helper Methods ==========
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent."""
        return (
            "You are an expert CPG (Consumer Packaged Goods) business analyst and strategic advisor. "
            "You help business users understand their sales data, identify trends, detect issues, "
            "and evaluate strategic scenarios. Your responses are:\n"
            "- Clear and concise, avoiding jargon\n"
            "- Data-driven with specific numbers and insights\n"
            "- Action-oriented with practical recommendations\n"
            "- Professional and business-focused\n\n"
            "CRITICAL RULES:\n"
            "- When answering data questions, use ONLY the exact numbers and values provided in the results\n"
            "- If a user asks about data that doesn't exist (e.g., 'North region' when only East/West/South exist), "
            "explicitly say it's NOT in the dataset and list what IS available\n"
            "- Always ground answers in the provided dataset schema and distinct values\n"
            "- NEVER invent column names, regions, categories, SKUs, or other data values\n"
            "- If unsure or data is missing, say 'The dataset does not contain...' and suggest alternatives\n"
        )
    
    def _format_results_for_llm(self, results: Dict[str, Any], query_type: str = 'general') -> str:
        """Format tool results for LLM consumption - extract key insights."""
        formatted = []
        
        for tool_name, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                formatted.append(f"{tool_name}: Error - {result['error']}")
                continue
            
            # Format based on tool type
            if tool_name == 'anomaly_detection':
                formatted.append(self._format_anomaly_results(result))
            elif tool_name == 'trend_analysis':
                formatted.append(self._format_trend_results(result))
            elif tool_name == 'scenario_simulation':
                formatted.append(self._format_scenario_results(result))
            elif tool_name == 'data_qa':
                formatted.append(self._format_data_qa_results(result))
            else:
                # Generic formatting
                try:
                    serializable_result = self._make_json_serializable(result)
                    formatted.append(json.dumps(serializable_result, indent=2)[:2000])
                except Exception as e:
                    formatted.append(str(result)[:2000])
        
        return "\n".join(formatted)
    
    def _format_anomaly_results(self, result: Dict[str, Any]) -> str:
        """Format anomaly detection results for concise LLM consumption with business context."""
        if not isinstance(result, dict):
            return str(result)
        
        formatted = []
        
        # Handle new format from get_anomaly_summary (matching reference)
        if 'overall_assessment' in result:
            # New format from get_anomaly_summary
            overall = result.get('overall_assessment', {})
            stat_outliers = result.get('statistical_outliers', {})
            ts_anomalies = result.get('time_series_anomalies', {})
            
            total_count = overall.get('total_anomalies', 0)
            anomaly_rate = overall.get('anomaly_rate', 0)
            data_quality = overall.get('data_quality', 'unknown')
            
            formatted.append(f"ANOMALY SUMMARY:")
            formatted.append(f"- Total anomalies: {total_count}")
            formatted.append(f"- Anomaly rate: {anomaly_rate:.2f}%")
            formatted.append(f"- Data quality: {data_quality}")
            
            # Statistical outliers
            stat_count = stat_outliers.get('count', 0)
            high_outliers = stat_outliers.get('high_outliers', 0)
            low_outliers = stat_outliers.get('low_outliers', 0)
            formatted.append(f"\nSTATISTICAL OUTLIERS (IQR):")
            formatted.append(f"- Count: {stat_count}")
            formatted.append(f"- High outliers: {high_outliers}")
            formatted.append(f"- Low outliers: {low_outliers}")
            
            # Time series anomalies
            ts_count = ts_anomalies.get('count', 0)
            spikes = ts_anomalies.get('spikes', 0)
            drops = ts_anomalies.get('drops', 0)
            formatted.append(f"\nTIME SERIES ANOMALIES (Z-score):")
            formatted.append(f"- Count: {ts_count}")
            formatted.append(f"- Spikes: {spikes}")
            formatted.append(f"- Drops: {drops}")
            
            # Examples
            stat_examples = stat_outliers.get('examples', [])
            if stat_examples:
                formatted.append(f"\nSAMPLE STATISTICAL OUTLIERS:")
                for i, ex in enumerate(stat_examples[:3], 1):
                    date_str = str(ex.get('date', ''))
                    value = ex.get('value', 0)
                    formatted.append(f"  {i}. Date: {date_str}, Value: ${value:.2f}")
            
            return "\n".join(formatted)
        
        # Old format (backward compatibility)
        anomalies = result.get('anomalies', [])
        count = result.get('count', len(anomalies) if isinstance(anomalies, list) else 0)
        method = result.get('method', 'unknown')
        anomaly_rate = result.get('anomaly_rate', 0)
        
        # Calculate high/low outliers and extract business context
        if isinstance(anomalies, list) and len(anomalies) > 0:
            # Get mean value for comparison
            if self.data is not None and 'revenue' in self.data.columns:
                mean_value = float(self.data['revenue'].mean())
            else:
                mean_value = None
            
            # Analyze anomalies and extract patterns
            high_outliers = 0
            low_outliers = 0
            top_anomalies = []
            
            # Collect patterns from anomalies
            promo_types = {}
            regions = {}
            categories = {}
            max_value = 0
            max_value_date = None
            
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    value = anomaly.get('value', 0)
                    date = anomaly.get('date', '')
                    index = anomaly.get('index', None)
                    
                    if mean_value:
                        if value > mean_value:
                            high_outliers += 1
                        else:
                            low_outliers += 1
                    
                    if value > max_value:
                        max_value = value
                        max_value_date = date
                    
                    # Get business context for this anomaly
                    if self.data is not None and index is not None and index < len(self.data):
                        try:
                            row = self.data.iloc[index]
                            
                            # Track promo types
                            if 'promo_type' in row and pd.notna(row['promo_type']):
                                promo = str(row['promo_type'])
                                promo_types[promo] = promo_types.get(promo, 0) + 1
                            
                            # Track regions
                            if 'store_region' in row:
                                region = str(row['store_region'])
                                regions[region] = regions.get(region, 0) + 1
                            
                            # Track categories
                            if 'category' in row:
                                cat = str(row['category'])
                                categories[cat] = categories.get(cat, 0) + 1
                        except:
                            pass
                    
                    if len(top_anomalies) < 5:
                        top_anomalies.append({
                            'date': str(date),
                            'value': float(value),
                            'index': index
                        })
            
            formatted.append(f"ANOMALY SUMMARY:")
            formatted.append(f"- Total anomalies: {count}")
            formatted.append(f"- Anomaly rate: {anomaly_rate*100:.2f}%")
            formatted.append(f"- High outliers: {high_outliers}")
            formatted.append(f"- Low outliers: {low_outliers}")
            
            if mean_value:
                formatted.append(f"- Mean value: ${mean_value:.2f}")
            
            if max_value > 0:
                formatted.append(f"- Highest anomaly: ${max_value:.2f} on {max_value_date}")
            
            # Add pattern analysis
            if promo_types:
                top_promo = max(promo_types.items(), key=lambda x: x[1])
                formatted.append(f"\nPATTERNS:")
                formatted.append(f"- Most common promo type: {top_promo[0]} ({top_promo[1]} occurrences)")
            
            if regions:
                top_region = max(regions.items(), key=lambda x: x[1])
                formatted.append(f"- Most affected region: {top_region[0]} ({top_region[1]} occurrences)")
            
            if categories:
                top_cat = max(categories.items(), key=lambda x: x[1])
                formatted.append(f"- Most affected category: {top_cat[0]} ({top_cat[1]} occurrences)")
            
            # Add sample anomalies with context
            if top_anomalies and self.data is not None:
                formatted.append(f"\nSAMPLE ANOMALIES:")
                for i, anomaly in enumerate(top_anomalies[:3], 1):
                    date_str = str(anomaly['date']).split()[0] if ' ' in str(anomaly['date']) else str(anomaly['date'])
                    value = anomaly['value']
                    index = anomaly.get('index')
                    
                    context_parts = []
                    if index is not None and index < len(self.data):
                        try:
                            row = self.data.iloc[index]
                            if 'promo_type' in row and pd.notna(row['promo_type']) and str(row['promo_type']) != 'None':
                                context_parts.append(f"Promo: {row['promo_type']}")
                            if 'store_region' in row:
                                context_parts.append(f"Region: {row['store_region']}")
                            if 'category' in row:
                                context_parts.append(f"Category: {row['category']}")
                            if 'holiday_flag' in row and pd.notna(row['holiday_flag']) and row['holiday_flag']:
                                context_parts.append("Holiday")
                        except:
                            pass
                    
                    formatted.append(f"  {i}. Date: {date_str}, Value: ${value:.2f}")
                    if context_parts:
                        formatted.append(f"     Context: {', '.join(context_parts)}")
        else:
            formatted.append(f"ANOMALY SUMMARY:")
            formatted.append(f"- Count: {count}")
            formatted.append(f"- Method: {method}")
            formatted.append(f"- Anomaly rate: {anomaly_rate*100:.2f}%")
        
        return "\n".join(formatted)
    
    def _format_trend_results(self, result: Dict[str, Any]) -> str:
        """Format trend analysis results."""
        if not isinstance(result, dict):
            return str(result)
        
        formatted = []
        
        linear_trend = result.get('linear_trend', {})
        seasonality = result.get('seasonality', {})
        
        formatted.append("TREND ANALYSIS:")
        if linear_trend:
            trend_dir = linear_trend.get('trend', 'unknown')
            slope = linear_trend.get('slope', 0)
            r2 = linear_trend.get('r2', 0)
            is_sig = linear_trend.get('is_significant', False)
            pct_change = linear_trend.get('percentage_change', 0)
            
            formatted.append(f"- Trend direction: {trend_dir}")
            formatted.append(f"- Slope: {slope:.6f}")
            formatted.append(f"- R²: {r2:.4f}")
            formatted.append(f"- Significant: {is_sig}")
            formatted.append(f"- Percentage change: {pct_change:.2f}%")
        
        if seasonality:
            has_season = seasonality.get('has_seasonality', False)
            strength = seasonality.get('strength', 'none')
            formatted.append(f"\nSEASONALITY:")
            formatted.append(f"- Has seasonality: {has_season}")
            formatted.append(f"- Strength: {strength}")
        
        return "\n".join(formatted)
    
    def _format_scenario_results(self, result: Dict[str, Any]) -> str:
        """Format scenario simulation results."""
        if not isinstance(result, dict):
            return str(result)
        
        formatted = []
        
        baseline = result.get('baseline', {})
        simulated = result.get('simulated', {})
        impact = result.get('impact', {})
        
        formatted.append("SCENARIO SIMULATION:")
        formatted.append(f"\nBASELINE:")
        if baseline:
            formatted.append(f"- Revenue: ${baseline.get('revenue', 0):,.2f}")
            formatted.append(f"- Units: {baseline.get('units_sold', 0):,.0f}")
        
        formatted.append(f"\nSIMULATED:")
        if simulated:
            formatted.append(f"- Revenue: ${simulated.get('revenue', 0):,.2f}")
            formatted.append(f"- Units: {simulated.get('units_sold', 0):,.0f}")
        
        formatted.append(f"\nIMPACT:")
        if impact:
            formatted.append(f"- Revenue change: {impact.get('revenue_change_pct', 0):+.1f}%")
            formatted.append(f"- Units change: {impact.get('units_change_pct', 0):+.1f}%")
        
        return "\n".join(formatted)
    
    def _format_data_qa_results(self, result: Dict[str, Any]) -> str:
        """Format data QA results."""
        if not isinstance(result, dict):
            return str(result)
        
        formatted = []
        data = result.get('data', {})
        operation = result.get('operation', 'unknown')
        
        formatted.append(f"DATA QUERY ({operation.upper()}):")
        
        if 'error' in data:
            formatted.append(f"Error: {data['error']}")
            if 'available_values' in data:
                formatted.append(f"Available: {', '.join(map(str, data['available_values']))}")
        else:
            # Format based on operation
            if operation == 'top_n' or operation == 'bottom_n':
                results_dict = data.get('results', {})
                formatted.append(f"Results ({data.get('group_by', 'N/A')}):")
                for key, value in list(results_dict.items())[:10]:
                    formatted.append(f"  {key}: {value:,.2f}")
            elif operation == 'aggregate':
                if 'results' in data:
                    results_dict = data.get('results', {})
                    formatted.append(f"Results ({data.get('group_by', 'N/A')}):")
                    for key, value in list(results_dict.items())[:10]:
                        formatted.append(f"  {key}: {value:,.2f}")
                else:
                    formatted.append(f"Total: {data.get('total', 0):,.2f}")
            elif operation == 'count':
                if 'results' in data:
                    results_dict = data.get('results', {})
                    formatted.append(f"Counts ({data.get('group_by', 'N/A')}):")
                    for key, value in list(results_dict.items())[:10]:
                        formatted.append(f"  {key}: {value}")
                else:
                    formatted.append(f"Total rows: {data.get('total_rows', 0)}")
            else:
                formatted.append(json.dumps(data, indent=2)[:1000])
        
        return "\n".join(formatted)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        import numpy as np
        from datetime import datetime, date

        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _extract_metric(self, query: str) -> str:
        """Extract metric name from query."""
        query_lower = query.lower()
        
        if 'revenue' in query_lower or 'sales' in query_lower:
            return 'revenue'
        elif 'units' in query_lower or 'volume' in query_lower or 'quantity' in query_lower:
            return 'units_sold'
        elif 'price' in query_lower or 'pricing' in query_lower:
            return 'price'
        else:
            return 'revenue'
    
    def _extract_promotion_params(self, query: str) -> Dict[str, Any]:
        """Extract promotion parameters from query."""
        params = {}
        discount_match = re.search(r'(\d+)%?\s*(?:off|discount)', query.lower())
        if discount_match:
            params['discount_pct'] = float(discount_match.group(1))
        duration_match = re.search(r'(\d+)\s*(?:day|week)', query.lower())
        if duration_match:
            days = int(duration_match.group(1))
            if 'week' in query.lower():
                days *= 7
            params['duration_days'] = days
        return params
    
    def _extract_price_params(self, query: str) -> Dict[str, Any]:
        """Extract price change parameters from query."""
        params = {}
        price_match = re.search(r'([+-]?\d+)%?\s*(?:price|pricing)', query.lower())
        if price_match:
            params['price_change_pct'] = float(price_match.group(1))
        elif 'increase' in query.lower():
            params['price_change_pct'] = 10.0
        elif 'decrease' in query.lower() or 'reduce' in query.lower():
            params['price_change_pct'] = -10.0
        return params
    
    def _extract_stockout_params(self, query: str) -> Dict[str, Any]:
        """Extract stockout parameters from query."""
        params = {}
        prob_match = re.search(r'(\d+)%?\s*(?:stockout|shortage)', query.lower())
        if prob_match:
            params['stockout_probability'] = float(prob_match.group(1)) / 100
        return params
    
    def run(self, question: str, generate_memo: bool = False) -> Dict[str, Any]:
        """
        Run the agent to answer a business question (backward compatibility).
        
        Args:
            question: Business question to answer
            generate_memo: Whether to generate a strategy memo (not used in this implementation)
        
        Returns:
            Dictionary with response and analysis results
        """
        response = self.chat(question)
        
        # Get recent tool calls
        recent_tool_calls = self.memory.tool_calls[-5:] if self.memory.tool_calls else []
        analysis_results = {
            tc['tool_name']: tc['result'] for tc in recent_tool_calls
        } if recent_tool_calls else {}
        
        return {
            'response': response,
            'analysis_results': analysis_results,
            'strategy_memo': None,
            'tool_calls': [tc['tool_name'] for tc in recent_tool_calls] if recent_tool_calls else []
        }
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear()
