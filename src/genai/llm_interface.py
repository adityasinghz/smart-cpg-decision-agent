"""
Wrapper for GenAI calls (OpenAI / Azure OpenAI / Hugging Face).
Provides a unified interface for LLM interactions.
"""

import os
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    ChatOpenAI = None

try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEndpoint = None
    ChatHuggingFace = None
    HuggingFacePipeline = None


class LLMInterface:
    """
    Interface for LLM interactions.
    Supports OpenAI, Azure OpenAI, and Hugging Face models.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        use_azure: bool = False,
        use_huggingface: bool = False,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        huggingface_model: Optional[str] = None,
        use_local_model: bool = False
    ):
        """
        Initialize LLM interface.
        
        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'meta-llama/Llama-2-7b-chat-hf')
            temperature: Sampling temperature
            use_azure: Whether to use Azure OpenAI
            use_huggingface: Whether to use Hugging Face (instead of OpenAI)
            api_key: API key (if None, reads from environment)
            azure_endpoint: Azure endpoint URL
            api_version: Azure API version
            huggingface_model: Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            use_local_model: Whether to load model locally (requires GPU/CPU resources)
        """
        self.model = model
        self.temperature = temperature
        self.use_azure = use_azure
        self.use_huggingface = use_huggingface
        self.use_local_model = use_local_model
        
        # Hugging Face setup
        if use_huggingface:
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError(
                    "Hugging Face packages required. Install with: "
                    "pip install langchain-huggingface transformers torch"
                )
            
            hf_model = huggingface_model or model or os.getenv('HUGGINGFACE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
            
            if use_local_model:
                # Load model locally (requires GPU recommended)
                print(f"Loading Hugging Face model locally: {hf_model}")
                print("This may take a few minutes and requires significant RAM/GPU...")
                
                tokenizer = AutoTokenizer.from_pretrained(hf_model)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    hf_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model_obj,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=temperature,
                    return_full_text=False
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                self.client = None
            else:
                # Use Hugging Face Inference API (free tier available)
                hf_token = api_key or os.getenv('HUGGINGFACE_API_TOKEN')
                if not hf_token:
                    print("⚠️  Warning: No Hugging Face token provided. Using public API (may have rate limits).")
                    print("   Get free token at: https://huggingface.co/settings/tokens")
                
                # Use Hugging Face Inference Client directly for better control
                # This works with both conversational and text-generation models
                from huggingface_hub import InferenceClient
                
                client = InferenceClient(
                    model=hf_model,
                    token=hf_token
                )
                
                # Create a simple wrapper that uses the client
                class HFLLMWrapper:
                    def __init__(self, client, model, temperature):
                        self.client = client
                        self.model = model
                        self.temperature = temperature
                        self.task = None  # Will be determined on first call
                    
                    def invoke(self, prompt):
                        # Try conversational first (for Mistral, etc.)
                        try:
                            # Format as messages for conversational API
                            if isinstance(prompt, str):
                                messages = [{"role": "user", "content": prompt}]
                            else:
                                messages = prompt
                            
                            response = self.client.chat_completion(
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=512
                            )
                            # Extract content from response
                            if isinstance(response, dict):
                                return response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                            return str(response)
                        except Exception:
                            # Fallback to text generation
                            if isinstance(prompt, str):
                                text_prompt = prompt
                            else:
                                # Extract text from messages
                                text_prompt = str(prompt)
                            
                            response = self.client.text_generation(
                                prompt=text_prompt,
                                temperature=self.temperature,
                                max_new_tokens=512
                            )
                            return response
                
                self.llm = HFLLMWrapper(client, hf_model, temperature)
                self.client = None
            
            self.deployment_name = None
            return
        
        # OpenAI/Azure setup
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI and langchain packages required. Install with: pip install openai langchain-openai")
        
        # Get API key
        if api_key is None:
            if use_azure:
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
            else:
                api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key is None:
            raise ValueError("API key required. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
        
        # Initialize client
        if use_azure:
            if azure_endpoint is None:
                azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            if api_version is None:
                api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=f"{azure_endpoint}/openai/deployments/{model}",
                api_version=api_version
            )
            self.deployment_name = model
        else:
            self.client = OpenAI(api_key=api_key)
            self.deployment_name = None
        
        # Initialize LangChain LLM
        if use_azure:
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=f"{azure_endpoint}/openai/deployments/{model}",
                openai_api_version=api_version
            )
        else:
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if self.use_huggingface:
            # Hugging Face models - use the wrapper's invoke method
            # Format as conversational prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
            
            response = self.llm.invoke(full_prompt)
            # Handle different response formats
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        else:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm(messages, max_tokens=max_tokens)
            return response.content
    
    def generate_strategy_memo(
        self,
        analysis_results: Dict,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a strategy memo based on analysis results.
        
        Args:
            analysis_results: Dictionary with analysis results
            question: Original business question
            context: Additional context
        
        Returns:
            Strategy memo text
        """
        system_prompt = """You are a strategic business analyst for a CPG (Consumer Packaged Goods) company.
Generate clear, actionable strategy memos based on data analysis results.
Focus on:
1. Key insights and findings
2. Business implications
3. Recommended actions
4. Risks and considerations
Use professional but accessible language."""
        
        prompt = f"""Based on the following analysis results, generate a strategy memo to answer this business question:

Question: {question}

Analysis Results:
{self._format_analysis_results(analysis_results)}

{('Additional Context: ' + context) if context else ''}

Generate a comprehensive strategy memo with clear recommendations."""
        
        return self.generate(prompt, system_prompt=system_prompt, max_tokens=2000)
    
    def explain_trends(
        self,
        trends: Dict,
        context: Optional[str] = None
    ) -> str:
        """
        Generate natural language explanation of trends.
        
        Args:
            trends: Trend analysis results
            context: Additional context
        
        Returns:
            Natural language explanation
        """
        system_prompt = """You are a data analyst explaining sales trends to business stakeholders.
Provide clear, concise explanations of trends and their business implications."""
        
        prompt = f"""Explain the following sales trends in business terms:

{self._format_analysis_results(trends)}

{('Context: ' + context) if context else ''}

Provide a clear explanation of what these trends mean for the business."""
        
        return self.generate(prompt, system_prompt=system_prompt)
    
    def explain_anomalies(
        self,
        anomalies: Dict,
        context: Optional[str] = None
    ) -> str:
        """
        Generate natural language explanation of anomalies.
        
        Args:
            anomalies: Anomaly detection results
            context: Additional context
        
        Returns:
            Natural language explanation
        """
        system_prompt = """You are a data analyst explaining anomalies to business stakeholders.
Help identify potential causes and recommend actions."""
        
        prompt = f"""Explain the following sales anomalies:

{self._format_analysis_results(anomalies)}

{('Context: ' + context) if context else ''}

Explain what these anomalies might indicate and what actions should be considered."""
        
        return self.generate(prompt, system_prompt=system_prompt)
    
    def _format_analysis_results(self, results: Dict) -> str:
        """Format analysis results as a readable string."""
        import json
        return json.dumps(results, indent=2, default=str)
