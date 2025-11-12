"""
Command-line interface for the CPG Decision Support Agent.
Provides an interactive CLI for querying the agent.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.agent_core import CPGDecisionAgent
from src.agent.memory import SessionMemory
from src.genai.llm_interface import LLMInterface
from src.data_loader import get_data_summary


def create_agent(data_path: str = None, model: str = "gpt-4", use_azure: bool = False):
    """Create and initialize the agent."""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: API key not found.")
        print("Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize LLM
    llm = LLMInterface(
        model=model,
        use_azure=use_azure
    )
    
    # Initialize memory
    memory = SessionMemory()
    
    # Initialize agent
    if data_path and os.path.exists(data_path):
        agent = CPGDecisionAgent(
            llm=llm,
            memory=memory,
            data_path=data_path
        )
        print(f"✅ Data loaded from {data_path}")
    else:
        agent = CPGDecisionAgent(
            llm=llm,
            memory=memory
        )
        if data_path:
            print(f"⚠️  Warning: Data file not found at {data_path}")
    
    return agent


def interactive_mode(agent: CPGDecisionAgent):
    """Run interactive CLI mode."""
    print("\n" + "="*60)
    print("CPG Decision Support Agent - Interactive Mode")
    print("="*60)
    print("Type 'help' for commands, 'quit' or 'exit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n📖 Available commands:")
                print("  help          - Show this help message")
                print("  summary       - Show data summary")
                print("  memory        - Show memory status")
                print("  clear         - Clear memory")
                print("  quit/exit      - Exit the program")
                print("\n💡 Example questions:")
                print("  - What are the sales trends?")
                print("  - Compare stores")
                print("  - Simulate a 15% promotion")
                continue
            
            if question.lower() == 'summary':
                if agent.data is not None:
                    summary = get_data_summary(agent.data)
                    print("\n📊 Data Summary:")
                    print(f"  Rows: {summary.get('rows', 'N/A'):,}")
                    print(f"  Stores: {summary.get('stores', 'N/A')}")
                    print(f"  SKUs: {summary.get('skus', 'N/A')}")
                    if summary.get('total_revenue'):
                        print(f"  Total Revenue: ${summary['total_revenue']:,.2f}")
                    if summary.get('date_range'):
                        print(f"  Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
                else:
                    print("❌ No data loaded")
                continue
            
            if question.lower() == 'memory':
                memory = agent.memory
                print(f"\n🧠 Memory Status:")
                print(f"  Conversation turns: {len(memory.conversation_history)}")
                print(f"  Tool calls: {len(memory.tool_calls)}")
                print(f"  Cached analyses: {len(memory.analysis_cache)}")
                continue
            
            if question.lower() == 'clear':
                agent.clear_memory()
                print("✅ Memory cleared!")
                continue
            
            # Process question
            print("\n🤔 Analyzing...")
            result = agent.run(question, generate_memo=True)
            
            print("\n" + "="*60)
            print("📋 Response:")
            print("="*60)
            print(result['response'])
            
            if result.get('strategy_memo'):
                print("\n" + "="*60)
                print("📄 Strategy Memo:")
                print("="*60)
                print(result['strategy_memo'])
            
            if result.get('tool_calls'):
                print(f"\n🔧 Tools used: {', '.join(result['tool_calls'])}")
            
            if result.get('error'):
                print(f"\n❌ Error: {result['error']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CPG Decision Support Agent - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with data
  python -m src.ui.cli --data data/cpg_sales_data.parquet
  
  # Single question
  python -m src.ui.cli --data data/cpg_sales_data.parquet --question "What are the sales trends?"
  
  # Use Azure OpenAI
  python -m src.ui.cli --data data/cpg_sales_data.parquet --azure
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/cpg_sales_data.parquet',
        help='Path to CPG sales data parquet file'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4',
        help='LLM model to use (default: gpt-4)'
    )
    
    parser.add_argument(
        '--azure',
        action='store_true',
        help='Use Azure OpenAI instead of OpenAI'
    )
    
    args = parser.parse_args()
    
    # Create agent
    try:
        agent = create_agent(
            data_path=args.data,
            model=args.model,
            use_azure=args.azure
        )
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Run in single question mode or interactive mode
    if args.question:
        print(f"🤔 Question: {args.question}\n")
        result = agent.run(args.question, generate_memo=True)
        
        print("\n" + "="*60)
        print("📋 Response:")
        print("="*60)
        print(result['response'])
        
        if result.get('strategy_memo'):
            print("\n" + "="*60)
            print("📄 Strategy Memo:")
            print("="*60)
            print(result['strategy_memo'])
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
