import asyncio
import logging
from typing import List
from urllib.parse import urljoin

from backend.config.utils import initialize_chatbot
from backend.macro_specialist import MacroSpecialist
from backend.schemas import ChatbotResponse

logger = logging.getLogger(__name__)

async def process_test_queries(
        chatbot: MacroSpecialist,
        queries: List[str],
        base_url: str = "http://localhost:8000",
) -> None:
    """
    Process test queries and display results.

    Args:
        chatbot: Initialized EconomicChatbot instance
        queries: List of test queries to process
        base_url: Base URL for plot URLs (default: http://localhost:8000)
    """
    for idx, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx}: {query}")
        print('='*80)

        try:
            response: ChatbotResponse = await chatbot.process_query(query)

            if response.status == "success":
                # Print Query Info
                if response.query_info:
                    print("\nQuery Information:")
                    print(f"Region: {response.query_info.metadata['region']}")
                    print(f"Economic Concept: {response.query_info.metadata['economic_concept']}")

                # Print Series Info
                if response.series_info:
                    print("\nSeries Information:")
                    print(f"Title: {response.series_info.title}")
                    print(f"Frequency: {response.series_info.frequency}")
                    print(f"Last Updated: {response.series_info.last_updated}")

                # Print Analysis
                if response.analysis:
                    print("\nAnalysis Results:")
                    print(f"Latest Value: {response.analysis.latest_value}")
                    print(f"Trend: {response.analysis.trend}")

                    print("\nKey Observations:")
                    for obs in response.analysis.key_observations:
                        print(f"• {obs}")

                # Handle Visualization
                if response.visualization and response.visualization.plot:
                    plot_url = urljoin(base_url, f"/plots/{response.visualization.plot.filename}")
                    print(f"\nPlot saved and available at: {plot_url}")
                    print(f"Local file path: {response.visualization.plot.path}")

            else:
                print(f"\nError: {response.message}")
                if response.details:
                    print(f"Details: {response.details}")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"\nError processing query: {str(e)}")

async def main() -> None:
    """Main entry point for testing the economic chatbot"""
    try:
        chatbot = await initialize_chatbot()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        return

    # Test queries
    test_queries = [
        # "What is the current GDP of the US?",
        "What is the inflation rate in the Eurozone?",
        # "What is the current interest rate set by the Federal Reserve?",
        # "What is the unemployment rate in Japan?",
        # "What was the GDP growth rate in the US over the past 10 years?",
        # "Show me the US unemployment rate trend for the last 20 years.",
    ]

    try:
        await process_test_queries(chatbot, test_queries)
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())