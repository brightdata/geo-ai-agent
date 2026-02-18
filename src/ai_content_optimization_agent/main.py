#!/usr/bin/env python

import warnings
from ai_content_optimization_agent.crew import AiContentOptimizationAgent
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    # Read URL from the terminal
    url = input("Please enter the URL to process: ").strip()

    if not url:
        raise ValueError("No URL provided. Exiting.")


    # Build the required agent input
    inputs = {
        "url": url,
    }

    try:
        print(f"Analyzing '{url}' for AI content optimization...")

        # Run the multi-agent workflow
        AiContentOptimizationAgent().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
