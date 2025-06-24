"""
File: test_rag.py
Author: Lokesh Reddy Kasumuru
Contributors: 
Date: 03-30-2025

"""

import os
import requests
import json
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Set the OpenAI API key as an environment variable

apikey = os.getenv("OPENAI_API_KEY")

# Load test cases from an external JSON file
with open("test_cases.json", "r") as file:
    test_cases = json.load(file)




# Function to send a question to the chatbot and retrieve its response
def get_chatbot_response(question):
    """
    Sends a question to the chatbot and retrieves the response.
    Args:
        question (str): The question to ask the chatbot.
    Returns:
        str: The chatbot's response or None if an error occurs.
    """
    response = requests.post('http://127.0.0.1:80/ask', json={"message": question})
    if response.status_code == 200:
        return response.json().get("response", "")
    return None

# Function to evaluate a single test case
def evaluate_test_case(test_case):
    """
    Evaluates a chatbot response against a test case using relevancy and faithfulness metrics.
    Args:
        test_case (dict): A dictionary containing the question, expected answer, and retrieval context.
    Returns:
        dict: Evaluation results including relevancy and faithfulness scores.
    """
    actual_output = get_chatbot_response(test_case["question"])
    if not actual_output:
        print(f"Error fetching chatbot response for question: {test_case['question']}")
        return None

    # Create a test case object for evaluation
    llm_test_case = LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        retrieval_context=test_case["retrieval_context"]
    )

    # Evaluate relevancy and faithfulness of the chatbot response
    relevancy_metric = AnswerRelevancyMetric(threshold=0.9, model="gpt-3.5-turbo")
    faithfulness_metric = FaithfulnessMetric(threshold=0.9, model="gpt-3.5-turbo")


    relevancy_metric.measure(llm_test_case)
    faithfulness_metric.measure(llm_test_case)

    # Compile results
    result = {
        "question": test_case["question"],
        "actual_answer": actual_output,
        "expected_answer": test_case["expected_answer"],
        "relevancy_score": relevancy_metric.score,
        "faithfulness_score": faithfulness_metric.score,
        "relevancy_reason": relevancy_metric.reason,
        "faithfulness_reason": faithfulness_metric.reason,
        "retrieval_context": test_case["retrieval_context"]
    }

    # Update the test case with the actual chatbot output
    test_case["actual_answer"] = actual_output
    test_case["relevancy_reason"] = relevancy_metric.reason
    test_case["faithfulness_reason"] = faithfulness_metric.reason
    return result

# Function to evaluate all test cases and save results
def run_tests():
    """
    Runs all test cases, evaluates chatbot performance, and saves results to JSON files.
    """
    results = []
    updated_test_cases = []

    # Evaluate each test case
    for test_case in test_cases:
        result = evaluate_test_case(test_case)
        if result:
            results.append(result)
            updated_test_cases.append(test_case)

    # Save evaluation results to a JSON file
    with open("evaluation_results.json", "w") as results_file:
        json.dump(results, results_file, indent=4)

    # Save updated test cases (with actual outputs) to another JSON file
    with open("updated_test_cases.json", "w") as test_cases_file:
        json.dump(updated_test_cases, test_cases_file, indent=4)

    print("\nEvaluation completed. Results saved to 'evaluation_results.json'.")
    print("Updated test cases saved to 'updated_test_cases.json'.")

# Entry point for the script
if __name__ == "__main__":
    run_tests()