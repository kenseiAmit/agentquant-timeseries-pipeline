# Standard & Semantic Kernel Runtime Imports
# Complete the imports for all the necessary components from the semantic_kernel library.
import os
import re
import asyncio
import json
import logging
import traceback
import pandas as pd
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.functions import KernelArguments

# -----------------
# Logging Setup
# -----------------
# The logging setup below captures all agent interactions and saves them to 'logs/agent_chat.log'.
# 1. Create a dedicated logger for agent interactions.
agent_logger = logging.getLogger("semantic_kernel.agents")
agent_logger.setLevel(logging.DEBUG)

# 2. Prevent agent logs from propagating to other handlers (like console).
agent_logger.propagate = False

# 3. Create a file handler to write to 'agent_chat.log' in write mode.
os.makedirs("logs", exist_ok=True)
agent_chat_handler = logging.FileHandler("logs/agent_chat.log", mode='w')
agent_chat_handler.setLevel(logging.DEBUG)

# 4. Create a minimal formatter to log only the message content.
chat_formatter = logging.Formatter('%(asctime)s - %(name)s:%(message)s')
agent_chat_handler.setFormatter(chat_formatter)

# 5. Add the dedicated file handler to the agent logger.
agent_logger.addHandler(agent_chat_handler)

# 6. Function to log agent messages
def log_agent_message(content):
    try:
        agent_logger.info(f"Agent: {content.role} - {content.name or '*'}: {content.content}")
    except Exception:
        agent_logger.exception("Failed to write agent message to log")

# -----------------
# Environment Setup
# -----------------
# Environment & Azure OpenAI Configuration
# Load the API key and endpoint URL from the .env file.
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_KEY")
BASE_URL = os.getenv("URL")
API_VERSION = "2024-05-01-preview"


# -----------------
# Kernel and Chat Service
# -----------------
# Semantic Kernel & Chat Service Initialization
# Initialize the Kernel, define the AzureChatCompletion service, and add it to the kernel.
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name="none",
    api_key=API_KEY,
    base_url=BASE_URL,
    api_version=API_VERSION,
)
kernel.add_service(chat_service)


# -----------------
# Helper Functions
# -----------------
# Data Ingestion, Parsing & Self-Healing Code Execution Helpers

def load_quality_instructions(file_path):
    """
    Loads instructional text from a file within the 'specs' directory.

    This function constructs the full path to the file, reads its content,
    and processes it into a list of non-empty, stripped lines.

    Args:
        file_path (str): The name of the file in the 'specs' directory.

    Returns:
        list[str]: A list of strings, where each string is a line of instruction.
                   Returns an empty list if the file does not exist.
    """
    full_path = os.path.join("specs", file_path)
    if not os.path.exists(full_path):
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_reports_instructions(file_path):
    """
    Loads report generation instructions from a file within the 'specs' directory.

    Args:
        file_path (str): The name of the file in the 'specs' directory.

    Returns:
        list[str]: A list of strings for building the report. Returns an
                   empty list if the file does not exist.
    """
    full_path = os.path.join("specs", file_path)
    if not os.path.exists(full_path):
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_logs(file_path):
    """
    Loads agent interaction logs from a file within the 'logs' directory.

    Args:
        file_path (str): The name of the log file in the 'logs' directory.

    Returns:
        list[str]: A list of log entries. Returns an empty list if the file
                   does not exist.
    """
    full_path = os.path.join("logs", file_path)
    if not os.path.exists(full_path):
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

def get_csv_name():
    """
    Interactively prompts the user to select a CSV file from the 'data' directory.

    It lists all available .csv files and asks for a numerical selection.

    Returns:
        str: The relative path to the selected CSV file (e.g., 'data/my_file.csv').
    """
    data_dir = "data"
    csv_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the '{data_dir}' directory.")

    print("\nAvailable CSV files:")
    for idx, name in enumerate(csv_files, start=1):
        print(f"  {idx}. {name}")

    while True:
        choice = input(f"Select a data file (1-{len(csv_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            return os.path.join(data_dir, csv_files[int(choice) - 1])
        print("Invalid selection, please try again.")

def load_csv_file(file_path):
    """
    Reads a CSV file and converts its entire content into a single string.

    The CSV data is flattened into a list and then joined by ', '.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        str: A single string containing all the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    values = df.to_numpy().flatten().tolist()
    return ", ".join(str(v) for v in values)

def build_labeled_series(file_path):
    """
    Builds a list of {label, value} points from a CSV file, preserving any
    date/time/index column as the label so it can carry through cleaning,
    statistics, and visualization.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        list[dict]: A list of {'label': str, 'value': float} dictionaries,
                    in the original row order.
    """
    df = pd.read_csv(file_path)
    if df.shape[1] >= 2:
        labels = df.iloc[:, 0].astype(str).tolist()
        values = df.iloc[:, 1].tolist()
    else:
        labels = [str(i) for i in range(len(df))]
        values = df.iloc[:, 0].tolist()
    return [{"label": label, "value": value} for label, value in zip(labels, values)]

def strip_code_fences(code):
    """Removes surrounding markdown code fences (```python ... ```) if present."""
    text = code.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

def extract_cleaned_data(text):
    """Parses the 'Cleaned Data (JSON)' block of a DataCleaning agent response."""
    match = re.search(r"```(?:json)?\s*(\[[^\]]*\])\s*```", text, re.DOTALL)
    candidate = match.group(1) if match else None

    if candidate is None:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]

    if not candidate:
        return []

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    cleaned = []
    for item in data:
        if isinstance(item, dict) and "value" in item:
            cleaned.append({"label": str(item.get("label", "")), "value": item["value"]})
    return cleaned

class PythonExecutor:
    """
    A safe executor for dynamically generated Python code strings.

    This class is designed to run code provided by an AI agent in a controlled
    manner. It includes a retry mechanism and captures execution errors.
    """
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def run(self, code):
        """
        Executes a string of Python code using the exec() function.

        Args:
            code (str): The Python code to execute.

        Returns:
            tuple[bool, str | None]: A tuple containing:
                - A boolean indicating if the execution was successful.
                - The error traceback as a string if an exception occurred,
                  otherwise None.
        """
        cleaned_code = strip_code_fences(code)
        try:
            exec(cleaned_code, {"__name__": "__main__"})
            return True, None
        except Exception:
            return False, traceback.format_exc()

