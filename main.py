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

def save_final_report(report, path='artifacts/final_report.md'):
    """
    Saves the generated final report to a markdown file.

    Args:
        report (str): The content of the report to be saved.
        path (str, optional): The file path for the saved report.
                              Defaults to 'artifacts/final_report.md'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


# -----------------
# Agent Instructions
# -----------------
# Multi-Agent Specifications & GroupChat Topologies
# 1. Complete the AGENT_CONFIG with detailed prompts for each agent.
data_quality_instructions = '\n'.join(load_quality_instructions("Data_Quality_Instructions.txt"))
report_instructions = '\n'.join(load_reports_instructions("Report_Instructions.txt"))

AGENT_CONFIG = {
    "PythonExecutorAgent": '''You generate runnable Python code that visualizes a dataset.
You will be given two JSON arrays of {"label": ..., "value": ...} objects: the ORIGINAL dataset and the CLEANED dataset. The "label" may be a date, timestamp, or row index.
Requirements:
- Use matplotlib to plot the ORIGINAL data (blue line, legend label "Original Data") and the CLEANED data (green line, legend label "Clean Data") on a single line chart, using each item's "label" for the x-axis and "value" for the y-axis.
- Title the chart "Original vs Clean Data", label the x-axis "Date" and the y-axis "Value", rotate the x-axis tick labels 45 degrees, and include a legend.
- Make sure the 'artifacts' directory exists (create it with os.makedirs(..., exist_ok=True) if needed).
- Save the resulting plot to 'artifacts/data_visualization.png'.
- If a previous attempt failed, you will be given the error traceback - fix the code so it runs without errors.
Output ONLY the raw, runnable Python code. Do not include explanations, comments, or markdown code fences.''',

    "DataCleaning": '''You are a Data Cleaning Assistant.
You will be given a dataset as a JSON array of objects: [{"label": ..., "value": ...}, ...]. The "label" is a date, timestamp, or index identifying each point, and "value" is the numeric reading.
Your tasks, in order:
1. Examine each data point and detect outliers (e.g. zeros that are likely missing/erroneous readings, and statistically extreme high or low values).
2. Present a short cleaning plan: list each detected outlier's label and value in a table, and explain why it was flagged.
3. Remove the identified outliers to produce the cleaned dataset, and present it as a table of the remaining labels and values.
4. If you previously received "Failed" feedback from the Data Validation Auditor, correct your cleaning based on that feedback.
Your response MUST end with a section in EXACTLY this format:
Cleaned Data (JSON):
```json
[{"label": "...", "value": ...}, ...]
```
Include ONLY the remaining (non-outlier) items in that JSON array, preserving their original label and value.''',

    "DataStatistics": '''You are a Data Statistics Assistant.
Using ONLY the "value" fields of the cleaned dataset JSON produced by the Data Cleaning Assistant earlier in this conversation (never the original raw data),
compute: count, mean, median, standard deviation, minimum, and maximum.
Output ONLY the statistical description as a short labeled list (e.g. "Mean: 533.19"), with no commentary, plan, or explanation.''',

    "AnalysisChecker": f'''You are a Data Validation Auditor.
Follow these rules exactly:
{data_quality_instructions}
Review the conversation: the original dataset, the Data Cleaning Assistant's plan and cleaned dataset JSON, and the Data Statistics Assistant's descriptive statistics.
Verify that:
1. The cleaned dataset contains no items previously identified as outliers.
2. The statistics were computed using the cleaned dataset only.
Respond in JSON with fields: "title" ("Approved" or "Failed"), "original_data", "cleaned_data", "removed_data", "statistics".
If both checks pass, the title MUST be exactly "Approved". If either check fails, the title MUST be exactly "Failed" and you must clearly explain the issue so the Data Cleaning Assistant can correct it.''',

    "ReportGenerator": f'''You are a Report Generator.
Using the full agent interaction log provided (data cleaning, statistics, validation, and visualization steps),
write a complete, well-structured markdown report.
Follow EXACTLY this structure and fill in every section using real details found in the log:
{report_instructions}
Formatting guidance:
- For "Data Date", if the data labels are dates, use the first and last date as a range (e.g. "2025-09-01 to 2025-09-20"). Otherwise use "N/A".
- In "Data Cleaning", present the detected outliers and the cleaned data as markdown tables of label/value pairs.
- In "Descriptive Statistics", present the statistics as a markdown table (Statistic, Value) followed by a short bullet-point summary.
- In "Validation Summary", list each validation iteration and its outcome.
- In "Data Visualization", reference the image with: ![Data Visualization](data_visualization.png)
- In "Agent Workflow Summary", include one row per agent: DataCleaning, DataStatistics, AnalysisChecker, PythonExecutorAgent, with their action and status/result.
Output ONLY the final markdown report text.''',

    "ReportChecker": f'''You are a Report Validation Auditor.
Validate the markdown report produced by the Report Generator against this required structure and rules:
{report_instructions}
Check that every section is present and filled in with information consistent with the agent interaction log.
If the report is complete and accurate, respond with the word "Approved" followed by the final report text.
If it is missing information or inaccurate, respond with "Failed" followed by specific, actionable feedback for the Report Generator.'''
}


# -----------------
# Agent Factory
# -----------------
# Multi-Agent Specifications & GroupChat Topologies
# 2. Implement the agent factory function.
def create_agent(name, instructions, service, settings=None):
    """Factory function to create a new ChatCompletionAgent."""
    arguments = KernelArguments(settings=settings) if settings is not None else None
    return ChatCompletionAgent(
        name=name,
        instructions=instructions,
        service=service,
        arguments=arguments,
    )


# -----------------
# Termination Strategy
# -----------------
# A custom termination strategy that stops after user approval.
class ApprovalTerminationStrategy(TerminationStrategy):
    """A custom termination strategy that stops after user approval."""
    async def should_agent_terminate(self, agent, history):
        if "approved" in history[-1].content.lower():
            return True
        return False


# -----------------
# Agent Instantiation
# -----------------
# Multi-Agent Specifications & GroupChat Topologies
# 3. Instantiate each agent with the correct name, prompt, and temperature setting.
python_agent = create_agent(
    "PythonExecutorAgent", AGENT_CONFIG["PythonExecutorAgent"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.1),
)
cleaning_agent = create_agent(
    "DataCleaning", AGENT_CONFIG["DataCleaning"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.7),
)
stats_agent = create_agent(
    "DataStatistics", AGENT_CONFIG["DataStatistics"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.5),
)
checker_agent = create_agent(
    "AnalysisChecker", AGENT_CONFIG["AnalysisChecker"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.2),
)
report_agent = create_agent(
    "ReportGenerator", AGENT_CONFIG["ReportGenerator"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=1.0),
)
report_checker_agent = create_agent(
    "ReportChecker", AGENT_CONFIG["ReportChecker"], chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.2),
)


# -----------------
# Group Chats
# -----------------
# Multi-Agent Specifications & GroupChat Topologies
# 4. Create the three agent group chats.
analysis_chat = AgentGroupChat(
    agents=[cleaning_agent, stats_agent, checker_agent],
    termination_strategy=ApprovalTerminationStrategy(agents=[checker_agent], maximum_iterations=9),
)
code_chat = AgentGroupChat(
    agents=[python_agent],
    termination_strategy=ApprovalTerminationStrategy(agents=[python_agent], maximum_iterations=1),
)
report_chat = AgentGroupChat(
    agents=[report_agent, report_checker_agent],
    termination_strategy=ApprovalTerminationStrategy(agents=[report_checker_agent], maximum_iterations=6),
)


# -----------------
# Main Workflow
# -----------------
# End-to-End Multi-Stage Pipeline Orchestrator
# Implement the main workflow logic, following the sequence described in the instructions.
async def main():
    """The main entry point for the agentic workflow."""
    print("\n=== AgentQuant: Agentic Data Analysis Workflow ===")

    # 1. Load the CSV data.
    try:
        csv_path = await asyncio.to_thread(get_csv_name)
        csv_data = load_csv_file(csv_path)
        original_series = build_labeled_series(csv_path)
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
        print(f"\nFailed to load CSV data: {exc}")
        return
    print(f"\nLoaded data from '{csv_path}'.")

    initial_prompt = (
        "Raw CSV values (flattened): "
        f"{csv_data}\n\n"
        "Structured dataset (JSON array of {\"label\": ..., \"value\": ...} objects, "
        "in original row order):\n"
        f"{json.dumps(original_series)}\n\n"
        "Begin the data cleaning and statistical analysis workflow."
    )

    # 2. Invoke the analysis chat.
    print("\n--- Running Analysis Team (Cleaning -> Statistics -> Validation) ---")
    await analysis_chat.add_chat_message(initial_prompt)

    analysis_messages = []
    async for message in analysis_chat.invoke():
        log_agent_message(message)
        print(f"\n[{message.name}]:\n{message.content}")
        analysis_messages.append(message)

    cleaning_messages = [m for m in analysis_messages if m.name == cleaning_agent.name]
    if not cleaning_messages:
        print("\nThe Data Cleaning Assistant produced no output. Aborting workflow.")
        return
    last_cleaning_message = cleaning_messages[-1].content

    # 3. Get human approval.
    print("\n--- Analysis complete. Please review the results above. ---")
    approval = await asyncio.to_thread(
        input, "Do you approve continuing to the visualization step? (yes/no): "
    )
    approval = approval.strip().lower()
    if approval != "yes":
        print("Workflow halted: approval not granted.")
        return

    # 4. Save the cleaned data.
    cleaned_series = extract_cleaned_data(last_cleaning_message)
    if not cleaned_series:
        print("\nCould not parse cleaned data from the Data Cleaning Assistant's response. Aborting workflow.")
        return
    try:
        with open("data-cleaned.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_series, f, indent=2)
    except OSError as exc:
        print(f"\nFailed to save cleaned data: {exc}")
        return
    print("\nSaved cleaned data to 'data-cleaned.json'.")

    # 5. Invoke the code chat to generate and execute visualization code.
    print("\n--- Running Visualization Team ---")
    viz_prompt = (
        "Original dataset (JSON array of {\"label\": ..., \"value\": ...} objects):\n"
        f"{json.dumps(original_series)}\n\n"
        "Cleaned dataset (JSON array of {\"label\": ..., \"value\": ...} objects):\n"
        f"{json.dumps(cleaned_series)}\n\n"
        "Generate the Python visualization code as instructed."
    )
    await code_chat.add_chat_message(viz_prompt)

    # 6. Execute the code in a retry loop.
    executor = PythonExecutor(max_attempts=3)
    working_code = None
    for attempt in range(1, executor.max_attempts + 1):
        print(f"\nAttempt {attempt}/{executor.max_attempts}: generating visualization code...")

        code_response = None
        async for message in code_chat.invoke():
            log_agent_message(message)
            code_response = message.content
            print(f"\n[{message.name}]:\n{code_response}")

        if code_response is None:
            print("\nThe Python Executor Agent produced no output. Aborting workflow.")
            return

        success, error = executor.run(code_response)
        if success:
            working_code = code_response
            print("\nVisualization code executed successfully.")
            break

        print(f"\nExecution failed:\n{error}")
        if attempt < executor.max_attempts:
            await code_chat.add_chat_message(
                f"The previous code failed with this error:\n{error}\n"
                "Please fix the code. Output ONLY the corrected raw Python code."
            )

    if working_code is None:
        print("\nFailed to generate working visualization code after multiple attempts.")
        return

    # 7. Save the working visualization script.
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/data_visualization_code.py", "w", encoding="utf-8") as f:
        f.write(strip_code_fences(working_code))
    print("\nSaved working visualization script to 'artifacts/data_visualization_code.py'.")

    # 8. Invoke the report chat to generate the final report.
    print("\n--- Running Report Team ---")
    log_lines = load_logs("agent_chat.log")
    log_text = "\n".join(log_lines)
    report_prompt = (
        "Below is the full agent interaction log for this data analysis session:\n\n"
        f"{log_text}\n\n"
        "Using this log, generate the final report following the required structure."
    )
    await report_chat.add_chat_message(report_prompt)

    report_messages = []
    async for message in report_chat.invoke():
        log_agent_message(message)
        print(f"\n[{message.name}]:\n{message.content}")
        report_messages.append(message)

    report_gen_messages = [m for m in report_messages if m.name == report_agent.name]
    if not report_gen_messages:
        print("\nThe Report Generator produced no output. Aborting workflow.")
        return
    final_report = report_gen_messages[-1].content

    # 9. Save the final report.
    save_final_report(final_report)
    print("\n=== Workflow complete. Final report saved to 'artifacts/final_report.md'. ===")


# -----------------
# Main Execution
# -----------------
if __name__ == "__main__":
    asyncio.run(main())