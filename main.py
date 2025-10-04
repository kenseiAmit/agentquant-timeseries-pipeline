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
