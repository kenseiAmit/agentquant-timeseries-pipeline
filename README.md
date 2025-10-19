# AgentQuant: Multi-Agent Time-Series Cleaning, Audit & Self-Healing Visualization Pipeline

Four-stage multi-agent quantitative data analysis workflow built on **Microsoft Semantic Kernel (`1.37.0`)** and **Azure OpenAI (`gpt-4.1`)**—coordinating **6 specialized agents** across **3 `AgentGroupChat` topologies** with deterministic auditor feedback loops, interactive **Human-in-the-Loop (HITL)** approval gating, and **self-healing `PythonExecutor` matplotlib code generation**.

---

## Multi-Agent GroupChat Architecture

```mermaid
flowchart TD
    A["Raw Time-Series CSV<br/>(data/timeseries_sep2025.csv)"] --> B["Stage 1: analysis_chat (AgentGroupChat)<br/>DataCleaning -> DataStatistics -> AnalysisChecker"]
    B --> C{"Stage 2: Human-in-the-Loop<br/>Approval Gate"}
    C -- "Approved" --> D["Persist data-cleaned.json"]
    D --> E["Stage 3: code_chat + PythonExecutor (Max 3 Retries)<br/>PythonExecutorAgent Self-Healing Matplotlib Generation"]
    E --> F["Stage 4: report_chat (AgentGroupChat)<br/>ReportGenerator <-> ReportChecker -> artifacts/final_report.md"]
```

---

