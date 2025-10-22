<div align="center">
  <img src="assets/hero-banner.svg" alt="AgentQuant Time-Series Pipeline" width="100%" />
  <br/><br/>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Agents-Semantic%20Kernel%20AgentGroupChat-0078D4?style=flat-square" alt="Semantic Kernel" />
    <img src="https://img.shields.io/badge/LLM-Azure%20OpenAI%20gpt--4.1-2563EB?style=flat-square" alt="Azure OpenAI" />
    <img src="https://img.shields.io/badge/Safety-HITL%20Gate%20%2B%20Self--Healing%20Exec-0D9488?style=flat-square" alt="HITL" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-475569?style=flat-square" alt="License" /></a>
  </p>
</div>

---

Four-stage multi-agent quantitative time-series workflow orchestrated with **Microsoft Semantic Kernel (`1.37.0`)** and **Azure OpenAI (`gpt-4.1`)**—coordinating **6 specialized agents** across **3 `AgentGroupChat` stages** with deterministic auditor loops, interactive **Human-in-the-Loop (HITL)** approval gating, and **self-healing `PythonExecutor` matplotlib code generation**.

![AgentQuant Time-Series Cleaning Telemetry](docs/images/agentquant_timeseries_cleaning.png)

---

## 1. Four-Stage Multi-Agent GroupChat Topology

```mermaid
flowchart TD
    A["data/timeseries_sep2025.csv<br/>(20 Daily Observations)"] --> B["Stage 1: analysis_chat (Max 9x)<br/>DataCleaning -> DataStatistics -> AnalysisChecker"]
    B --> C{"Stage 2: Human-in-the-Loop<br/>Console Approval Gate"}
    C -- "yes" --> D["Persist data-cleaned.json (n = 16)"]
    D --> E["Stage 3: code_chat + PythonExecutor (Max 3x)<br/>Self-Healing Matplotlib Script Synthesis & Run"]
    E --> F["Stage 4: report_chat (Max 6x)<br/>ReportGenerator <-> ReportChecker -> final_report.md"]
```

---

## 2. Benchmark Dataset & Executed Report ([`reports/sample_analysis_report.pdf`](reports/sample_analysis_report.pdf))

```csv
Date,Value
2025-09-01,542
2025-09-02,489
2025-09-03,563
2025-09-04,512
2025-09-05,0
2025-09-06,598
2025-09-07,621
2025-09-08,505
2025-09-09,0
2025-09-10,534
2025-09-11,511
2025-09-12,490
2025-09-13,523
2025-09-14,514
2025-09-15,2500
2025-09-16,527
2025-09-17,499
2025-09-18,5545
2025-09-19,488
2025-09-20,531
```

- **Removed Anomalies (`4`)**: `2025-09-05 (0)`, `2025-09-09 (0)`, `2025-09-15 (2500)`, `2025-09-18 (5545)`.
- **Verified Clean Statistics (`n = 16`)**: `Mean = 533.19`, `Median = 521.50`, `StdDev = 37.21`, `Min = 488`, `Max = 621`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## License

MIT License — see [LICENSE](LICENSE).
