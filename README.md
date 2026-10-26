# Credit Card Transaction Screening Assistant

An enterprise-grade hybrid AI system for real-time credit card fraud detection. This project combines deterministic machine learning for precise risk scoring with a multi-agent orchestration workflow and LLM-driven compliance reporting.

---
## Key Features

* **Hybrid Architecture**: Combines **XGBoost** for numerical fraud probability scoring with **OpenAI GPT-4o-mini** (via LangChain) for narrative reporting.
* **Multi-Agent Orchestration**: Built using **LangGraph** to coordinate specialized agents across intake validation, scoring, cost-policy evaluation, and final executive report generation.
* **PCA Safety Protocols**: Enforces strict prompt constraints to prevent the LLM from hallucinating real-world semantic meanings for anonymized features ($V1$–$V28$).
* **Resilient Fallback Engine**: Built-in exception handling catches API rate limits/quota errors (429) to ensure zero system downtime and deterministic report fallback.

---
## System Architecture & Workflow

The transaction screening pipeline is orchestrated using a stateful **LangGraph** workflow:

1. **Intake Agent (`intake_agent.py`)**: Sanitizes and validates transaction schema, feature counts, and non-negative amount inputs.
2. **Fraud Scoring Agent (`fraud_agent.py`)**: Executes a pre-trained **XGBoost** model to calculate exact fraud probability.
3. **Cost Policy Agent (`cost_policy_agent.py`)**: Evaluates model risk score against business cost-benefit decision rules:
   * **Low Risk (< 0.30)**: `APPROVE`
   * **Medium Risk (0.30 – 0.75)**: `STEP-UP AUTHENTICATION`
   * **High Risk (> 0.75)**: `BLOCK`
4. **Validation Agent (`validation_agent.py`)**: Synthesizes agent decisions and generates an executive summary using LLM narrative output with strict fallback protection.

---
##Project Structure

```text
credit_card_fraud_assistant/
│
├── data/
│   └── creditcard.csv       # ULB Credit Card Fraud Dataset
├── models/
│   ├── fraud_classifier.pkl # Trained XGBoost model
│   └── scaler.pkl           # Feature scaler
├── src/
│   ├── agents/
│   │   ├── intake_agent.py  # Data validation agent
│   │   ├── fraud_agent.py   # XGBoost scoring agent
│   │   ├── cost_policy_agent.py # Threshold business rule agent
│   │   ├── validation_agent.py  # LLM report generation agent
│   │   └── graph.py         # LangGraph workflow orchestration
│   ├── config.py            # Paths and global configuration settings
│   ├── model_tool.py        # ML model inference tool
│   ├── preprocessing.py     # Data cleaning & pipeline transformations
│   ├── state.py             # Agent state definition
│   └── utils.py             # Utility functions and report formatting
├── .env                     # Environment variables (API Keys)
├── .gitignore               # Ignored files (virtualenvs, .env, caches)
├── main.py                  # Entry point to execute the screening assistant
├── requirements.txt         # Project dependencies
└── train_model.py           # Model training and evaluation script
