# Agentic_RAG_FOMC_FRED_Project
Research Study on Agentic RAG application of FOMC and FRED dataset

Project Setup and Execution Guide:-

A] All required datasets are available in the `data/` folder for reference.

B] Execution in Google Colab (EDA & Database Generation)

Please use **Google Colab with GPU runtime** for the following steps:

1. Run `data_load_fomc.ipynb` for implementation of Exploratory Data Analysis (EDA) on FOMC dataset

2. Run `data_load_FRED.ipynb` for implementation of Exploratory Data Analysis (EDA) on FRED dataset

3. Run `FOMC_and_FRED_RAG.ipynb` to generate below files:-
   Master FRED dataset: fred_master_df.pkl
   Embedded vector database: FOMC_ChromaDB_Data_v2
   FOMC chunk dataset: fomc_chunk_new_df.pkl

Note : Ensure the following dataset files are uploaded to Colab environment before execution:

FOMC_updated_dataset.xlsx
fed_fund_data.csv
gdp_fred_data.csv
inflation_data.csv
population_data.csv

C] Execution in VS Code (End-to-End Pipeline)

Run the following scripts locally:

1. Execute Agentic and Traditional RAG pipelines:

   agentic_rag_pipeline_vscode.py
   traditional_rag_pipeline_vscode.py

2. Run evaluation or assessment of RAG models:

   evaluation_vscode.py

3. Launch Streamlit UI: ui_app.py (using command streamlit run ui_app.py in terminal)

D] API Configuration

The system requires API keys for: OpenAI and Anthropic

These keys should be provided via an external configuration file before execution.

---

Important notes to consider:-

1. Please ensure all dependencies are installed before running the scripts.
2. GPU runtime is recommended for faster processing in Colab
3. If the preview of the notebook fails on GitHub, download and run locally in Colab
