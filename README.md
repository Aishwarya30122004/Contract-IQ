# ContractIQ Enhanced Portal

## Features
- Login-style landing page with demo roles: employee, vendor, business partner, contract reviewer
- Dashboard with Plotly charts and metrics
- PDF/DOCX/TXT contract extraction
- Clause type and risk predictions using TF-IDF + Logistic Regression
- Risk explanation hints and confidence values
- Configurable heuristic policy compliance checks
- Chat-style contract assistant that retrieves passages from the uploaded contract
- CSV report exports

## Run in VS Code
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Then:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Put `ContractIQ_clause_dataset.csv` in `data/`. Required columns: `clause_text`, `clause_type`, `risk_level`.

## Streamlit deployment
Push the folder to GitHub and create a Streamlit Community Cloud app with main file path `app.py`.

## Security / limitations
The login is a demo gate only; it does not authenticate real accounts. The chatbot is retrieval-based, not a generative LLM. Compliance rules are heuristics and the dataset is small. Do not use real credentials or upload confidential contracts without organizational approval. Human review is required.
