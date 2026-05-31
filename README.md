## Quick Start

Take Windows PowerShell for example:

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the dashboard demo

```powershell
streamlit run .\src\dashboard.py -- --proxy-port 8890 --dashboard-port 8501
```

Then open:

```text
http://localhost:8501
```

Remember to set your browser proxy to:

```text
127.0.0.1:8890
```
<!-- 
### 4. Run the integration test

```powershell
python .\scripts\test_system_integration.py
``` -->

<!-- ## Config

Runtime settings are in [config.yaml](C:\Users\25117\Desktop\大二下\网络\大作业\project\config.yaml).

Important fields:

- `model.path`
- `proxy.host`
- `proxy.port`
- `dashboard.port` -->

## Project Layout

- `src/`: active runtime code
- `model/`: model definitions and selected checkpoint
<!-- - `test/`: test and helper scripts -->
