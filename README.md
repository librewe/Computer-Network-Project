# Project Setup

## Prerequisites

- Windows PowerShell
- A working Python installation available in your terminal

## Create a Virtual Environment

Run these commands in the project root:

```powershell
python -m venv .venv
```

## Activate the Virtual Environment

In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this once in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Runtime Config

Runtime defaults for the integrated demo live in [src-trained/config.yaml](C:\Users\25117\Desktop\大二下\网络\大作业\project\src-trained\config.yaml).

Use it for:

- model path
- proxy host and port
- dashboard port
- label names

Do not use it for page layout or detailed UI design. Those should stay in code.

## Run the Demo

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Option 1: Launch the Dashboard Demo

This starts the Streamlit page and, inside the page process, starts the proxy thread as well.

```powershell
streamlit run .\src-trained\dashboard.py
```

Then open the browser URL shown by Streamlit, normally:

```text
http://localhost:8501
```

Set your browser proxy to:

```text
127.0.0.1:8888
```

These default values come from `src-trained/config.yaml`.

### Option 2: Launch the Proxy-Only Inference Service

If you only want to verify model loading and proxy startup:

```powershell
python .\src-trained\realtime_inference.py
```

### Run the Integration Test

```powershell
python .\scripts\test_system_integration.py
```
