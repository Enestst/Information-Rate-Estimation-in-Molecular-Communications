# Information-Rate-Estimation-in-Molecular-Communications
Information Rate Estimation in Molecular Communications

# Project Setup & Environment

Follow these steps to get the project running on your local machine.

## 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed. You can check your version by running:
```bash
python3 --version  # macOS / Linux
python --version   # Windows
```

## 2. Create a Virtual Environment
A virtual environment keeps your project dependencies isolated.
macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies
Once the environment is active, upgrade pip and install the required libraries.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Optional: Managing Packages
If you install new libraries and want to save them to the project configuration for others:
```bash
pip freeze > requirements.txt
```
Note: Only run the command above if you want to update the official dependency list. If you are just running the project for the first time, you can skip this!