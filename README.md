# An Empirical Analysis of Hallucination Behavior in LLM Medical Domain

## Creating and Activating the Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## Installing Required Packages
```bash
pip install -r requirements.txt
```

## Loading the Dataset
```python
 python load_dataset.py
```
## Running the Model
get token from Huggingface (following: https://huggingface.co/docs/hub/security-tokens)
copy the token and paste it when running the following command:
```bash
huggingface-cli login
python run.py
```