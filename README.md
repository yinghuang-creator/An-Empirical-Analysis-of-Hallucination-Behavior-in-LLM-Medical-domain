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

Optional (only needed for gated/private models): get a token from Hugging Face (following [this guide](https://huggingface.co/docs/hub/security-tokens)).
copy the token and paste it when running the following command:

```bash
huggingface-cli login
python ./src/run.py
```

You can also select model via environment variable:

```bash
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct python ./src/run.py
```

If using a gated model (e.g., Llama), set token in env:

```bash
HF_TOKEN=your_token_here MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct python ./src/run.py
```

