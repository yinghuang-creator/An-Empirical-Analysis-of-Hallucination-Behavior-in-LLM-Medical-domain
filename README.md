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

## Running experiments
```python src/run_experiments.pu --runs {select run config codes from the table in the run_experiments} --n {no. of samples}
```

### Example:
```
python src/run_experiments.py --runs 1 2 3 --n 200
```
