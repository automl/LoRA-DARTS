# LoRA-DARTS

## Installation and Development
First, install the dependencies required for development and testing in your environment.

```
conda create -n lora_darts python=3.9
conda activate lora_darts
pip install -e ".[dev, test]"
pip install -e ".[benchmark]"
```

Install the precommit hooks
```
pre-commit install
```

Run the tests
```
pytest tests
```

Try running an example
```
python examples/run_lora_darts.py
```
