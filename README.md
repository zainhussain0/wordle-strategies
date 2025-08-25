# wordle-strategies


## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli run --profile fast_dev
python -m src.cli figures
# for final numbers:
python -m src.cli run --profile full_benchmark
python -m src.cli figures
