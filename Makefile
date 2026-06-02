# EcoSort — common project commands.
# Usage: make <target>

PY := python

.PHONY: help install data features train evaluate experiment all app clean

help:
	@echo "EcoSort targets:"
	@echo "  install     Install Python dependencies into the active environment"
	@echo "  data        Download TrashNet subset + build split index"
	@echo "  features    Extract classical colour+HOG features"
	@echo "  train       Train all three models (naive, classical, deep)"
	@echo "  evaluate    Evaluate on the test set (metrics + confusion matrices)"
	@echo "  experiment  Run the robustness experiment"
	@echo "  all         Full pipeline: data -> features -> train -> evaluate -> experiment"
	@echo "  app         Run the inference web app locally"
	@echo "  clean       Remove generated outputs (keeps raw data)"

install:
	pip install -r requirements.txt

data:
	$(PY) scripts/make_dataset.py

features:
	$(PY) scripts/build_features.py

train:
	$(PY) scripts/model.py

evaluate:
	$(PY) scripts/evaluate.py

experiment:
	$(PY) scripts/experiment.py

all:
	$(PY) setup.py

app:
	$(PY) main.py

clean:
	rm -f data/outputs/*.png data/outputs/*.json data/processed/*.npz
	find . -type d -name __pycache__ -exec rm -rf {} +
