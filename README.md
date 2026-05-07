# Synth-MIA: A Testbed for Auditing Privacy Leakage in Tabular Data Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

**Synth-MIA** is a comprehensive open-source framework for conducting membership inference attacks (MIA) on synthetic tabular data generators. The library provides a unified interface for implementing and evaluating various privacy attacks, enabling researchers and practitioners to assess the privacy risks of their synthetic data generation models.

## Available Attack Methods

- **DCR**: Distance Closest Record 
- **DCRDiff**: Calibrated variant of DCR
- **GenLRA**: Generative Likelihood Ratio Attack
- **LOGAN**: LOss-based Generative Adversarial Network attack
- **DOMIAS**: DOMIAS
- **DPI**: Data Plagiarism Index
- **MC**: Monte Carlo-based attack
- **Classifier**: Machine learning classifier attack
- **LocalNeighborhood**: Local neighborhood attack
- **DensityEstimate**: Density estimation attack

## Installation

### From Source

```bash
git clone Anonymized
cd Synth-MIA
pip install -e .
```

### Requirements

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- SciPy
- torch 
See `requirements.txt` for complete dependency list.

## Quick Start

```python
import pandas as pd
from synth_mia.attackers import DCR
from synth_mia.utils import TabularPreprocessor, create_random_equal_dfs

# Load your datasets
train_data = pd.read_csv('path/to/training_data.csv')
test_data = pd.read_csv('path/to/test_data.csv')  
synthetic_data = pd.read_csv('path/to/synthetic_data.csv')

# Create member/non-member splits
non_member_set, reference_set = create_random_equal_dfs(test_data, 250, num_dfs=2, seed=42)

# Preprocess data
preprocessor = TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
preprocessor.fit(train_data, non_member_set, synthetic_data)
mem, non_mem, synth, ref, _ = preprocessor.transform(train_data, non_member_set, synthetic_data)

# Run attack
attacker = DCR()
true_labels, scores = attacker.attack(mem, non_mem, synth, ref)

# Evaluate results  
results = attacker.eval(true_labels, scores, metrics=['roc', 'classification', 'privacy'])
print(results)
```

## Usage

### Data Preparation

The framework expects three primary datasets:

- **Member Set**: Training data used to generate the synthetic dataset
- **Non-Member Set**: Holdout data not used during training 
- **Synthetic Set**: Generated synthetic data to be audited
- **Reference Set** (optional): Additional reference data for some attacks

### Preprocessing Pipeline

```python
from synth_mia.utils import TabularPreprocessor

# Initialize preprocessor
preprocessor = TabularPreprocessor(
    fit_target='synth',  # Fit preprocessing on synthetic data
    categorical_encoding='one-hot',  # or 'ordinal', 'passthrough'
    numeric_encoding='standard'  # or 'minmax', 'passthrough'
)

# Fit and transform
preprocessor.fit(member_data, non_member_data, synthetic_data)
processed_data = preprocessor.transform(member_data, non_member_data, synthetic_data)
```

### Running Multiple Attacks

```python
from synth_mia.attackers import DCR, GenLRA, LOGAN, DOMIAS

attackers = [DCR(), GenLRA(), LOGAN(), DOMIAS()]
results = {}

for attacker in attackers:
    true_labels, scores = attacker.attack(mem, non_mem, synth, ref)
    eval_results = attacker.eval(true_labels, scores, metrics=['roc', 'privacy'])
    results[attacker.name] = eval_results

# Convert to DataFrame for easy analysis
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)
```

### Evaluation Metrics

The framework provides comprehensive evaluation through the `AttackEvaluator` class:

```python
# Available metric types
metrics = ['roc', 'classification', 'privacy', 'epsilon']

# ROC metrics: AUC and TPR at specific FPR thresholds
results = attacker.eval(true_labels, scores, metrics=['roc'], target_fprs=[0.001, 0.01, 0.1])

# Privacy metrics: MIA advantage and privacy gain  
results = attacker.eval(true_labels, scores, metrics=['privacy'])

# Epsilon-differential privacy bounds
results = attacker.eval(true_labels, scores, metrics=['epsilon'], confidence_level=0.9)
```

## Replicating Paper Experiments

The experiments from the paper can be replicated using the benchmark data and scripts provided in this repository.

### Step 1: Download the Benchmark Data

Pull the benchmark datasets from Hugging Face and place them in the `experiments/data` directory:

```bash
# Using the Hugging Face CLI
huggingface-cli download joshward/synth-mia-benchmark --repo-type dataset --local-dir experiments/data
```

Alternatively, you can download the data directly from [https://huggingface.co/datasets/joshward/synth-mia-benchmark](https://huggingface.co/datasets/joshward/synth-mia-benchmark) and manually place the contents into the `experiments/data` directory.

### Step 2: Run the Attack Experiments

Once the data is in place, execute the full attack benchmark by running:

```bash
python run_attack_experiment.py \
    --base_dir exp_data/ \
    --json_output mia_results.json \
    --log_dir experiment_logs
```

This script will run all attack methods across the benchmark datasets and output the resulting MIA and fidelity metrics.

### Step 3: Reproduce Paper Figures and Tables

For convenience, the results of our experiments are already included in the repository as `MIA_results.json`. To reproduce the figures and tables from the paper directly, open and run the provided notebook:

```bash
jupyter notebook experiments/analyze_final_data.ipynb
```

This notebook loads `MIA_results.json` and regenerates all paper figures and tables, so you can replicate the analysis without needing to re-run the full attack experiments.

## Project Structure

```
synth_mia/
├── __init__.py          # Main package initialization with dynamic attacker discovery
├── base.py              # BaseAttacker abstract class
├── evaluation.py        # AttackEvaluator for comprehensive metrics
├── utils.py             # Data preprocessing and utility functions
└── attackers/           # Attack implementations
    ├── dcr.py           # Distance-based Closest Record
    ├── gen_lra.py       # Generalized Likelihood Ratio Attack
    └── ...              # Additional attack methods
```

## Extending the Framework

To implement a custom attack, inherit from `BaseAttacker`:

```python
from synth_mia.base import BaseAttacker
import numpy as np

class CustomAttack(BaseAttacker):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _compute_attack_scores(self, X_test, synth, ref=None):
        # Implement your attack logic here
        # Return attack scores for each test sample
        scores = np.random.rand(len(X_test))  # Placeholder
        return scores
```

## Citation

If you use Synth-MIA in your research, please cite:

```bibtex
Anonymized
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Repository**: Anonymized
- **Issues**: Anonymized

For questions or suggestions, please open an issue on GitHub.