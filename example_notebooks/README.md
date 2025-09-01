# Example Notebooks

This directory contains Jupyter notebooks demonstrating various aspects of the Synth-MIA framework for Membership Inference Attacks.

## Notebooks

### 1. `tutorial.ipynb`
**General Pipeline Tutorial**

This notebook provides a comprehensive introduction to using Synth-MIA for privacy auditing. It covers:
- Data preprocessing and splitting
- Running basic attacks (DCR example)
- Evaluating attack results
- Using different evaluation metrics

### 2. `creating_new_attacks.ipynb`
**Creating Custom MIA Attackers**

Learn how to implement your own Membership Inference Attack methods:
- Understanding the `BaseAttacker` class
- Implementing custom attack logic
- Step-by-step example of creating a DCR attacker
- Testing and evaluating custom attackers

### 3. `deploying_multiple_attacks.ipynb`
**Running Multiple Attacks**

Demonstrates how to:
- Deploy multiple different attack methods simultaneously
- Compare attack performance across different techniques
- Evaluate and compare results in a structured way

## Getting Started

1. **Data Requirements**: The notebooks expect example data to be available in the `../example_data/` directory relative to the notebooks location.

2. **File Paths**: All data loading paths in these notebooks use relative paths (`../example_data/...`) to work correctly from the `example_notebooks/` directory.

3. **Running the Notebooks**:
   ```bash
   # Navigate to the example_notebooks directory
   cd example_notebooks
   
   # Start Jupyter
   jupyter notebook
   ```

## Data Structure

The notebooks expect the following data structure:
```
example_data/
├── housing/
│   ├── mem.csv          # Member data
│   ├── non_mem.csv      # Non-member data  
│   ├── synth.csv        # Synthetic data
│   └── ref.csv          # Reference data
└── insurance/
    ├── train_set.csv    # Training data
    ├── holdout_set.csv  # Test data
    └── bayesian_network_synth_250.csv  # Synthetic data
```

## Prerequisites

- Python 3.7+
- Jupyter Notebook
- synth-mia package installed
- Required dependencies (pandas, numpy, scipy, etc.)

## Notes

- The notebooks are designed to work from the `example_notebooks/` directory
- File paths have been adjusted to use `../` to access data in the parent directory
- All notebooks maintain the same functionality as the original ones, just with corrected paths
