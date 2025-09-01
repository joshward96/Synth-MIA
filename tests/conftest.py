import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the Python path to ensure we import the local version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_fake_data
from synth_mia import utils


@pytest.fixture(scope="session")
def sample_data_small():
    """Generate small sample data for quick tests."""
    np.random.seed(42)
    mem = np.random.rand(10, 5)
    non_mem = np.random.rand(10, 5)
    synth = np.random.rand(20, 5)
    ref = np.random.rand(20, 5)
    return mem, non_mem, synth, ref


@pytest.fixture(scope="session")
def sample_data_medium():
    """Generate medium-sized sample data for standard tests."""
    np.random.seed(42)
    mem = np.random.rand(50, 10)
    non_mem = np.random.rand(50, 10)
    synth = np.random.rand(100, 10)
    ref = np.random.rand(100, 10)
    return mem, non_mem, synth, ref


@pytest.fixture(scope="session")
def sample_dataframes():
    """Generate sample pandas DataFrames with mixed data types."""
    np.random.seed(42)
    
    def create_mixed_df(n_rows, prefix=""):
        return pd.DataFrame({
            f'numeric1': np.random.rand(n_rows),
            f'numeric2': np.random.randint(1, 100, n_rows),
            f'categorical1': np.random.choice(['A', 'B', 'C'], n_rows),
            f'categorical2': np.random.choice(['X', 'Y', 'Z'], n_rows),
            f'binary': np.random.choice([0, 1], n_rows)
        })
    
    mem_df = create_mixed_df(30, "mem_")
    non_mem_df = create_mixed_df(30, "non_mem_")
    ref_df = create_mixed_df(60, "ref_")
    synth_df = create_mixed_df(60, "synth_")
    
    return mem_df, non_mem_df, ref_df, synth_df


@pytest.fixture(scope="session")
def preprocessed_dataframes(sample_dataframes):
    """Generate preprocessed data from sample DataFrames."""
    mem_df, non_mem_df, ref_df, synth_df = sample_dataframes
    
    # Use the modern TabularPreprocessor pattern
    prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
    prep.fit(mem_df, non_mem_df, synth_df)
    mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
    
    metadata = {
        'feature_names': prep.get_feature_names_out().tolist(),
        'n_features': len(prep.get_feature_names_out()),
        'transformer': transformer
    }
    
    return mem_proc, non_mem_proc, ref_proc, synth_proc, metadata


@pytest.fixture(scope="function")
def evaluation_data():
    """Generate consistent evaluation data for testing metrics."""
    np.random.seed(42)
    # Create a realistic scenario where members have higher scores
    member_scores = np.random.beta(2, 1, 50)  # Skewed towards higher values
    non_member_scores = np.random.beta(1, 2, 50)  # Skewed towards lower values
    
    true_labels = np.concatenate([np.ones(50), np.zeros(50)])
    predicted_scores = np.concatenate([member_scores, non_member_scores])
    
    return true_labels, predicted_scores


@pytest.fixture(scope="session", params=[
    (5, 0, 100),    # Only numerical features
    (0, 5, 100),    # Only categorical features  
    (3, 3, 100),    # Mixed features
    (10, 5, 50),    # Many features, few samples
])
def parametric_fake_data(request):
    """Parametric fixture for generating different types of fake data."""
    n_num, n_cat, n_rows = request.param
    chunk_size = min(10000, n_rows)
    
    mem_df, non_mem_df, ref_df, synth_df = generate_fake_data(
        n_num=n_num, 
        n_cat=n_cat, 
        n_rows=n_rows, 
        chunk_size=chunk_size, 
        missing_percent=0
    )
    
    # Use the modern TabularPreprocessor pattern
    prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
    prep.fit(mem_df, non_mem_df, synth_df)
    mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
    
    metadata = {
        'feature_names': prep.get_feature_names_out().tolist(),
        'n_features': len(prep.get_feature_names_out()),
        'transformer': transformer
    }
    
    return {
        'raw': (mem_df, non_mem_df, ref_df, synth_df),
        'processed': (mem_proc, non_mem_proc, ref_proc, synth_proc),
        'metadata': metadata,
        'params': (n_num, n_cat, n_rows)
    }


@pytest.fixture(scope="session")
def all_attacker_classes():
    """Fixture providing all available attacker classes."""
    from synth_mia.attackers import (
        GenLRA, DCR, MC, LOGAN, DCRDiff, DPI, DOMIAS,
        Classifier, LocalNeighborhood, DensityEstimate
    )
    
    return [
        GenLRA, DCR, MC, LOGAN, DCRDiff, 
        DPI, DOMIAS, Classifier, LocalNeighborhood, DensityEstimate
    ]


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """Create a temporary directory with sample CSV files."""
    # Create temporary CSV files for testing data loading
    mem_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A']
    })
    non_mem_data = pd.DataFrame({
        'feature1': [6, 7, 8],
        'feature2': ['B', 'A', 'B']
    })
    ref_data = pd.DataFrame({
        'feature1': [0, 1, 2],
        'feature2': ['A', 'B', 'A']
    })
    synth_data = pd.DataFrame({
        'feature1': [9, 10, 11],
        'feature2': ['A', 'B', 'A']
    })
    
    mem_data.to_csv(tmp_path / "mem.csv", index=False)
    non_mem_data.to_csv(tmp_path / "non_mem.csv", index=False)
    ref_data.to_csv(tmp_path / "ref.csv", index=False)
    synth_data.to_csv(tmp_path / "synth.csv", index=False)
    
    return tmp_path


@pytest.fixture(autouse=True)
def reset_random_state():
    """Automatically reset random state before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup if needed


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to tests in certain classes
        if "TestBase" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.unit)
        elif "TestIntegration" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.integration)
        
        # Add 'slow' marker to parameterized tests with many parameters
        if hasattr(item, 'callspec') and len(getattr(item.callspec, 'params', {})) > 5:
            item.add_marker(pytest.mark.slow)
        
        # Add 'requires_data' marker to tests that use example_data
        if "example_data" in str(item.function):
            item.add_marker(pytest.mark.requires_data)
