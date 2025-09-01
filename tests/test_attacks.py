import pytest
import pandas as pd
import numpy as np
import os
import itertools
import sys

# Add the parent directory to the Python path to ensure we import the local version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

##############################
# This file runs comprehensive testing for synth-mia library
# Change workdir to root dir and run: pytest tests/ -v
##############################

from synth_mia.attackers import (
    GenLRA, DCR, MC, LOGAN, DCRDiff, DPI, DOMIAS, 
    Classifier, LocalNeighborhood, DensityEstimate
)
from synth_mia import utils
from synth_mia.base import BaseAttacker
from synth_mia.evaluation import AttackEvaluator
from utils import generate_fake_data, load_data, get_subfolders

# Get list of subfolders as dataset sources
try:
    data_folders = get_subfolders('example_data')
except (FileNotFoundError, OSError):
    data_folders = []  # Handle case where example_data doesn't exist

# Updated attacker settings with proper class names and parameters
attacker_settings = [
    (GenLRA, {'k_nearest': 20}),  
    (DCR, None),
    (MC, None),
    (LOGAN, None),
    (DCRDiff, None),
    (DPI, None),    
    (DOMIAS, None),
    (Classifier, None),
    (LocalNeighborhood, None),
    (DensityEstimate, None),
]

attacker_ids = [
    f"{attacker.__name__}_{params}" if params else f"{attacker.__name__}"
    for attacker, params in attacker_settings
]

class TestBaseAttacker:
    """Test the BaseAttacker class functionality."""
    
    def test_base_attacker_initialization(self):
        """Test BaseAttacker initialization with hyperparameters."""
        params = {'param1': 'value1', 'param2': 42}
        attacker = BaseAttacker(**params)
        
        assert attacker.hyper_parameters == params
        assert attacker.param1 == 'value1'
        assert attacker.param2 == 42
        
    def test_base_attacker_get_properties(self):
        """Test get_properties method."""
        params = {'learning_rate': 0.01, 'epochs': 100}
        attacker = BaseAttacker(**params)
        
        properties = attacker.get_properties()
        assert properties == params
        
    def test_base_attacker_data_validation(self):
        """Test input data validation."""
        mem = np.random.rand(10, 5)
        non_mem = np.random.rand(8, 5)
        synth = np.random.rand(12, 5)
        ref = np.random.rand(6, 5)
        
        attacker = BaseAttacker()
        
        # Should not raise exception with valid data
        attacker._validate_input_data(mem, non_mem, synth, ref)
        
        # Should raise exception with mismatched columns
        invalid_synth = np.random.rand(12, 4)  # Different number of columns
        with pytest.raises(ValueError, match="same number of columns"):
            attacker._validate_input_data(mem, non_mem, invalid_synth, ref)
            
        # Should raise exception with different data types
        invalid_mem = mem.astype(np.int32)  # Different dtype
        with pytest.raises(ValueError, match="same data type"):
            attacker._validate_input_data(invalid_mem, non_mem, synth, ref)
    
    def test_build_test_data(self):
        """Test building test data from mem and non_mem."""
        mem = np.random.rand(10, 5)
        non_mem = np.random.rand(8, 5)
        
        attacker = BaseAttacker()
        X_test = attacker._build_X_test(mem, non_mem)
        y_test = attacker._build_y_test(mem, non_mem)
        
        assert X_test.shape == (18, 5)  # Combined data
        assert y_test.shape == (18,)    # Combined labels
        assert np.array_equal(X_test[:10], mem)  # First part is mem
        assert np.array_equal(X_test[10:], non_mem)  # Second part is non_mem
        assert np.all(y_test[:10] == 1)  # Mem labels are 1
        assert np.all(y_test[10:] == 0)  # Non-mem labels are 0


class TestAttackEvaluator:
    """Test the AttackEvaluator class functionality."""
    
    def setup_method(self):
        """Set up test data for evaluation."""
        np.random.seed(42)
        self.true_labels = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        self.predicted_scores = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.7, 0.1, 0.85, 0.15])
        self.evaluator = AttackEvaluator(self.true_labels, self.predicted_scores)
    
    def test_roc_metrics(self):
        """Test ROC metrics calculation."""
        metrics = self.evaluator.roc_metrics()
        
        assert 'auc_roc' in metrics
        assert 0 <= metrics['auc_roc'] <= 1
        
        # Test default FPR targets
        expected_fprs = [0, 0.001, 0.01, 0.1]
        for fpr in expected_fprs:
            assert f'tpr_at_fpr_{fpr}' in metrics
            assert 0 <= metrics[f'tpr_at_fpr_{fpr}'] <= 1
    
    def test_roc_metrics_custom_fprs(self):
        """Test ROC metrics with custom FPR targets."""
        custom_fprs = [0.05, 0.2, 0.5]
        metrics = self.evaluator.roc_metrics(target_fprs=custom_fprs)
        
        for fpr in custom_fprs:
            assert f'tpr_at_fpr_{fpr}' in metrics
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = self.evaluator.classification_metrics()
        
        expected_keys = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'true_positive_rate', 'false_positive_rate'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1
    
    def test_privacy_metrics(self):
        """Test privacy metrics calculation."""
        metrics = self.evaluator.privacy_metrics()
        
        assert 'mia_advantage' in metrics
        assert 'privacy_gain' in metrics
        assert -1 <= metrics['mia_advantage'] <= 1
        assert 0 <= metrics['privacy_gain'] <= 2
    
    def test_epsilon_evaluator(self):
        """Test epsilon evaluation."""
        # Create more realistic test data for epsilon evaluation
        np.random.seed(42)
        # Create clear separation between member and non-member scores
        member_scores = np.random.uniform(0.6, 1.0, 20)  # Higher scores for members
        non_member_scores = np.random.uniform(0.0, 0.4, 20)  # Lower scores for non-members
        
        true_labels = np.concatenate([np.ones(20), np.zeros(20)])
        predicted_scores = np.concatenate([member_scores, non_member_scores])
        
        evaluator = AttackEvaluator(true_labels, predicted_scores)
        metrics = evaluator.epsilon_evaluator()
        
        expected_keys = [
            'threshold', 'confidence_level', 
            'epsilon_lower_bound', 'epsilon_upper_bound'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        assert metrics['confidence_level'] == 0.9
        assert metrics['threshold'] is not None
        assert metrics['epsilon_lower_bound'] <= metrics['epsilon_upper_bound']


# Define pytest parametrize to automatically loop through all data folders
@pytest.mark.parametrize("data_folder", data_folders)
@pytest.mark.parametrize("attacker_setting", attacker_settings, ids=attacker_ids)
@pytest.mark.skipif(len(data_folders) == 0, reason="No example data folders found")
def test_attack_evaluation(data_folder, attacker_setting):
    """Test attack evaluation with real data folders."""
    # Load the data
    mem_df, non_mem_df, ref_df, synth_df = load_data(data_folder)
    
    # Use the modern TabularPreprocessor pattern with ordinal encoding to avoid singular matrices
    prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='ordinal', numeric_encoding='standard')
    prep.fit(mem_df, non_mem_df, synth_df)
    mem, non_mem, synth, ref, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
    
    # Instantiate the attacker
    attacker_class, hyperparameters = attacker_setting
    if hyperparameters:
        attacker = attacker_class(**hyperparameters)
    else:
        attacker = attacker_class()
    
    # Run the attack (note: updated API returns y_true, scores)
    y_true, scores = attacker.attack(mem, non_mem, synth, ref)
    
    # Evaluate the attack with updated API
    eval_results = attacker.eval(y_true, scores, metrics=["roc"])
    
    # Assert that the evaluation results meet expected conditions
    assert eval_results is not None, "Evaluation results should not be None"
    assert isinstance(eval_results, dict), "Evaluation results should be a dictionary"

    # Expected keys in eval_results for ROC metrics
    expected_keys = ['auc_roc', 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']
    
    # Check that all expected keys are present in eval_results
    for key in expected_keys:
        assert key in eval_results, f"Missing key in eval_results: {key}"
        value = eval_results[key]
        # Assert that the values are within a reasonable range (0.0 to 1.0)
        assert isinstance(value, (float, int)), f"{key} should be a number, got {type(value)}"
        assert 0.0 <= value <= 1.0, f"{key} value {value} out of expected range [0.0, 1.0]"

    # Print results for debugging
    dataset_name = os.path.basename(data_folder)
    print(f"Results for {attacker.__class__.__name__} on dataset {dataset_name}: {eval_results}")


# Generate all combinations of the parameter choices for fake data
n_num_choices = [0, 5]
n_cat_choices = [0, 5] 
n_rows_choices = [30, 100, 500]
missing_percent_choices = [0]

parameter_combinations = list(itertools.product(
    n_num_choices,
    n_cat_choices,
    n_rows_choices,
    missing_percent_choices,
    attacker_settings
))

# Format the combinations into the desired structure
test_parameters = [
    (n_num, n_cat, n_rows, missing_percent, attacker_setting)
    for n_num, n_cat, n_rows, missing_percent, attacker_setting in parameter_combinations 
    if (n_cat > 0 or n_num > 0)
]   
    
# Generate descriptive test IDs for better output
def generate_test_id(params):
    n_num, n_cat, n_rows, missing_percent, attacker_setting = params
    attacker_name = attacker_setting[0].__name__
    attacker_params = attacker_setting[1]
    attacker_id = f"{attacker_name}_{attacker_params}" if attacker_params else attacker_name
    return f"num={n_num}_cat={n_cat}_rows={n_rows}_missing={missing_percent}_attacker={attacker_id}"

test_ids = [generate_test_id(params) for params in test_parameters]

# Parametrize with custom test IDs
@pytest.mark.parametrize("n_num, n_cat, n_rows, missing_percent, attacker_setting", test_parameters, ids=test_ids)
def test_attack_fake_data(n_num, n_cat, n_rows, missing_percent, attacker_setting):
    """Test attacks on generated fake data with various configurations."""
    # Generate the fake data
    chunk_size = min(10000, n_rows)
    mem_df, non_mem_df, ref_df, synth_df = generate_fake_data(n_num, n_cat, n_rows, chunk_size, missing_percent)
    
    # Use the modern TabularPreprocessor pattern with ordinal encoding to avoid singular matrices
    prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='ordinal', numeric_encoding='standard')
    prep.fit(mem_df, non_mem_df, synth_df)
    mem, non_mem, synth, ref, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
    
    # Instantiate the attacker
    attacker_class, hyperparameters = attacker_setting
    if hyperparameters:
        attacker = attacker_class(**hyperparameters)
    else:
        attacker = attacker_class()
    
    # Run the attack (note: updated API returns y_true, scores)
    y_true, scores = attacker.attack(mem, non_mem, synth, ref)
    
    # Evaluate the attack with updated API
    eval_results = attacker.eval(y_true, scores, metrics=["roc"])
    
    # Assert that the evaluation results meet expected conditions
    assert eval_results is not None, "Evaluation results should not be None"
    assert isinstance(eval_results, dict), "Evaluation results should be a dictionary"

    # Expected keys in eval_results for ROC metrics
    expected_keys = ['auc_roc', 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']
    
    # Check that all expected keys are present in eval_results
    for key in expected_keys:
        assert key in eval_results, f"Missing key in eval_results: {key}"
        value = eval_results[key]
        # Assert that the values are within a reasonable range (0.0 to 1.0)
        assert isinstance(value, (float, int)), f"{key} should be a number, got {type(value)}"
        assert 0.0 <= value <= 1.0, f"{key} value {value} out of expected range [0.0, 1.0]"

    # Print results for debugging
    print(f"Results for {attacker.__class__.__name__} with data (num={n_num}, cat={n_cat}, rows={n_rows}, missing={missing_percent}): {eval_results}")


class TestAttackerSpecific:
    """Test specific attacker implementations."""
    
    def setup_method(self):
        """Set up test data for attacker-specific tests."""
        np.random.seed(42)
        self.mem = np.random.rand(50, 10)
        self.non_mem = np.random.rand(50, 10)
        self.synth = np.random.rand(100, 10)
        self.ref = np.random.rand(100, 10)
    
    @pytest.mark.parametrize("attacker_class, params", attacker_settings, ids=attacker_ids)
    def test_attacker_instantiation(self, attacker_class, params):
        """Test that all attackers can be instantiated properly."""
        if params:
            attacker = attacker_class(**params)
            # Check that parameters are set
            for key, value in params.items():
                assert hasattr(attacker, key)
                assert getattr(attacker, key) == value
        else:
            attacker = attacker_class()
        
        assert isinstance(attacker, BaseAttacker)
        assert hasattr(attacker, 'attack')
        assert hasattr(attacker, 'eval')
        assert hasattr(attacker, 'get_properties')
    
    @pytest.mark.parametrize("attacker_class, params", attacker_settings, ids=attacker_ids)
    def test_attacker_attack_method(self, attacker_class, params):
        """Test that all attackers' attack methods work properly."""
        if params:
            attacker = attacker_class(**params)
        else:
            attacker = attacker_class()
        
        # Test attack method
        y_true, scores = attacker.attack(self.mem, self.non_mem, self.synth, self.ref)
        
        # Check return types and shapes
        assert isinstance(y_true, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert y_true.shape == (100,)  # 50 mem + 50 non_mem
        assert scores.shape == (100,)
        
        # Check label correctness
        assert np.all(y_true[:50] == 1)  # First 50 are member labels
        assert np.all(y_true[50:] == 0)  # Last 50 are non-member labels
        
        # Check that scores are finite
        assert np.all(np.isfinite(scores)), f"Attack scores contain non-finite values for {attacker_class.__name__}"


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline_with_all_metrics(self):
        """Test complete pipeline with all available metrics."""
        # Generate small test data
        np.random.seed(42)
        mem = np.random.rand(20, 5)
        non_mem = np.random.rand(20, 5)
        synth = np.random.rand(40, 5)
        ref = np.random.rand(40, 5)
        
        # Test with one attacker
        attacker = GenLRA(k_nearest=10)
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        
        # Test all metrics
        all_metrics = ["roc", "classification", "privacy", "epsilon"]
        eval_results = attacker.eval(y_true, scores, metrics=all_metrics)
        
        # Check that all metric categories are present
        assert len(eval_results) > 0
        
        # ROC metrics should be present
        assert 'auc_roc' in eval_results
        
        # Classification metrics should be present
        assert 'accuracy' in eval_results
        
        # Privacy metrics should be present
        assert 'mia_advantage' in eval_results
        
        # Epsilon metrics should be present
        assert 'epsilon_lower_bound' in eval_results
        assert 'epsilon_upper_bound' in eval_results
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimal data
        mem = np.random.rand(2, 3)
        non_mem = np.random.rand(2, 3)
        synth = np.random.rand(4, 3)
        
        attacker = MC()  # Simple attacker for edge case testing
        y_true, scores = attacker.attack(mem, non_mem, synth)
        
        assert len(y_true) == 4
        assert len(scores) == 4
        
        # Test evaluation with minimal data
        eval_results = attacker.eval(y_true, scores, metrics=["roc"])
        assert 'auc_roc' in eval_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
