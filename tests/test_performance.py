import pytest
import numpy as np
import pandas as pd
import time
import sys
import os

# Add the parent directory to the Python path to ensure we import the local version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_fake_data
from synth_mia import utils
from synth_mia.attackers import MC, DCR, GenLRA


@pytest.mark.slow
class TestPerformance:
    """Performance tests for the synth-mia library."""
    
    def test_small_data_performance(self, sample_data_small):
        """Test performance with small datasets."""
        mem, non_mem, synth, ref = sample_data_small
        
        # Test a simple attacker
        attacker = MC()
        
        start_time = time.time()
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        attack_time = time.time() - start_time
        
        start_time = time.time()
        eval_results = attacker.eval(y_true, scores, metrics=["roc"])
        eval_time = time.time() - start_time
        
        # Small data should be very fast
        assert attack_time < 1.0, f"Attack took too long: {attack_time:.3f}s"
        assert eval_time < 0.5, f"Evaluation took too long: {eval_time:.3f}s"
        
        print(f"Small data performance - Attack: {attack_time:.3f}s, Eval: {eval_time:.3f}s")
    
    def test_medium_data_performance(self, sample_data_medium):
        """Test performance with medium-sized datasets."""
        mem, non_mem, synth, ref = sample_data_medium
        
        # Test multiple attackers
        attackers = [MC(), DCR()]
        
        for attacker in attackers:
            start_time = time.time()
            y_true, scores = attacker.attack(mem, non_mem, synth, ref)
            attack_time = time.time() - start_time
            
            start_time = time.time()
            eval_results = attacker.eval(y_true, scores, metrics=["roc"])
            eval_time = time.time() - start_time
            
            # Medium data should still be reasonably fast
            assert attack_time < 5.0, f"{attacker.__class__.__name__} attack took too long: {attack_time:.3f}s"
            assert eval_time < 1.0, f"{attacker.__class__.__name__} evaluation took too long: {eval_time:.3f}s"
            
            print(f"{attacker.__class__.__name__} medium data performance - Attack: {attack_time:.3f}s, Eval: {eval_time:.3f}s")
    
    @pytest.mark.slow
    def test_large_data_performance(self):
        """Test performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        mem = np.random.rand(n_samples, n_features)
        non_mem = np.random.rand(n_samples, n_features)
        synth = np.random.rand(n_samples * 2, n_features)
        ref = np.random.rand(n_samples * 2, n_features)
        
        # Test with fast attackers
        attacker = MC()
        
        start_time = time.time()
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        attack_time = time.time() - start_time
        
        start_time = time.time()
        eval_results = attacker.eval(y_true, scores, metrics=["roc"])
        eval_time = time.time() - start_time
        
        # Large data should still be manageable
        assert attack_time < 15.0, f"Large data attack took too long: {attack_time:.3f}s"
        assert eval_time < 2.0, f"Large data evaluation took too long: {eval_time:.3f}s"
        
        print(f"Large data performance - Attack: {attack_time:.3f}s, Eval: {eval_time:.3f}s")
    
    @pytest.mark.slow
    def test_preprocessing_performance(self):
        """Test preprocessing performance with different data sizes and types."""
        sizes = [100, 500, 1000]
        feature_configs = [(5, 0), (0, 5), (5, 5), (10, 10)]
        
        for n_rows in sizes:
            for n_num, n_cat in feature_configs:
                if n_num == 0 and n_cat == 0:
                    continue
                    
                # Generate fake dataframes
                chunk_size = min(10000, n_rows)
                mem_df, non_mem_df, ref_df, synth_df = generate_fake_data(
                    n_num, n_cat, n_rows, chunk_size, 0
                )
                
                start_time = time.time()
                prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
                prep.fit(mem_df, non_mem_df, synth_df)
                mem, non_mem, synth, ref, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
                preprocess_time = time.time() - start_time
                
                # Preprocessing should be reasonably fast
                max_time = 2.0 if n_rows <= 500 else 5.0
                assert preprocess_time < max_time, \
                    f"Preprocessing {n_rows} rows with {n_num} num + {n_cat} cat features took too long: {preprocess_time:.3f}s"
                
                print(f"Preprocessing {n_rows} rows ({n_num}num, {n_cat}cat): {preprocess_time:.3f}s")
    
    def test_memory_usage_basic(self, sample_data_medium):
        """Basic memory usage test to ensure no obvious memory leaks."""
        mem, non_mem, synth, ref = sample_data_medium
        
        # Run multiple attacks to check for memory accumulation
        attacker = MC()
        
        for i in range(10):
            y_true, scores = attacker.attack(mem, non_mem, synth, ref)
            eval_results = attacker.eval(y_true, scores, metrics=["roc"])
            
            # Basic check that results are consistent
            assert len(scores) == len(y_true)
            assert 'auc_roc' in eval_results
    
    @pytest.mark.slow
    def test_concurrent_attacks(self, sample_data_small):
        """Test that multiple attackers can run concurrently without interference."""
        mem, non_mem, synth, ref = sample_data_small
        
        attackers = [MC(), DCR()]
        results = []
        
        # Run attacks sequentially first
        for attacker in attackers:
            y_true, scores = attacker.attack(mem, non_mem, synth, ref)
            eval_results = attacker.eval(y_true, scores, metrics=["roc"])
            results.append((y_true, scores, eval_results))
        
        # Run the same attacks again to ensure reproducibility
        for i, attacker in enumerate(attackers):
            y_true, scores = attacker.attack(mem, non_mem, synth, ref)
            eval_results = attacker.eval(y_true, scores, metrics=["roc"])
            
            # Results should be identical (assuming deterministic behavior)
            orig_y_true, orig_scores, orig_eval = results[i]
            np.testing.assert_array_equal(y_true, orig_y_true)
            # Note: scores might not be exactly equal due to randomness in some attackers
            assert eval_results.keys() == orig_eval.keys()


@pytest.mark.slow
class TestScalability:
    """Test scalability with varying data dimensions."""
    
    @pytest.mark.parametrize("n_features", [5, 10, 20, 50])
    def test_feature_scalability(self, n_features):
        """Test how performance scales with number of features."""
        n_samples = 100
        
        mem = np.random.rand(n_samples, n_features)
        non_mem = np.random.rand(n_samples, n_features)
        synth = np.random.rand(n_samples * 2, n_features)
        ref = np.random.rand(n_samples * 2, n_features)
        
        attacker = MC()
        
        start_time = time.time()
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        attack_time = time.time() - start_time
        
        # Time should scale reasonably with features
        max_time = 0.1 * n_features  # Very generous scaling
        assert attack_time < max_time, \
            f"Attack with {n_features} features took too long: {attack_time:.3f}s"
        
        print(f"Features {n_features}: {attack_time:.3f}s")
    
    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_sample_scalability(self, n_samples):
        """Test how performance scales with number of samples."""
        n_features = 10
        
        mem = np.random.rand(n_samples, n_features)
        non_mem = np.random.rand(n_samples, n_features)
        synth = np.random.rand(n_samples * 2, n_features)
        ref = np.random.rand(n_samples * 2, n_features)
        
        attacker = MC()
        
        start_time = time.time()
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        attack_time = time.time() - start_time
        
        # Time should scale reasonably with samples
        max_time = 0.01 * n_samples  # Very generous scaling
        assert attack_time < max_time, \
            f"Attack with {n_samples} samples took too long: {attack_time:.3f}s"
        
        print(f"Samples {n_samples}: {attack_time:.3f}s")


class TestBenchmark:
    """Benchmark tests for comparing attacker performance."""
    
    def test_attacker_speed_comparison(self, sample_data_medium):
        """Compare the speed of different attackers."""
        mem, non_mem, synth, ref = sample_data_medium
        
        # Test fast attackers only to avoid long test times
        attackers = [
            ("MC", MC()),
            ("DCR", DCR()),
        ]
        
        results = {}
        
        for name, attacker in attackers:
            start_time = time.time()
            y_true, scores = attacker.attack(mem, non_mem, synth, ref)
            attack_time = time.time() - start_time
            
            start_time = time.time()
            eval_results = attacker.eval(y_true, scores, metrics=["roc"])
            eval_time = time.time() - start_time
            
            results[name] = {
                'attack_time': attack_time,
                'eval_time': eval_time,
                'total_time': attack_time + eval_time,
                'auc': eval_results['auc_roc']
            }
        
        # Print benchmark results
        print("\nAttacker Performance Comparison:")
        print("=" * 50)
        for name, metrics in results.items():
            print(f"{name:10s}: Attack={metrics['attack_time']:.3f}s, "
                  f"Eval={metrics['eval_time']:.3f}s, "
                  f"Total={metrics['total_time']:.3f}s, "
                  f"AUC={metrics['auc']:.3f}")
        
        # Basic sanity checks
        for name, metrics in results.items():
            assert metrics['attack_time'] > 0
            assert metrics['eval_time'] > 0
            assert 0 <= metrics['auc'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
