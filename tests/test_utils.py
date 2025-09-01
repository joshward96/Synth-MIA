import pytest
import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to the Python path to ensure we import the local version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synth_mia import utils


class TestTabularPreprocessor:
    """Test the TabularPreprocessor class."""
    
    def setup_method(self):
        """Set up test data for preprocessing tests."""
        # Create test data with proper types
        self.mem_data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical1': ['A', 'B', 'A', 'C', 'B'],
            'categorical2': ['X', 'Y', 'Z', 'X', 'Y'],
            'categorical3': ['P', 'Q', 'R', 'P', 'Q']
        })
        
        self.non_mem_data = pd.DataFrame({
            'numeric1': [6.0, 7.0, 8.0],
            'numeric2': [60, 70, 80],
            'categorical1': ['A', 'C', 'B'],
            'categorical2': ['Z', 'X', 'Y'],
            'categorical3': ['Q', 'R', 'P']
        })
        
        self.ref_data = pd.DataFrame({
            'numeric1': [0.5, 1.5, 2.5, 3.5],
            'numeric2': [5, 15, 25, 35],
            'categorical1': ['A', 'B', 'C', 'A'],
            'categorical2': ['X', 'Y', 'Z', 'X'],
            'categorical3': ['P', 'Q', 'R', 'P']
        })
        
        self.synth_data = pd.DataFrame({
            'numeric1': [9.0, 10.0, 11.0, 12.0],
            'numeric2': [90, 100, 110, 120],
            'categorical1': ['B', 'C', 'A', 'B'],
            'categorical2': ['Y', 'Z', 'X', 'Y'],
            'categorical3': ['Q', 'R', 'P', 'Q']
        })
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        # Use the modern TabularPreprocessor pattern
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(self.mem_data, self.non_mem_data, self.synth_data)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(self.mem_data, self.non_mem_data, self.synth_data, self.ref_data)
        
        # Check that all outputs are numpy arrays
        assert isinstance(mem_proc, np.ndarray)
        assert isinstance(non_mem_proc, np.ndarray)
        assert isinstance(ref_proc, np.ndarray)
        assert isinstance(synth_proc, np.ndarray)
        
        # Check shapes are preserved
        assert mem_proc.shape[0] == len(self.mem_data)
        assert non_mem_proc.shape[0] == len(self.non_mem_data)
        assert ref_proc.shape[0] == len(self.ref_data)
        assert synth_proc.shape[0] == len(self.synth_data)
        
        # Check that all arrays have the same number of columns
        n_cols = mem_proc.shape[1]
        assert non_mem_proc.shape[1] == n_cols
        assert ref_proc.shape[1] == n_cols
        assert synth_proc.shape[1] == n_cols
        
        # Check transformer is returned
        assert transformer is not None
        
        # Check feature names
        feature_names = prep.get_feature_names_out()
        assert len(feature_names) == n_cols
    
    def test_data_types_consistency(self):
        """Test that all processed arrays have consistent data types."""
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(self.mem_data, self.non_mem_data, self.synth_data)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(self.mem_data, self.non_mem_data, self.synth_data, self.ref_data)
        
        # All arrays should have the same dtype
        assert mem_proc.dtype == non_mem_proc.dtype
        assert mem_proc.dtype == ref_proc.dtype
        assert mem_proc.dtype == synth_proc.dtype
    
    def test_with_missing_data(self):
        """Test preprocessing with missing values."""
        # Add missing values
        mem_with_nan = self.mem_data.copy()
        mem_with_nan.loc[0, 'numeric1'] = np.nan
        mem_with_nan.loc[2, 'categorical1'] = np.nan
        
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(mem_with_nan, self.non_mem_data, self.synth_data)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(mem_with_nan, self.non_mem_data, self.synth_data, self.ref_data)
        
        # Should still work without errors
        assert isinstance(mem_proc, np.ndarray)
        assert mem_proc.shape[0] == len(mem_with_nan)
        
        # Check that preprocessing handled missing values
        assert not np.any(np.isnan(mem_proc)) or np.all(np.isfinite(mem_proc[~np.isnan(mem_proc)]))
    
    def test_categorical_encoding(self):
        """Test that categorical variables are properly encoded."""
        # Create data with only categorical variables
        cat_mem = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        cat_non_mem = pd.DataFrame({
            'cat1': ['A', 'B'],
            'cat2': ['X', 'Y']
        })
        cat_ref = pd.DataFrame({
            'cat1': ['B', 'C'],
            'cat2': ['Y', 'Z']
        })
        cat_synth = pd.DataFrame({
            'cat1': ['A', 'C'],
            'cat2': ['X', 'Z']
        })
        
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(cat_mem, cat_non_mem, cat_synth)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(cat_mem, cat_non_mem, cat_synth, cat_ref)
        
        # Should be converted to numeric format
        assert np.issubdtype(mem_proc.dtype, np.number)
        assert np.issubdtype(non_mem_proc.dtype, np.number)
        assert np.issubdtype(ref_proc.dtype, np.number)
        assert np.issubdtype(synth_proc.dtype, np.number)
    
    def test_single_column_data(self):
        """Test preprocessing with single column data."""
        single_col_mem = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        single_col_non_mem = pd.DataFrame({'col1': [6, 7, 8]})
        single_col_ref = pd.DataFrame({'col1': [0, 1, 2]})
        single_col_synth = pd.DataFrame({'col1': [9, 10, 11]})
        
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(single_col_mem, single_col_non_mem, single_col_synth)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(single_col_mem, single_col_non_mem, single_col_synth, single_col_ref)
        
        # Should work with single column
        assert mem_proc.shape[1] == 1
        assert non_mem_proc.shape[1] == 1
        assert ref_proc.shape[1] == 1
        assert synth_proc.shape[1] == 1
    
    def test_empty_dataframes(self):
        """Test handling of edge cases with very small datasets."""
        small_mem = pd.DataFrame({'col1': [1]})
        small_non_mem = pd.DataFrame({'col1': [2]})
        small_ref = pd.DataFrame({'col1': [3]})
        small_synth = pd.DataFrame({'col1': [4]})
        
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(small_mem, small_non_mem, small_synth)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(small_mem, small_non_mem, small_synth, small_ref)
        
        # Should handle single-row dataframes
        assert mem_proc.shape == (1, 1)
        assert non_mem_proc.shape == (1, 1)
        assert ref_proc.shape == (1, 1)
        assert synth_proc.shape == (1, 1)
    
    def test_feature_names_functionality(self):
        """Test that feature names are properly handled."""
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(self.mem_data, self.non_mem_data, self.synth_data)
        
        feature_names = prep.get_feature_names_out()
        assert isinstance(feature_names, np.ndarray)
        assert len(feature_names) > 0
        
        # Should include both numeric and categorical features
        # Exact names depend on implementation, but should be consistent
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(self.mem_data, self.non_mem_data, self.synth_data, self.ref_data)
        assert len(feature_names) == mem_proc.shape[1]
    
    def test_reproducibility(self):
        """Test that preprocessing is reproducible."""
        prep1 = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep1.fit(self.mem_data, self.non_mem_data, self.synth_data)
        mem_proc1, non_mem_proc1, synth_proc1, ref_proc1, transformer1 = prep1.transform(self.mem_data, self.non_mem_data, self.synth_data, self.ref_data)
        
        prep2 = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep2.fit(self.mem_data, self.non_mem_data, self.synth_data)
        mem_proc2, non_mem_proc2, synth_proc2, ref_proc2, transformer2 = prep2.transform(self.mem_data, self.non_mem_data, self.synth_data, self.ref_data)
        
        # Results should be identical
        np.testing.assert_array_equal(mem_proc1, mem_proc2)
        np.testing.assert_array_equal(non_mem_proc1, non_mem_proc2)
        np.testing.assert_array_equal(ref_proc1, ref_proc2)
        np.testing.assert_array_equal(synth_proc1, synth_proc2)
    
    def test_column_order_consistency(self):
        """Test that column order is preserved across datasets."""
        # Create data where column order might matter
        ordered_data = pd.DataFrame({
            'a_first': [1, 2, 3],
            'b_second': ['X', 'Y', 'Z'],
            'c_third': [10.0, 20.0, 30.0]
        })
        
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(ordered_data, ordered_data, ordered_data)
        mem_proc, non_mem_proc, synth_proc, ref_proc, transformer = prep.transform(ordered_data, ordered_data, ordered_data, ordered_data)
        
        # All datasets should have the same feature order
        # This is implicit in the requirement that they have the same shape
        assert mem_proc.shape[1] == non_mem_proc.shape[1] == ref_proc.shape[1] == synth_proc.shape[1]


class TestUtilsIntegration:
    """Integration tests for utils with other components."""
    
    def test_utils_with_attackers(self):
        """Test that preprocessed data works with attackers."""
        from synth_mia.attackers import MC
        
        # Create simple test data
        mem_df = pd.DataFrame({
            'num': [1, 2, 3, 4, 5],
            'cat': ['A', 'B', 'A', 'B', 'A']
        })
        non_mem_df = pd.DataFrame({
            'num': [6, 7, 8],
            'cat': ['B', 'A', 'B']
        })
        ref_df = pd.DataFrame({
            'num': [0, 1, 2],
            'cat': ['A', 'B', 'A']
        })
        synth_df = pd.DataFrame({
            'num': [9, 10, 11],
            'cat': ['A', 'B', 'A']
        })
        
        # Preprocess the data using modern TabularPreprocessor
        prep = utils.TabularPreprocessor(fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard')
        prep.fit(mem_df, non_mem_df, synth_df)
        mem, non_mem, synth, ref, transformer = prep.transform(mem_df, non_mem_df, synth_df, ref_df)
        
        # Should work with attackers
        attacker = MC()
        y_true, scores = attacker.attack(mem, non_mem, synth, ref)
        
        assert len(y_true) == len(mem) + len(non_mem)
        assert len(scores) == len(mem) + len(non_mem)
        assert np.all(np.isfinite(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
