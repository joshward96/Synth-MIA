# Testing Framework

The synth-mia library includes a comprehensive testing framework designed to ensure reliability, performance, and correctness of all privacy attack implementations.

## Overview

Our testing suite provides confidence that the library works correctly across different data types, sizes, and use cases. The tests are automatically run to validate:

- **Correctness**: All 10 attack methods produce valid results
- **Robustness**: Handles various data formats and edge cases  
- **Performance**: Maintains reasonable speed and memory usage
- **Consistency**: Results are reproducible and well-formatted

## Test Coverage

### Core Functionality
- **All Attack Methods**: Comprehensive testing of GenLRA, DCR, MC, LOGAN, DCRDiff, DPI, DOMIAS, Classifier, LocalNeighborhood, and DensityEstimate
- **Evaluation Metrics**: ROC analysis, classification metrics, privacy metrics, and epsilon-differential privacy bounds
- **Data Preprocessing**: Robust handling of numerical, categorical, and mixed-type data

### Data Compatibility
- **Multiple Data Types**: Numerical, categorical, and mixed datasets
- **Various Sizes**: From small test datasets to large-scale scenarios
- **Missing Data**: Graceful handling of incomplete datasets
- **Edge Cases**: Single columns, minimal datasets, and unusual configurations

### Performance Validation
- **Speed Benchmarks**: Ensures attacks complete in reasonable time
- **Memory Efficiency**: Validates memory usage stays within bounds
- **Scalability**: Tests performance across different data dimensions

## Running Tests

### Quick Start
```bash
# Run all tests
pytest tests/ -v

# Run only fast tests (recommended for development)
pytest tests/ -m "not slow" -v

# Run tests with coverage report
pytest tests/ --cov=synth_mia -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow validation  
- **Performance Tests**: Speed and memory benchmarks
- **Robustness Tests**: Edge case and error handling

## Quality Assurance

### Automated Validation
Every test run automatically validates:
- Attack methods return properly formatted results
- Evaluation metrics fall within expected ranges (0-1 for probabilities)
- Data preprocessing handles all input types correctly
- Memory usage remains stable across multiple runs

### Reproducibility
- Consistent random seeds ensure reproducible results
- Same inputs always produce identical outputs
- Cross-platform compatibility verified

### Error Handling
- Graceful handling of malformed input data
- Clear error messages for common issues
- Robust preprocessing for real-world data quirks

## For Contributors

### Adding New Tests
When contributing new features:
1. Add corresponding tests in the appropriate test file
2. Include both success and failure cases
3. Test with various data types and sizes
4. Verify performance meets expectations

### Test Structure
```
tests/
├── test_attacks.py      # Attack method testing
├── test_utils.py        # Utility function testing  
├── test_performance.py  # Performance benchmarks
├── conftest.py         # Shared test fixtures
└── utils.py            # Test helper functions
```

## Benefits for Users

### Reliability
- Extensive testing ensures consistent behavior
- Edge cases are handled gracefully
- Performance is validated across scenarios

### Ease of Use  
- Clear examples of expected input/output formats
- Validation that your data will work correctly
- Performance expectations for planning

### Confidence
- Know that results are mathematically sound
- Trust that privacy metrics are calculated correctly
- Assurance that the library scales to your data size

The testing framework is continuously updated to maintain the highest standards of reliability and performance as the library evolves.
