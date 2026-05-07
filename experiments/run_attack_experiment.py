#!/usr/bin/env python3
"""
Large-scale Privacy Attack Experiment Runner

This script processes datasets from a structured directory and runs multiple privacy attacks,
logging comprehensive results and performance metrics.
"""

import os
import sys
import time
import json
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
import csv

# Import your modules (adjust imports based on your actual module structure)
from synth_mia.attackers import *
from synth_mia import utils, evaluation

# Additional imports for utility and statistical evaluations
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon
import torch


def safe_json_serialize(obj):
    """
    Custom JSON serializer that handles numpy types, infinity, NaN values, and numpy boolean keys
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        else:
            return float(obj)
    elif isinstance(obj, np.bool_):  # Handle numpy boolean
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif obj == float('inf'):
        return "Infinity"
    elif obj == float('-inf'):
        return "-Infinity"
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {safe_json_serialize_key(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj


def safe_json_serialize_key(key):
    """
    Convert dictionary keys to JSON-compatible types
    """
    if isinstance(key, np.bool_):
        return bool(key)
    elif isinstance(key, np.integer):
        return int(key)
    elif isinstance(key, np.floating):
        if np.isnan(key):
            return "NaN"
        elif np.isinf(key):
            return "Infinity" if key > 0 else "-Infinity"
        else:
            return float(key)
    elif isinstance(key, (str, int, float, bool, type(None))):
        return key
    else:
        # Convert any other type to string as fallback
        return str(key)


def clean_results_for_json(results):
    """
    Enhanced clean results dictionary to ensure JSON serialization compatibility
    """
    def clean_value(value):
        if isinstance(value, (int, str, bool, type(None))):
            return value
        elif isinstance(value, float):
            if np.isnan(value):
                return None
            elif np.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            else:
                return value
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            if np.isnan(value):
                return None
            elif np.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            else:
                return float(value)
        elif isinstance(value, np.bool_):  # Handle numpy boolean
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            # Clean both keys and values
            cleaned_dict = {}
            for k, v in value.items():
                clean_key = clean_key_for_json(k)
                clean_val = clean_value(v)
                cleaned_dict[clean_key] = clean_val
            return cleaned_dict
        elif isinstance(value, (list, tuple)):
            return [clean_value(item) for item in value]
        else:
            try:
                # Try to convert to string as fallback
                return str(value)
            except:
                return "unconvertible_value"
    
    def clean_key_for_json(key):
        """Convert dictionary keys to JSON-compatible types"""
        if isinstance(key, np.bool_):
            return bool(key)
        elif isinstance(key, np.integer):
            return int(key)
        elif isinstance(key, np.floating):
            if np.isnan(key):
                return "NaN"
            elif np.isinf(key):
                return "Infinity" if key > 0 else "-Infinity"
            else:
                return float(key)
        elif isinstance(key, (str, int, float, bool, type(None))):
            return key
        else:
            # Convert any other type to string as fallback
            return str(key)
    
    return clean_value(results)


def save_results_to_json(self, experiment_key: str, results: Dict, timing: Dict):
    """Save results as a single entry in JSON file with enhanced error handling"""
    try:
        # Load existing results
        try:
            with open(self.json_file, 'r') as f:
                all_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_results = {}
        
        # Parse experiment key for metadata
        parts = experiment_key.split('/')
        dataset_name, model_name, seed, synth_name = parts
        
        # Clean results for JSON serialization with enhanced cleaning
        clean_results = clean_results_for_json(results)
        clean_timing = clean_results_for_json(timing)
        
        # Create comprehensive result entry
        result_entry = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'model_name': model_name,
                'seed': seed,
                'synth_name': synth_name,
                'experiment_key': experiment_key
            },
            'timing': clean_timing,
            'results': clean_results,
            'status': {
                'completed_at': datetime.now().isoformat(),
                'success': True
            }
        }
        
        # Add error information if any attacks failed
        failed_attacks = []
        error_count = 0
        
        privacy_results = clean_results.get('privacy_attacks', {})
        for attacker_name, attack_result in privacy_results.items():
            if isinstance(attack_result, dict) and 'error' in attack_result:
                failed_attacks.append(str(attacker_name))
                error_count += 1
        
        if isinstance(clean_results.get('utility_evaluation', {}), dict) and 'error' in clean_results.get('utility_evaluation', {}):
            error_count += 1
        
        if isinstance(clean_results.get('statistical_metrics', {}), dict) and 'error' in clean_results.get('statistical_metrics', {}):
            error_count += 1
        
        result_entry['status'].update({
            'error_count': error_count,
            'failed_attacks': failed_attacks,
            'success': error_count == 0
        })
        
        # Add to results
        all_results[experiment_key] = result_entry
        
        # Write back to file with custom serializer
        with open(self.json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=safe_json_serialize, ensure_ascii=False)
        
        self.logger.info(f"Results saved to JSON: {self.json_file}")
        
    except Exception as e:
        self.logger.error(f"Failed to save JSON results: {e}")
        self.logger.error(f"Error details: {traceback.format_exc()}")
        
        # Try to save a simplified version
        try:
            simplified_results = {
                'experiment_key': str(experiment_key),
                'timestamp': datetime.now().isoformat(),
                'error': f"JSON serialization failed: {str(e)}",
                'original_error': str(e),
                'error_type': type(e).__name__
            }
            
            backup_file = self.experiment_dir / f"backup_{experiment_key.replace('/', '_')}.json"
            with open(backup_file, 'w') as f:
                json.dump(simplified_results, f, indent=2, default=str)
            
            self.logger.info(f"Backup results saved to: {backup_file}")
            
        except Exception as backup_error:
            self.logger.error(f"Failed to save backup results: {backup_error}")
            
            # Last resort: save as pickle
            try:
                import pickle
                pickle_file = self.experiment_dir / f"pickle_{experiment_key.replace('/', '_')}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump({'results': results, 'timing': timing, 'experiment_key': experiment_key}, f)
                self.logger.info(f"Results saved as pickle: {pickle_file}")
            except Exception as pickle_error:
                self.logger.error(f"Failed to save pickle results: {pickle_error}")


class ExperimentLogger:
    """Handles logging for the privacy attack experiments"""
    
    def __init__(self, log_dir: str = "experiment_logs", json_output: str = None):
        """Initialize logger with timestamped log directory"""
        self.log_dir = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"experiment_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup JSON output
        if json_output:
            self.json_file = Path(json_output)
        else:
            self.json_file = self.experiment_dir / "results.json"
        
        # Initialize JSON file with empty dict if it doesn't exist
        if not self.json_file.exists():
            with open(self.json_file, 'w') as f:
                json.dump({}, f)
        
        # Setup logging
        log_file = self.experiment_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Error logging
        self.error_file = self.experiment_dir / "errors.json"
        self.error_log = {}
    
    
    def maximum_mean_discrepancy(self, X_real, X_syn, kernel="rbf", max_samples=5000):
        """
        Compute empirical maximum mean discrepancy between two one-hot encoded arrays.
        The lower the result, the more evidence that distributions are the same.
        
        Args:
            X_real: numpy array, ground truth one-hot encoded data (n_samples_real, n_features)
            X_syn: numpy array, synthetic one-hot encoded data (n_samples_syn, n_features)
            kernel: str, "rbf", "linear" or "polynomial"
            max_samples: int, maximum number of samples to use (to prevent memory issues)
            
        Returns:
            float: MMD score (0 = distributions are the same, 1 = totally different)
        """
        try:
            # Input validation
            if X_real.size == 0 or X_syn.size == 0:
                self.logger.warning("Empty arrays provided to MMD computation")
                return 1.0
                
            if X_real.shape[1] != X_syn.shape[1]:
                self.logger.warning(f"Feature dimension mismatch: {X_real.shape[1]} vs {X_syn.shape[1]}")
                return 1.0
            
            # Subsample if datasets are too large to prevent memory issues
            if len(X_real) > max_samples:
                indices = np.random.choice(len(X_real), max_samples, replace=False)
                X_real = X_real[indices]
                self.logger.info(f"Subsampled X_real from {len(X_real)} to {max_samples} samples")
                
            if len(X_syn) > max_samples:
                indices = np.random.choice(len(X_syn), max_samples, replace=False)
                X_syn = X_syn[indices]
                self.logger.info(f"Subsampled X_syn from {len(X_syn)} to {max_samples} samples")
            
            # Flatten arrays if needed
            X_real_flat = X_real.reshape(len(X_real), -1)
            X_syn_flat = X_syn.reshape(len(X_syn), -1)
            
            # Convert to float32 to save memory
            X_real_flat = X_real_flat.astype(np.float32)
            X_syn_flat = X_syn_flat.astype(np.float32)
            
            if kernel == "linear":
                # MMD using linear kernel (i.e., k(x,y) = <x,y>)
                delta = X_real_flat.mean(axis=0) - X_syn_flat.mean(axis=0)
                score = np.dot(delta, delta.T)
               
            elif kernel == "rbf":
                # MMD using rbf (gaussian) kernel with memory-efficient computation
                gamma = 1.0 / X_real_flat.shape[1]  # Scale gamma by feature dimension
                
                # Compute kernel matrices in chunks to avoid memory issues
                chunk_size = min(1000, len(X_real_flat), len(X_syn_flat))
                
                XX_sum = 0.0
                XX_count = 0
                for i in range(0, len(X_real_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_real_flat))
                    for j in range(0, len(X_real_flat), chunk_size):
                        end_j = min(j + chunk_size, len(X_real_flat))
                        XX_chunk = metrics.pairwise.rbf_kernel(
                            X_real_flat[i:end_i], X_real_flat[j:end_j], gamma
                        )
                        XX_sum += XX_chunk.sum()
                        XX_count += XX_chunk.size
                XX_mean = XX_sum / XX_count if XX_count > 0 else 0
                
                YY_sum = 0.0
                YY_count = 0
                for i in range(0, len(X_syn_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_syn_flat))
                    for j in range(0, len(X_syn_flat), chunk_size):
                        end_j = min(j + chunk_size, len(X_syn_flat))
                        YY_chunk = metrics.pairwise.rbf_kernel(
                            X_syn_flat[i:end_i], X_syn_flat[j:end_j], gamma
                        )
                        YY_sum += YY_chunk.sum()
                        YY_count += YY_chunk.size
                YY_mean = YY_sum / YY_count if YY_count > 0 else 0
                
                XY_sum = 0.0
                XY_count = 0
                for i in range(0, len(X_real_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_real_flat))
                    for j in range(0, len(X_syn_flat), chunk_size):
                        end_j = min(j + chunk_size, len(X_syn_flat))
                        XY_chunk = metrics.pairwise.rbf_kernel(
                            X_real_flat[i:end_i], X_syn_flat[j:end_j], gamma
                        )
                        XY_sum += XY_chunk.sum()
                        XY_count += XY_chunk.size
                XY_mean = XY_sum / XY_count if XY_count > 0 else 0
                
                score = XX_mean + YY_mean - 2 * XY_mean
               
            elif kernel == "polynomial":
                # MMD using polynomial kernel with chunked computation
                degree = 2
                gamma = 1.0 / X_real_flat.shape[1]
                coef0 = 0
                
                chunk_size = min(1000, len(X_real_flat), len(X_syn_flat))
                
                # Similar chunked computation as RBF
                XX_sum = 0.0
                XX_count = 0
                for i in range(0, len(X_real_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_real_flat))
                    XX_chunk = metrics.pairwise.polynomial_kernel(
                        X_real_flat[i:end_i], X_real_flat[i:end_i], degree, gamma, coef0
                    )
                    XX_sum += XX_chunk.sum()
                    XX_count += XX_chunk.size
                XX_mean = XX_sum / XX_count if XX_count > 0 else 0
                
                YY_sum = 0.0
                YY_count = 0
                for i in range(0, len(X_syn_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_syn_flat))
                    YY_chunk = metrics.pairwise.polynomial_kernel(
                        X_syn_flat[i:end_i], X_syn_flat[i:end_i], degree, gamma, coef0
                    )
                    YY_sum += YY_chunk.sum()
                    YY_count += YY_chunk.size
                YY_mean = YY_sum / YY_count if YY_count > 0 else 0
                
                XY_sum = 0.0
                XY_count = 0
                for i in range(0, len(X_real_flat), chunk_size):
                    end_i = min(i + chunk_size, len(X_real_flat))
                    for j in range(0, len(X_syn_flat), chunk_size):
                        end_j = min(j + chunk_size, len(X_syn_flat))
                        XY_chunk = metrics.pairwise.polynomial_kernel(
                            X_real_flat[i:end_i], X_syn_flat[j:end_j], degree, gamma, coef0
                        )
                        XY_sum += XY_chunk.sum()
                        XY_count += XY_chunk.size
                XY_mean = XY_sum / XY_count if XY_count > 0 else 0
                
                score = XX_mean + YY_mean - 2 * XY_mean
               
            else:
                raise ValueError(f"Unsupported kernel {kernel}")
            
            # Handle potential infinity/NaN values
            if np.isnan(score) or np.isinf(score):
                self.logger.warning(f"MMD computation resulted in {score}, returning 1.0")
                return 1.0
            
            # Ensure score is non-negative (MMD should always be >= 0)
            score = max(0.0, float(score))
            
            return score
            
        except Exception as e:
            self.logger.error(f"MMD computation failed: {e}")
            return 1.0  # Return maximum distance as fallback
    
    def jensen_shannon_distance(self, X_real, X_syn, max_samples=10000):
        """
        Evaluate the Jensen-Shannon distance between two one-hot encoded arrays.
        For one-hot encoded data, this computes the JS distance between the categorical distributions.
       
        Args:
            X_real: numpy array, ground truth one-hot encoded data (n_samples_real, n_features)
            X_syn: numpy array, synthetic one-hot encoded data (n_samples_syn, n_features)
            max_samples: int, maximum number of samples to use
           
        Returns:
            float: Jensen-Shannon distance between the categorical distributions
        """
        try:
            # Input validation
            if X_real.size == 0 or X_syn.size == 0:
                self.logger.warning("Empty arrays provided to Jensen-Shannon computation")
                return 1.0
                
            if X_real.shape[1] != X_syn.shape[1]:
                self.logger.warning(f"Feature dimension mismatch: {X_real.shape[1]} vs {X_syn.shape[1]}")
                return 1.0
            
            # Subsample if datasets are too large
            if len(X_real) > max_samples:
                indices = np.random.choice(len(X_real), max_samples, replace=False)
                X_real = X_real[indices]
                
            if len(X_syn) > max_samples:
                indices = np.random.choice(len(X_syn), max_samples, replace=False)
                X_syn = X_syn[indices]
            
            # Convert to float32 to save memory
            X_real = X_real.astype(np.float32)
            X_syn = X_syn.astype(np.float32)
            
            # For one-hot encoded data, compute the probability distribution over features
            # Each feature represents a different category/dimension
            real_total = np.maximum(X_real.sum(), 1e-10)  # Avoid division by zero
            syn_total = np.maximum(X_syn.sum(), 1e-10)
            
            gt_probs = X_real.sum(axis=0) / real_total
            syn_probs = X_syn.sum(axis=0) / syn_total
           
            # Ensure probabilities are non-negative and handle numerical errors
            gt_probs = np.maximum(gt_probs, 0)
            syn_probs = np.maximum(syn_probs, 0)
            
            # Add small epsilon to avoid log(0) issues in jensenshannon
            epsilon = 1e-10
            gt_probs = gt_probs + epsilon
            syn_probs = syn_probs + epsilon
           
            # Renormalize after adding epsilon
            gt_sum = gt_probs.sum()
            syn_sum = syn_probs.sum()
            
            if gt_sum <= 0 or syn_sum <= 0:
                self.logger.warning("Invalid probability sums in JS computation")
                return 1.0
                
            gt_probs = gt_probs / gt_sum
            syn_probs = syn_probs / syn_sum
            
            # Additional validation
            if not (np.isfinite(gt_probs).all() and np.isfinite(syn_probs).all()):
                self.logger.warning("Non-finite values in probability distributions")
                return 1.0
           
            # Compute Jensen-Shannon distance
            js_dist = jensenshannon(gt_probs, syn_probs)
           
            # Validate result
            if np.isnan(js_dist) or np.isinf(js_dist) or js_dist < 0:
                self.logger.warning(f"JS distance computation resulted in invalid value {js_dist}, returning 1.0")
                return 1.0
           
            return float(js_dist)
            
        except Exception as e:
            self.logger.error(f"JS distance computation failed: {e}")
            return 1.0  # Return maximum distance as fallback
    def log_experiment_start(self, base_dir: str):
        """Log the start of the experiment"""
        self.logger.info("="*80)
        self.logger.info("PRIVACY ATTACK EXPERIMENT STARTED")
        self.logger.info(f"Timestamp: {datetime.now()}")
        self.logger.info(f"Base directory: {base_dir}")
        self.logger.info(f"Results will be saved to: {self.experiment_dir}")
        self.logger.info("="*80)
        
    def log_dataset_start(self, dataset_name: str, model_name: str, seed: str, synth_name: str):
        """Log the start of processing a dataset"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {dataset_name}/{model_name}/{seed}/{synth_name}")
        self.logger.info(f"{'='*60}")
        
    def log_preprocessing(self, dataset_shapes: Dict[str, Tuple[int, int]], preprocessing_time: float):
        """Log preprocessing information"""
        self.logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        for name, shape in dataset_shapes.items():
            self.logger.info(f"  {name}: {shape}")
            
    def log_attack_start(self, attacker_name: str):
        """Log the start of an attack"""
        self.logger.info(f"\nRunning attack: {attacker_name}")
        

    def log_attack_complete(self, attacker_name: str, attack_time: float, eval_results: Dict):
        """Log completion of an attack - updated to handle multi-class results"""
        self.logger.info(f"  ✓ {attacker_name} completed in {attack_time:.2f}s")
        
        # Check if this is a multi-class result
        is_multiclass = eval_results.get('_metadata', {}).get('is_multiclass', False)
        
        if is_multiclass:
            # Log multi-class summary
            num_classes = eval_results.get('_metadata', {}).get('num_classes', 0)
            self.logger.info(f"    Multi-class results ({num_classes} classes)")
        else:
            # Log single-class results (original behavior)
            if 'auc_roc' in eval_results:
                auc = eval_results['auc_roc']
                if isinstance(auc, (int, float)) and not (np.isnan(auc) or np.isinf(auc)):
                    self.logger.info(f"    AUC-ROC: {auc:.4f}")
                else:
                    self.logger.info(f"    AUC-ROC: {auc}")
            if 'tpr_at_fpr_0.1' in eval_results:
                tpr = eval_results['tpr_at_fpr_0.1']
                if isinstance(tpr, (int, float)) and not (np.isnan(tpr) or np.isinf(tpr)):
                    self.logger.info(f"    TPR@FPR=0.1: {tpr:.3f}")
                else:
                    self.logger.info(f"    TPR@FPR=0.1: {tpr}")
    def log_utility_evaluation(self, xgb_results: Dict, statistical_results: Dict):
        """Log utility and statistical evaluation results"""
        self.logger.info(f"Utility Evaluation (XGBoost):")
        
        # Safe formatting for test_accuracy
        test_acc = xgb_results.get('test_accuracy', 'N/A')
        if isinstance(test_acc, (int, float)) and not (np.isnan(test_acc) if isinstance(test_acc, float) else False):
            self.logger.info(f"  Test Accuracy: {test_acc:.4f}")
        else:
            self.logger.info(f"  Test Accuracy: {test_acc}")
        
        # Safe formatting for test_auc
        test_auc = xgb_results.get('test_auc', 'N/A')
        if isinstance(test_auc, (int, float)) and not (np.isnan(test_auc) if isinstance(test_auc, float) else False):
            self.logger.info(f"  Test AUC: {test_auc:.4f}")
        else:
            self.logger.info(f"  Test AUC: {test_auc}")
        
        self.logger.info(f"Statistical Distance Metrics:")
        
        # Safe formatting for rbf_mmd
        rbf_mmd = statistical_results.get('rbf_mmd', 'N/A')
        if isinstance(rbf_mmd, (int, float)) and not (np.isnan(rbf_mmd) if isinstance(rbf_mmd, float) else False):
            self.logger.info(f"  RBF MMD: {rbf_mmd:.6f}")
        else:
            self.logger.info(f"  RBF MMD: {rbf_mmd}")
        
        # Safe formatting for js_distance
        js_dist = statistical_results.get('js_distance', 'N/A')
        if isinstance(js_dist, (int, float)) and not (np.isnan(js_dist) if isinstance(js_dist, float) else False):
            self.logger.info(f"  Jensen-Shannon Distance: {js_dist:.6f}")
        else:
            self.logger.info(f"  Jensen-Shannon Distance: {js_dist}")
            
    def log_attack_error(self, attacker_name: str, error: Exception, experiment_key: str):
        """Log attack errors"""
        self.logger.error(f"  ✗ {attacker_name} failed: {str(error)}")
        
        if experiment_key not in self.error_log:
            self.error_log[experiment_key] = {}
        
        self.error_log[experiment_key][attacker_name] = {
            'error': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
    def save_results_to_json(self, experiment_key: str, results: Dict, timing: Dict):
        """Save results as a single entry in JSON file"""
        try:
            # Load existing results
            try:
                with open(self.json_file, 'r') as f:
                    all_results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_results = {}
            
            # Parse experiment key for metadata
            parts = experiment_key.split('/')
            dataset_name, model_name, seed, synth_name = parts
            
            # Clean results for JSON serialization
            clean_results = clean_results_for_json(results)
            clean_timing = clean_results_for_json(timing)
            
            # Create comprehensive result entry
            result_entry = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'dataset_name': dataset_name,
                    'model_name': model_name,
                    'seed': seed,
                    'synth_name': synth_name,
                    'experiment_key': experiment_key
                },
                'timing': clean_timing,
                'results': clean_results,
                'status': {
                    'completed_at': datetime.now().isoformat(),
                    'success': True
                }
            }
            
            # Add error information if any attacks failed
            failed_attacks = []
            error_count = 0
            
            privacy_results = clean_results.get('privacy_attacks', {})
            for attacker_name, attack_result in privacy_results.items():
                if 'error' in attack_result:
                    failed_attacks.append(attacker_name)
                    error_count += 1
            
            if 'error' in clean_results.get('utility_evaluation', {}):
                error_count += 1
            
            if 'error' in clean_results.get('statistical_metrics', {}):
                error_count += 1
            
            result_entry['status'].update({
                'error_count': error_count,
                'failed_attacks': failed_attacks,
                'success': error_count == 0
            })
            
            # Add to results
            all_results[experiment_key] = result_entry
            
            # Write back to file with custom serializer
            with open(self.json_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=safe_json_serialize, ensure_ascii=False)
            
            self.logger.info(f"Results saved to JSON: {self.json_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            # Try to save a simplified version
            try:
                simplified_results = {
                    'experiment_key': experiment_key,
                    'timestamp': datetime.now().isoformat(),
                    'error': f"JSON serialization failed: {str(e)}",
                    'original_error': str(e)
                }
                
                backup_file = self.experiment_dir / f"backup_{experiment_key.replace('/', '_')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(simplified_results, f, indent=2)
                
                self.logger.info(f"Backup results saved to: {backup_file}")
                
            except Exception as backup_error:
                self.logger.error(f"Failed to save backup results: {backup_error}")
    
    def save_results(self, experiment_key: str, results: Dict, timing: Dict):
        """Save results to JSON file"""
        self.save_results_to_json(experiment_key, results, timing)
        
        # Save errors separately if any
        if self.error_log:
            try:
                clean_errors = clean_results_for_json(self.error_log)
                with open(self.error_file, 'w') as f:
                    json.dump(clean_errors, f, indent=2, default=safe_json_serialize)
            except Exception as e:
                self.logger.error(f"Failed to save error log: {e}")
                
    def log_experiment_summary(self):
        """Log final experiment summary"""
        # Initialize with empty dict if attribute doesn't exist
        if not hasattr(self, 'all_results'):
            self.all_results = {}
            
        total_experiments = len(self.all_results)
        total_errors = sum(len(errors) for errors in self.error_log.values())
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("EXPERIMENT COMPLETED")
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Total errors: {total_errors}")
        self.logger.info(f"Results saved to: {self.experiment_dir}")
        self.logger.info("="*80)


class PrivacyAttackRunner:
    """Main class for running privacy attacks"""
    
    def __init__(self, base_dir: str, log_dir: str = "experiment_logs", json_output: str = None, seed_filter: str = None):
        """Initialize the attack runner"""
        self.base_dir = Path(base_dir)
        self.logger = ExperimentLogger(log_dir, json_output)
        self.seed_filter = seed_filter
        
        # Initialize attackers with unique names that include hyperparameters
        self.attackers = self._create_attackers_with_unique_names()
        
    def _create_attackers_with_unique_names(self):
        """Create attackers with unique names that include their hyperparameters"""
        attackers = []
        
        # GenLRA with different k_nearest values
        for k in [1, 5, 10, 25, 50, 100]:
            attacker = GenLRA(k_nearest=k)
            attacker.name = f"GenLRA_k{k}"  # Override the name
            attackers.append(attacker)
            
        for k in [1, 5, 10, 25, 50, 100]:
            attacker = DPI(k_nearest=k)
            attacker.name = f"DPI_k{k}"  # Override the name
            attackers.append(attacker)
        # Other attackers (these typically don't have conflicting names)
        other_attackers = [
            DCR(),
            LOGAN(),
            DCRDiff(),
            DOMIAS(),
            MC(),
            LocalNeighborhood(),
            Classifier()
        ]
        attackers.extend(other_attackers)
        
        # DensityEstimate with different methods (if you have multiple)
        density_methods = ["kde"]  # Add more methods if available
        for method in density_methods:
            attacker = DensityEstimate(method=method)
            attacker.name = f"DensityEstimate_{method}"  # Override the name
            attackers.append(attacker)
        
        return attackers
        
    def find_experiments(self) -> List[Tuple[str, str, str, str]]:
        """Find all dataset/model/seed/synth combinations to process"""
        experiments = []
        
        for dataset_path in self.base_dir.iterdir():
            if not dataset_path.is_dir():
                continue
                
            dataset_name = dataset_path.name
            
            # Check for required files
            required_files = ['mem_set.csv', 'holdout_set.csv', 'ref_set.csv']
            if not all((dataset_path / f).exists() for f in required_files):
                self.logger.logger.warning(f"Skipping {dataset_name}: missing required files")
                continue
                
            # Find model directories
            for model_path in dataset_path.iterdir():
                if not model_path.is_dir() or model_path.name in required_files:
                    continue
                    
                model_name = model_path.name
                
                # Find seed directories
                for seed_path in model_path.iterdir():
                    if not seed_path.is_dir():
                        continue
                        
                    seed = seed_path.name
                    
                    # Apply seed filter if specified
                    if self.seed_filter and seed != self.seed_filter:
                        continue
                    
                    # Find all synthetic data files
                    synth_files = ['synth_1x.csv', 'synth_2x.csv', 'synth_3x.csv']
                    available_synth = [f for f in synth_files if (seed_path / f).exists()]
                    
                    if not available_synth:
                        self.logger.logger.warning(f"Skipping {dataset_name}/{model_name}/{seed}: no synth files found")
                        continue
                    
                    # Create experiment for each synthetic dataset
                    for synth_file in available_synth:
                        synth_name = synth_file.replace('.csv', '')  # e.g., 'synth_1x'
                        experiments.append((dataset_name, model_name, seed, synth_name))
                    
        return experiments
    
    def load_datasets(self, dataset_name: str, model_name: str, seed: str, synth_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load all datasets for a given experiment"""
        try:
            base_path = self.base_dir / dataset_name
            seed_path = base_path / model_name / seed
            
            datasets = {}
            
            # Load base datasets
            datasets['mem_set'] = pd.read_csv(base_path / 'mem_set.csv')
            datasets['holdout_set'] = pd.read_csv(base_path / 'holdout_set.csv')
            datasets['ref_set'] = pd.read_csv(base_path / 'ref_set.csv')
            
            # Load the specific synthetic dataset
            synth_file = f"{synth_name}.csv"
            datasets['synth_set'] = pd.read_csv(seed_path / synth_file)
            
            return datasets
            
        except Exception as e:
            self.logger.logger.error(f"Failed to load datasets for {dataset_name}/{model_name}/{seed}/{synth_name}: {e}")
            return None
    
    def preprocess_data(self, datasets: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Preprocess datasets using TabularPreprocessor"""
        try:
            start_time = time.time()
            
            # Initialize preprocessor
            prep = utils.TabularPreprocessor(
                fit_target='synth', 
                categorical_encoding='ordinal', 
                numeric_encoding='standard'
            )
            
            # Use the loaded datasets with row limits for mem and non_mem
            train_set = datasets['mem_set']
            non_member_set = datasets['holdout_set'] 
            synth_set = datasets['synth_set']  # Now using the specific synth dataset
            ref_set = datasets['ref_set']
            
            # Limit mem and non_mem datasets to 1000 rows if they're larger
            if len(train_set) > 1000:
                train_set = train_set.iloc[:1000]
            
            if len(non_member_set) > 1000:
                non_member_set = non_member_set.iloc[:1000]
            
            # Fit preprocessor
            prep.fit(train_set, non_member_set, synth_set, ref_set)
            
            # Transform datasets
            mem, non_mem, synth, ref, transformer = prep.transform(
                train_set, non_member_set, synth_set, ref_set
            )
            
            preprocessing_time = time.time() - start_time
            
            # Log dataset shapes
            dataset_shapes = {
                'mem': mem.shape,
                'non_mem': non_mem.shape,
                'synth': synth.shape,
                'ref': ref.shape
            }
            
            self.logger.log_preprocessing(dataset_shapes, preprocessing_time)
            
            return {
                'mem': mem,
                'non_mem': non_mem,
                'synth': synth,
                'ref': ref,
                'transformer': transformer,
                'preprocessing_time': preprocessing_time,
                'shapes': dataset_shapes
            }
            
        except Exception as e:
            self.logger.logger.error(f"Preprocessing failed: {e}")
            return None
    
    def evaluate_utility_and_statistics(self, datasets: Dict[str, pd.DataFrame], processed_data: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Evaluate synthetic data utility using XGBoost and compute statistical distance metrics
        
        Args:
            datasets: Original pandas DataFrames
            processed_data: Preprocessed numpy arrays
            
        Returns:
            Tuple of (xgboost_results, statistical_results)
        """
        xgb_results = {}
        statistical_results = {}
        
        try:
            # XGBoost Utility Evaluation
            self.logger.logger.info("Running XGBoost utility evaluation...")
            
            # Get original datasets for XGBoost (work with raw data, not preprocessed)
            synth_df = datasets['synth_set']
            holdout_df = datasets['holdout_set']
            
            # Assume last column is the target
            X_synth = synth_df.iloc[:, :-1]
            y_synth = synth_df.iloc[:, -1]
            X_holdout = holdout_df.iloc[:, :-1]
            y_holdout = holdout_df.iloc[:, -1]
            
            # One-hot encode features
            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            import pandas as pd
            
            # Identify categorical columns (non-numeric)
            categorical_columns = X_synth.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = X_synth.select_dtypes(include=['number']).columns.tolist()
            
            if categorical_columns:
                self.logger.logger.info(f"  One-hot encoding categorical columns: {categorical_columns}")
                
                # Use pandas get_dummies for consistent encoding across train/test
                # Combine datasets to ensure consistent encoding
                X_combined = pd.concat([X_synth, X_holdout], keys=['synth', 'holdout'])
                
                # One-hot encode categorical variables
                X_combined_encoded = pd.get_dummies(
                    X_combined, 
                    columns=categorical_columns, 
                    drop_first=True,  # Drop first category to avoid multicollinearity
                    dummy_na=False    # Don't create dummy for NaN values
                )
                
                # Split back into synthetic and holdout sets
                X_synth_encoded = X_combined_encoded.loc['synth']
                X_holdout_encoded = X_combined_encoded.loc['holdout']
                
                # Reset indices
                X_synth_encoded = X_synth_encoded.reset_index(drop=True)
                X_holdout_encoded = X_holdout_encoded.reset_index(drop=True)
                
                self.logger.logger.info(f"  Features after encoding: {X_synth_encoded.shape[1]} (was {X_synth.shape[1]})")
                
            else:
                self.logger.logger.info("  No categorical columns found, skipping one-hot encoding")
                X_synth_encoded = X_synth.copy()
                X_holdout_encoded = X_holdout.copy()
            
            # Encode target variables to ensure they are integers (0, 1)
            label_encoder = LabelEncoder()
            
            # Fit encoder on combined labels to ensure consistent encoding
            all_labels = pd.concat([y_synth, y_holdout])
            label_encoder.fit(all_labels)
            
            # Transform labels
            y_synth_encoded = label_encoder.transform(y_synth)
            y_holdout_encoded = label_encoder.transform(y_holdout)
            
            # Log label encoding information
            unique_original = sorted(all_labels.unique())
            unique_encoded = sorted(np.unique(np.concatenate([y_synth_encoded, y_holdout_encoded])))
            self.logger.logger.info(f"  Label encoding: {unique_original} -> {unique_encoded}")
            
            # Check if we have a valid binary classification problem
            if len(unique_encoded) != 2:
                raise ValueError(f"Expected binary classification, got {len(unique_encoded)} classes: {unique_encoded}")
            
            # Train XGBoost on synthetic data
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                verbosity=0
            )
            
            xgb_model.fit(X_synth_encoded, y_synth_encoded)
            
            # Test on holdout data
            y_pred = xgb_model.predict(X_holdout_encoded)
            y_pred_proba = xgb_model.predict_proba(X_holdout_encoded)[:, 1]
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_holdout_encoded, y_pred)
            test_auc = roc_auc_score(y_holdout_encoded, y_pred_proba)
            
            xgb_results = {
                'test_accuracy': float(test_accuracy),
                'test_auc': float(test_auc),
                'n_synth_samples': len(X_synth_encoded),
                'n_holdout_samples': len(X_holdout_encoded),
                'n_features': X_synth_encoded.shape[1],
                'n_features_original': X_synth.shape[1],
                'categorical_columns': categorical_columns,
                'label_mapping': dict(zip(unique_original, unique_encoded)),
                'n_classes': len(unique_encoded)
            }
            
        except Exception as e:
            self.logger.logger.error(f"XGBoost evaluation failed: {e}")
            xgb_results = {'error': str(e)}
        
        try:
            # Statistical Distance Metrics
            self.logger.logger.info("Computing statistical distance metrics...")
            
            # Use preprocessed data for statistical comparisons
            mem_data = processed_data['mem']  # Training data
            synth_data = processed_data['synth']  # Synthetic data
            
            # Compute RBF MMD
            rbf_mmd = self.logger.maximum_mean_discrepancy(mem_data, synth_data, kernel="rbf")
            
            # Compute Jensen-Shannon distance
            js_distance = self.logger.jensen_shannon_distance(mem_data, synth_data)
            
            statistical_results = {
                'rbf_mmd': float(rbf_mmd),
                'js_distance': float(js_distance),
                'mem_shape': mem_data.shape,
                'synth_shape': synth_data.shape
            }
            
        except Exception as e:
            self.logger.logger.error(f"Statistical evaluation failed: {e}")
            statistical_results = {'error': str(e)}
        
        # Log results
        self.logger.log_utility_evaluation(xgb_results, statistical_results)
        
        return xgb_results, statistical_results
    
    def run_attacks(self, processed_data: Dict[str, Any], experiment_key: str) -> Dict[str, Dict]:
        """Run all privacy attacks on processed data"""
        mem = processed_data['mem']
        non_mem = processed_data['non_mem'] 
        synth = processed_data['synth']
        ref = processed_data['ref']
        
        results = {}
        timing = {'preprocessing_time': processed_data['preprocessing_time']}
        
        for attacker in self.attackers:
            # Use the attacker's name (which now includes hyperparameters)
            attacker_name = attacker.name
            self.logger.log_attack_start(attacker_name)
            
            try:
                attack_start = time.time()
                
                # Execute attack
                true_labels, scores = attacker.attack(mem, non_mem, synth, ref)
                
                attack_time = time.time() - attack_start
                timing[f"{attacker_name}_time"] = attack_time
                
                # Handle different score formats
                if isinstance(scores, dict):
                    print(scores)
                    # Multi-class case (e.g., GenLRA with Multi=True)
                    eval_results = {}
                    
                    for class_key, class_scores in scores.items():
                        try:
                            # Evaluate attack for each class
                            class_eval = attacker.eval(true_labels, class_scores, metrics=['roc'])
                            eval_results[f"class_{class_key}"] = class_eval
                            
                            # Log individual class results
                            if 'auc_roc' in class_eval:
                                auc = class_eval['auc_roc']
                                if isinstance(auc, (int, float)) and not (np.isnan(auc) or np.isinf(auc)):
                                    self.logger.logger.info(f"    Class {class_key} AUC-ROC: {auc:.4f}")
                            if 'tpr_at_fpr_0.1' in class_eval:
                                tpr = class_eval['tpr_at_fpr_0.1']
                                if isinstance(tpr, (int, float)) and not (np.isnan(tpr) or np.isinf(tpr)):
                                    self.logger.logger.info(f"    Class {class_key} TPR@FPR=0.1: {tpr:.3f}")
                                    
                        except Exception as class_e:
                            self.logger.logger.error(f"    Class {class_key} evaluation failed: {class_e}")
                            eval_results[f"class_{class_key}"] = {'error': str(class_e)}
                    
                    # Add metadata about multi-class results
                    eval_results['_metadata'] = {
                        'is_multiclass': True,
                        'num_classes': len(scores),
                        'class_keys': list(scores.keys())
                    }
                    
                else:
                    # Single-class case (traditional behavior)
                    eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
                    eval_results['_metadata'] = {
                        'is_multiclass': False,
                        'num_classes': 1
                    }
                
                # Store results using the unique attacker name
                results[attacker_name] = eval_results
                
                self.logger.log_attack_complete(attacker_name, attack_time, eval_results)
                
            except Exception as e:
                self.logger.log_attack_error(attacker_name, e, experiment_key)
                results[attacker_name] = {'error': str(e)}
                timing[f"{attacker_name}_time"] = 0
        
        return results, timing

    def run_experiment(self, dataset_name: str, model_name: str, seed: str, synth_name: str) -> bool:
        """Run a complete experiment for one dataset/model/seed/synth combination"""
        experiment_key = f"{dataset_name}/{model_name}/{seed}/{synth_name}"
        self.logger.log_dataset_start(dataset_name, model_name, seed, synth_name)
        
        try:
            # Load datasets
            datasets = self.load_datasets(dataset_name, model_name, seed, synth_name)
            if datasets is None:
                return False
            
            # Preprocess data
            processed_data = self.preprocess_data(datasets)
            if processed_data is None:
                return False
            
            # Run attacks
            results, timing = self.run_attacks(processed_data, experiment_key)
            
            # Run utility and statistical evaluations
            xgb_results, statistical_results = self.evaluate_utility_and_statistics(datasets, processed_data)
            
            # Combine all results
            combined_results = {
                'privacy_attacks': results,
                'utility_evaluation': xgb_results,
                'statistical_metrics': statistical_results
            }
            
            # Combine timing data
            combined_timing = timing.copy()
            
            # Save results
            self.logger.save_results(experiment_key, combined_results, combined_timing)
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Experiment {experiment_key} failed: {e}")
            self.logger.error_log[experiment_key] = {
                'general_error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_all_experiments(self):
        """Run all experiments found in the directory structure"""
        self.logger.log_experiment_start(str(self.base_dir))
        
        # Find all experiments
        experiments = self.find_experiments()
        self.logger.logger.info(f"Found {len(experiments)} experiments to run")
        
        # Log all unique attacker names for verification
        attacker_names = [attacker.name for attacker in self.attackers]
        self.logger.logger.info(f"Configured attackers: {attacker_names}")
        
        # Run each experiment
        successful = 0
        failed = 0
        
        for dataset_name, model_name, seed, synth_name in experiments:
            if self.run_experiment(dataset_name, model_name, seed, synth_name):
                successful += 1
            else:
                failed += 1
        
        self.logger.logger.info(f"\nCompleted: {successful} successful, {failed} failed")
        self.logger.log_experiment_summary()


def main():
    """Main function to run the privacy attack experiments"""
    parser = argparse.ArgumentParser(description='Run privacy attack experiments')
    parser.add_argument('--base_dir', default='data/', 
                       help='Base directory containing experiment data')
    parser.add_argument('--log_dir', default='experiment_logs',
                       help='Directory to save experiment logs')
    parser.add_argument('--json_output', default=None,
                       help='Specific JSON file path for results')
    parser.add_argument('--seed_filter', default=None,
                       help='Filter experiments to specific seed only')
    
    args = parser.parse_args()
    
    # Check if base directory exists
    if not Path(args.base_dir).exists():
        print(f"Error: Base directory {args.base_dir} does not exist")
        sys.exit(1)
    
    # Initialize and run experiments
    runner = PrivacyAttackRunner(
        base_dir=args.base_dir,
        log_dir=args.log_dir,
        json_output=args.json_output,
        seed_filter=args.seed_filter
    )
    runner.run_all_experiments()


if __name__ == "__main__":
    main()