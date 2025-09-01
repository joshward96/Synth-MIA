import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import math
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch 

class DOMIAS(BaseAttacker):
    """
    Shadow-Box DOMIAS attack. 
    van Breugel, B., Sun, H., Qian, Z., and van der Schaar, M. Membership inference attacks against synthetic data through overfitting detection, 2023. 
    URL https://arxiv.org/abs/2302.12580
    """
    def __init__(self, estimation_method="kde", bw_method="silverman", 
                 epochs=100, save=False, **kwargs):
        """Initialize DOMIAS attacker.
        
        Args:
            estimation_method: Density estimation method, "kde" or "bnaf" (default: "kde")
            bw_method: Bandwidth method for KDE (default: "silverman")
            epochs: Number of epochs for BNAF training (default: 100)
            save: Whether to save BNAF model (default: False)
            **kwargs: Additional parameters
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(estimation_method=estimation_method, bw_method=bw_method,
                        epochs=epochs, save=save, **kwargs)
        self.name = "DOMIAS"

    def _fit_estimator(self, fit_data: np.ndarray):
        method = self.estimation_method
        
        if method == "kde":
            return stats.gaussian_kde(fit_data.T, bw_method=self.bw_method)
        elif method == "bnaf":
            return density_estimator_trainer(fit_data, epochs=self.epochs, save=self.save)[1]
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")


    def _compute_density(self, X_test: np.ndarray, model_fit: Any) -> np.ndarray:
        method = self.estimation_method

        if method == "kde":
            return model_fit.evaluate(X_test.T)
        elif method == "bnaf":
            return np.exp(
                compute_log_p_x(model_fit, torch.as_tensor(X_test).float().to(self.device)).cpu().detach().numpy()
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")
    
    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        de_synth = self._fit_estimator(synth)
        de_ref = self._fit_estimator(ref)

        P_s = self._compute_density(X_test, de_synth)
        P_r = self._compute_density(X_test, de_ref)

        return P_s / (P_r +1e-20)
