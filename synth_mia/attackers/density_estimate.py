import numpy as np
import pandas as pd
from scipy import stats
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch

class DensityEstimate(BaseAttacker):
    """
    Black Box Density Estimate Attack.
    Houssiau, F., Jordon, J., Cohen, S. N., Daniel, O., Elliott, A., Geddes, J., Mole, C., Rangel-Smith, C., and Szpruch, L. 
    Tapas: a toolbox for adversarial privacy auditing of synthetic data. 
    arXiv preprint arXiv:2211.06550, 2022.
    """
    def __init__(self, estimation_method="kde", bw_method="silverman", 
                 epochs=100, save=False, **kwargs):
        """Initialize Density Estimate attacker.
        
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
        self.name = "Density Estimator"

    def _compute_density(self, X_test: np.ndarray, fit_data: np.ndarray) -> np.ndarray:

        method = self.estimation_method

        if method == "kde":
            p_fit = stats.gaussian_kde(fit_data.T, bw_method=self.bw_method)
            return p_fit.evaluate(X_test.T)

        elif method == "bnaf":
            _, fit_model = density_estimator_trainer(fit_data, epochs=self.epochs, save=self.save)
            p_fit_evaluated = np.exp(
                compute_log_p_x(fit_model, torch.as_tensor(X_test).float().to(self.device))
                .cpu()
                .detach()
                .numpy()
            )
            return p_fit_evaluated

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")
    

    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self._compute_density(X_test, synth)
