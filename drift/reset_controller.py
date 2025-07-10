import numpy as np
from .drift_detector import DriftDetector

class ResetController:
    """
    Manages drift detection and triggers resets on a meta-algorithm.
    """
    def __init__(self, 
                 meta_algorithm, 
                 delta: float = 0.0008, 
                 granularity: int = 500,
                 abrupt_params: dict = None,
                 gradual_params: dict = None
                 ):
        """
        Initializes the ResetController.
        """
        self.meta_algorithm = meta_algorithm
        self.granularity = granularity
        
        # Default parameters if not provided
        self.abrupt_params = abrupt_params if abrupt_params is not None else {'cooldown': 2500, 'shrink_factor': 0.3}
        self.gradual_params = gradual_params if gradual_params is not None else {'cooldown': 2500, 'shrink_factor': 0.7}
        
        self._detector = DriftDetector(delta=delta)
        self._current_step = 0
        self._last_reset_step = -max(self.abrupt_params['cooldown'], self.gradual_params['cooldown']) # Start cooled down
        self._batch = []
        self._batch_means_history = []
        
        # Public logs for analysis
        self.drift_detection_log = []
        self.reset_log = []

    def select_algorithm(self):
        """Delegates algorithm selection to the wrapped meta-algorithm."""
        return self.meta_algorithm.select_algorithm()
    
    def _classify_drift(self, current_mean: float) -> str:
        """Classifies drift. For now, uses a simple magnitude check."""
        # A more robust mu_ref would be the mean of a stable window before drift.
        # As a proxy, we use the mean of all history before the current batch.
        if len(self._batch_means_history) > 1:
            mu_ref = np.mean(self._batch_means_history[:-1])
            delta_mu = abs(current_mean - mu_ref)
            # This threshold would also be a tunable parameter.
            if delta_mu > 0.1: 
                return "Abrupt"
        return "Gradual"

    def update(self, chosen_algo_index: int, reward: float):
        """
        Processes a single reward, updates the detector, delegates the update 
        to the meta-algorithm, and potentially triggers a reset.
        """
        # --- First, delegate the update to the wrapped meta-algorithm ---
        self.meta_algorithm.update(chosen_algo_index, reward)

        # --- Second, run the drift detection logic ---
        self._current_step += 1
        self._batch.append(reward)

        if len(self._batch) >= self.granularity:
            mean_reward = np.mean(self._batch)
            self._batch_means_history.append(mean_reward)
            self._batch = []

            self._detector.update(mean_reward)

            # Determine which cooldown period to check against
            active_cooldown = max(self.abrupt_params['cooldown'], self.gradual_params['cooldown'])
            in_cooldown = (self._current_step - self._last_reset_step) < active_cooldown

            if self._detector.drift_detected:
                self.drift_detection_log.append(self._current_step)

            if self._detector.drift_detected and not in_cooldown:
                drift_type = self._classify_drift(mean_reward)
                
                params = self.abrupt_params if drift_type == "Abrupt" else self.gradual_params
                
                reset_info = {
                    "step": self._current_step,
                    "type": "Full" if drift_type == "Abrupt" else "Partial",
                    "shrink_factor": params.get('shrink_factor')
                }
                # Log the dictionary itself
                self.reset_log.append(reset_info)
                print(f"\n>>> RESET CONTROLLER: Drift detected at step {self._current_step}!")
                print(f">>> Classified as: {drift_type.upper()}, performing {reset_info['type']} reset.")
                
                self.meta_algorithm.reset(
                    full=(drift_type == "Abrupt"), 
                    shrink_factor=params['shrink_factor']
                )
                
                self._detector.reset()
                self._last_reset_step = self._current_step
