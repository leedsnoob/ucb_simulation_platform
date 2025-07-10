
from river import drift

class DriftDetector:
    """
    A wrapper class for the ADWIN drift detector from the river library.
    """
    def __init__(self, delta: float = 0.002):
        """
        Initializes the ADWIN detector.

        Args:
            delta (float): The confidence value for the ADWIN algorithm. A small
                         delta results in a more sensitive detector.
        """
        self.adwin = drift.ADWIN(delta=delta)
        self._n_samples = 0

    def update(self, reward: float) -> None:
        """
        Updates the detector with a new reward value from the stream.
        
        Args:
            reward (float): The reward value (or any other metric to monitor).
        """
        self.adwin.update(reward)
        self._n_samples += 1

    @property
    def drift_detected(self) -> bool:
        """
        Returns True if a drift was detected in the last update.

        This is a one-time flag. The ADWIN algorithm internally resets its state
        (and this flag) after a drift is detected, making it ready to detect
        the next change.
        """
        return self.adwin.drift_detected

    def reset(self) -> None:
        """
        Manually resets the detector to its initial state.
        """
        self.adwin._reset()
        self._n_samples = 0

