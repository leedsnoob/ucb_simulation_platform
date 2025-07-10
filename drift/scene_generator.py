import numpy as np
from .drift_injector import DriftInjector

def create_drift_scenario(handler, scenario_type: str):
    """
    Applies a pre-defined drift recipe to a dataset handler.

    Args:
        handler: An instance of a data handler (e.g., MovieLensDataHandler).
        scenario_type (str): The name of the drift scenario. 
                             Supported: 'ml_abrupt', 'ml_gradual', 'obd_abrupt', 'obd_gradual'.

    Returns:
        pd.DataFrame: The data with the specified drift injected.
    """
    print(f"\nCreating drift scenario: {scenario_type}")
    
    data = handler.data.copy()
    injector = DriftInjector(data)
    
    t_total = len(data)
    t1, t2 = t_total // 3, 2 * t_total // 3

    if scenario_type == 'ml_abrupt':
        # Select top 40% most popular arms
        num_arms_total = data['chosen_arm'].nunique()
        top_40_arms = data['chosen_arm'].value_counts().nlargest(int(num_arms_total * 0.4)).index.tolist()
        
        # Split them into two disjoint sets for two separate drifts
        abrupt_arms_1 = top_40_arms[:len(top_40_arms)//2]
        abrupt_arms_2 = top_40_arms[len(top_40_arms)//2:]

        injector.inject_abrupt_drift(t1, abrupt_arms_1)  # Drift 1: Top 20% arms
        injector.inject_abrupt_drift(t2, abrupt_arms_2)  # Drift 2: Next 20% arms
    
    elif scenario_type == 'ml_gradual':
        num_arms_to_drift = int(data['chosen_arm'].nunique() * 0.2)
        top_arms = data['chosen_arm'].value_counts().nlargest(num_arms_to_drift).index.tolist()
        injector.inject_gradual_drift(t1, t2, top_arms)

    elif scenario_type == 'obd_bts_abrupt':
        # Select top 20% most popular arms for two-stage drift
        num_arms_total = data['chosen_arm'].nunique()
        top_20_arms = data['chosen_arm'].value_counts().nlargest(int(num_arms_total * 0.2)).index.tolist()
        
        # Split them into two disjoint sets (top 10% and next 10%)
        abrupt_arms_1 = top_20_arms[:len(top_20_arms)//2]
        abrupt_arms_2 = top_20_arms[len(top_20_arms)//2:]

        injector.inject_abrupt_drift(t1, abrupt_arms_1)  # Drift 1: Top 10%
        injector.inject_abrupt_drift(t2, abrupt_arms_2)  # Drift 2: Next 10%

    elif scenario_type == 'obd_bts_gradual':
        num_arms_to_drift = int(data['chosen_arm'].nunique() * 0.1)
        top_arms = data['chosen_arm'].value_counts().nlargest(num_arms_to_drift).index.tolist()
        injector.inject_gradual_drift(t1, t2, top_arms)
        
    else:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")

    return injector.get_drifted_data()

# Helper functions to attach metadata to plots
def get_drift_points(scenario_name, data_len):
    if 'Abrupt' in scenario_name:
        t1, t2 = data_len // 3, 2 * data_len // 3
        return {t1: 'Abrupt Drift 1', t2: 'Abrupt Drift 2'}
    return None

def get_drift_window(scenario_name, data_len):
    if 'Gradual' in scenario_name:
        t1, t2 = data_len // 3, 2 * data_len // 3
        return (t1, t2)
    return None 