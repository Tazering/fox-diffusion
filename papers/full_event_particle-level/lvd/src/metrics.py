from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import numpy as np
from scipy.stats import energy_distance

class Metrics:
    def __init__(s, u_val, v_val, u_weights = None, v_weights = None):
        s.u_val = u_val if not isinstance(u_val, list) else np.array(u_val)
        s.v_val = v_val if not isinstance(v_val, list) else np.array(v_val)
        s.u_weights = u_weights
        s.v_weights = v_weights
    
    # compute wasserstein distance
    def wasserstein(s):
        print(f"\nWasserstein Distance")
        return wasserstein_distance(s.u_val, s.v_val)

    # compute KL distance
    def kl(s):
        print(f"\nKL Divergence")

        return np.sum(s.u_val * np.log((s.u_val + 1e-6) / (s.v_val + 1e-6)) )
        # return sum(kl_div(s.u_val, s.v_val))

    
    # compute energy distance
    def energy(s):
        print(f"\nEnergy Distance")
        return energy_distance(s.u_val, s.v_val)

    # compute all the distances
    def compute_all_distances(s):
       print(s.wasserstein())
       print(s.kl())
       print(s.energy()) 


metrics_test = Metrics([0, 1, 2], [5, 6, 7])
metrics_test.compute_all_distances()

# print(wasserstein_distance([0, 1, 3], [5, 6, 8]))