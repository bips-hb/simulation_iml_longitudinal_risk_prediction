# Minimal example: DynamicWindowSHAP reports window values on time steps that are
# shifted by one relative to the windows it evaluates.
#
# Tested with: windowshap 1.0.1, shap 0.42.1, numpy 1.26.4, Python 3.11

import numpy as np
from windowshap.windowshap import DynamicWindowSHAP

T = 10          # time steps
t_star = 4      # the only time step the model uses (0-based)

# Model: the output is the value of the single feature at time step t_star.
# So the only time step that matters is t_star.
class Model:
    def predict(self, x):                 # x: (n, T, 1)
        return x[:, t_star, :]            # shape (n, 1)

background = np.zeros((1, T, 1))          # one background sequence: all zeros
x = np.ones((1, T, 1))                    # explained sequence: all ones

# Expected result: f(x) - f(background) = 1, and all of it belongs to the window
# that contains time step t_star.

explainer = DynamicWindowSHAP(Model(), delta=0.01, n_w=10, B_ts=background, test_ts=x)
phi = explainer.shap_values()             # shape (1, T, 1)

split_points = explainer.split_points[0]
print("split points:", split_points)

# Windows as they are EVALUATED (wraper_predict uses np.searchsorted)
window_of_step = np.searchsorted(split_points, np.arange(T))
print("\nwindow of each time step when evaluated:")
for j in range(len(split_points)):
    print(f"  window {j}: time steps {np.where(window_of_step == j)[0].tolist()}")

# Windows as they are REPORTED (start_ind / end_ind in shap_values)
print("\nwindow of each time step when reported:")
for j in range(len(split_points)):
    start_ind = 0 if j == 0 else split_points[j - 1]
    end_ind = split_points[j] + int((j + 1) / len(split_points))
    print(f"  window {j}: time steps {list(range(start_ind, end_ind))}")

print("\nreported Shapley value per time step:")
print(np.round(phi[0, :, 0], 3))
print(f"\nvalue at time step {t_star} (the only one the model uses): {phi[0, t_star, 0]:.3f}")
print(f"sum over all time steps: {phi.sum():.3f}  (expected: 1)")