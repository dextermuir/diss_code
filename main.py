# main
from experiments import bs_convergence_error_plots, compute_time_plots
N_values = [160, 320, 640, 1280, 2560, 5120]

bs_convergence_error_plots(
    S=100,
    E=100,
    r=0.05,
    sigma=0.1,
    T=1,
    N_values=N_values,
    save_as="numerical_bs_convergence"
)


compute_time_plots(
    S=100,
    E=100,
    r=0.05,
    sigma=0.1,
    T=1,
    N_values=N_values,
    save_as="numerical_time_divergence"
)
