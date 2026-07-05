# error and runtime analysis with graph generation
import time
import numpy as np
import matplotlib.pyplot as plt
from pricing_methods import black_scholes, binomial, finite_difference, monte_carlo

plt.rcParams.update({'font.size': 14})

def bs_convergence_error_plots(S, E, r, sigma, T, N_values, save_as=None):
    """
    Computes absolute error of various Numerical methods with Black–Scholes for a sequence of N values,
    generating raw convergence and log–log plots, with a linear regression line for monte carlo.
    """

    bs_price = black_scholes(S, E, r, sigma, T)
    bin_errors = []
    fin_diff_errors = []
    monte_errors = []

    for N in N_values:
        # binomial
        bin_price = binomial(S, E, r, sigma, T, N)
        bin_errors.append(abs(bin_price - bs_price))

        # finite differnce
        fin_diff_price = finite_difference(S, E, r, sigma, T, N)
        fin_diff_errors.append(abs(fin_diff_price - bs_price))

        # monte carlo
        monte_price = monte_carlo(S, E, r, sigma, T, N**2)
        monte_errors.append(abs(monte_price - bs_price))


    # plot errors vs N
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, bin_errors, marker='o', label = "Binomial")
    plt.plot(N_values, fin_diff_errors, marker='x', label = "Finite Difference")
    plt.plot(N_values, monte_errors, marker='+', label = "Monte Carlo")
    plt.xlabel("Time steps (n)/ Simulations (n^2)")
    plt.ylabel("Absolute Error (epsilon_n)")
    plt.title("Convergence of Numerical Methods to Black–Scholes Price")
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    if save_as is not None:
        plt.savefig(save_as + "_plot.png", dpi=300)

    plt.show()

    # log–log plot
    plt.figure(figsize=(8, 5))
    plt.loglog(N_values, bin_errors, marker='o', linestyle= 'None')
    plt.loglog(N_values, fin_diff_errors, marker='x', linestyle= 'None')
    plt.loglog(N_values, monte_errors, marker='+', linestyle= 'None')

    # binomial regression line
    x = np.log(N_values)
    y = np.log(bin_errors)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"Binomial Regression (slope={slope:.3f})")
    
    # fin diff regression line
    x = np.log(N_values)
    y = np.log(fin_diff_errors)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"EFDA Regression (slope={slope:.3f})")

    # monte regression line
    x = np.log(N_values)
    y = np.log(monte_errors)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"MCS Regression (slope={slope:.3f})")

    plt.xlabel("log(time steps (n)/ simulations (n^2))")
    plt.ylabel("log(Absolute Error (epsilon_n))")
    plt.title("Log–Log Convergence Plot Comparison of Numerical Methods")
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    if save_as is not None:
        plt.savefig(save_as + "_loglog.png", dpi=300)
   
    plt.show()

def compute_time_plots(S, E, r, sigma, T, N_values, save_as=None):
    """
    Computes time taken of various Numerical methods for a sequence of N values,
    and generates raw time and log-log plots.
    """

    bin_times = []
    fin_diff_times = []
    monte_times = []

    for N in N_values:
        # binomial
        t_0 = time.time()
        binomial(S, E, r, sigma, T, N)
        bin_compute_time = time.time() - t_0
        bin_times.append(bin_compute_time)

        # finite difference
        t_0 = time.time()
        finite_difference(S, E, r, sigma, T, N)
        fin_diff_compute_time = time.time() - t_0
        fin_diff_times.append(fin_diff_compute_time)

        # monte carlo
        t_0 = time.time()
        monte_carlo(S, E, r, sigma, T, N**2)
        monte_compute_time = time.time() - t_0
        monte_times.append(monte_compute_time)

        
    
    # plot error vs N
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, bin_times, marker='o', label = "Binomial")
    plt.plot(N_values, fin_diff_times, marker='x', label = "Finite Difference")
    plt.plot(N_values, monte_times, marker='+', label = "Monte Carlo")
    plt.xlabel("time steps (n)/ simulations (n^2)")
    plt.ylabel("Compute time /s")
    plt.title("Compute time of Numerical methods")
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    if save_as is not None:
        plt.savefig(save_as + "_plot.png", dpi=300)

    plt.show()

    # log–log plot
    plt.figure(figsize=(8, 5))
    plt.loglog(N_values, bin_times, marker='o', linestyle= 'None')
    plt.loglog(N_values, fin_diff_times, marker='x', linestyle= 'None')
    plt.loglog(N_values, monte_times, marker='+', linestyle= 'None')

    # binomial regression line
    x = np.log(N_values)
    y = np.log(bin_times)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"Binomial Regression (slope={slope:.3f})")
    
    # fin diff regression line
    x = np.log(N_values)
    y = np.log(fin_diff_times)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"EFDA Regression (slope={slope:.3f})")

    # monte regression line
    x = np.log(N_values)
    y = np.log(monte_times)

    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]
    fit = np.exp(intercept) * np.array(N_values)**slope
    plt.loglog(N_values, fit, '--', linewidth=2,
           label=f"MCS Regression (slope={slope:.3f})")


    plt.xlabel("log(time steps (n)/ simulations(n^2))")
    plt.ylabel("log(time /s)")
    plt.title("Log–Log Compute Time of Numerical methods")
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    if save_as is not None:
        plt.savefig(save_as + "_loglog.png", dpi=300)

    plt.show()

    