# black-scholes, binomial, finite difference, monte carlo
import numpy as np, scipy, random


def black_scholes(S, E, r, sigma, T, option_type = "call"):
    """
    Computes Black–Scholes value of a European Vanilla option for the given parameters.
    """

    d_1 = (np.log(S/E) + T*(r + 0.5*(sigma**2)))/(sigma * (T**0.5))
    d_2 = (np.log(S/E) + T*(r - 0.5*(sigma**2)))/(sigma * (T**0.5))
    norm_d_1 = scipy.stats.norm.cdf(d_1)
    norm_d_2 = scipy.stats.norm.cdf(d_2)


    V = S * norm_d_1 - E * np.exp(-r * T) * norm_d_2
    if option_type == "call":
        return V
    else:
        V = V - S + E * np.exp(-r * T)
        return V


def binomial(S, E, r, sigma, T, N, option_type = "call"):
    """
    Computes CRR binomial tree value of a European Vanilla option for the given parameters.
    """
    N = int(N)

    # u, d and p'
    u = np.exp(sigma * np.sqrt(T/N))
    d = np.exp(-sigma * np.sqrt(T/N))
    p_dash = (np.exp(r * T / N) - d) / (u - d)

    # stock prices at expiry
    S_T = np.array([(u**(N-n)) * (d**n) * S for n in range(N+1)])

    # option prices at expiryif option_type == "call":
    if option_type == "call":
        V_prev = np.array([np.maximum(S_T[n]-E, 0) for n in range(N+1)])
    else:
        V_prev = np.array([np.maximum(E-S_T[n], 0) for n in range(N+1)])
    
    # extra arrays
    V_new = np.zeros(N+1)
    
    # back recurs
    for n in range(N):
        for m in range(N-n):
            V_new[m] = np.exp(-r*T/N) * (p_dash*V_prev[m] + (1-p_dash)*V_prev[m+1])
        
        V_prev = V_new.copy()
    
    # value at t
    return V_new[0]

def binomial_american(S, E, r, sigma, T, N, option_type = "call"):
    """
    Computes CRR Binomial Tree value of an American call for the given parameters.
    """
    N = int(N)

    # u, d and p'
    u = np.exp(sigma * np.sqrt(T/N))
    d = np.exp(-sigma * np.sqrt(T/N))
    p_dash = (np.exp(r * T / N) - d) / (u - d)

    # stock prices at expiry
    S_T = np.array([(u**(N-n)) * (d**n) * S for n in range(N+1)])

    # option prices at expiry
    if option_type == "call":
        V_prev = np.array([np.maximum(S_T[n]-E, 0) for n in range(N+1)])
    else:
        V_prev = np.array([np.maximum(E-S_T[n], 0) for n in range(N+1)])
    
    # extra arrays
    V_new = np.zeros(N)
    
    # back recurs
    for n in range(N):
        for m in range(N-n):
            if option_type == "call":
                V_new[m] = np.maximum(np.exp(-r*T/N) * (p_dash*V_prev[m] + (1-p_dash)*V_prev[m+1]), (S * (u**((N-n-1)-m))*(d**m) - E))
            else:
                V_new[m] = np.maximum(np.exp(-r*T/N) * (p_dash*V_prev[m] + (1-p_dash)*V_prev[m+1]), (E - S * (u**((N-n-1)-m))*(d**m)))

        
        V_prev = V_new.copy()
    
    # value at t
    return V_new[0]

def finite_difference(S, E, r, sigma, T, N, option_type = "call"):
    """
    Computes Explicit Finite Difference Approximation of a European call for the given parameters.
    """

    y = np.log(S)

    # choose h, k
    k = T/N
    k_bound = sigma**2/(r - 0.5*sigma**2)**2
    h = 2 * sigma * np.sqrt(k)
    h_bound = sigma**2/np.abs(r - 0.5*sigma**2)

    if k > k_bound:
        print("ERROR: k non-negativity condition breached. k = " + str(k) + " is more than " + str(k_bound) + ".")
        return -1
    
    if h > h_bound:
        print("ERROR: h non-negativity condition breached. h = " + str(h) + " is more than " + str(h_bound) + ".")
        return -1
    
    # p-, p, p+
    p_minus = 0.5*(k/h) * ((sigma**2/h) - (r - (sigma**2/2))) 
    p = 1 - k*(sigma/h)**2 
    p_plus = 0.5*(k/h) * ((sigma**2/h) + (r - (sigma**2/2))) 

    # calculate potential values of the log of the stock price at expiry
    y_T = np.array([y + (n - N)* h for n in range(2*N+1)])

    # calculate value log value of the option
    if option_type == "call":
        V_T = np.array([np.maximum(np.exp(y_T[n])-E, 0) for n in range(2*N+1)])
    else:
        V_T = np.array([np.maximum(E-np.exp(y_T[n]), 0) for n in range(2*N+1)])
    
    V_new = np.zeros(2*N-1)
    V_prev = V_T.copy()


    # back recurs
    for n in range(N):
        for m in range(2*(N-n)-1):
            V_new[m] = (1/(1 + r*k)) * (p_minus*V_prev[m] + p*V_prev[m+1] + p_plus*V_prev[m+2]) 
        
        V_prev = V_new.copy()
    
    return V_new[0]

def finite_difference_american(S, E, r, sigma, T, N, option_type = "call"):
    """
    Computes Explicit Finite Difference Approximation of an American call for the given parameters.
    """

    y = np.log(S)

    # choose h, k
    k = T/N
    k_bound = sigma**2/(r - 0.5*sigma**2)**2
    h = 2 * sigma * np.sqrt(k)
    h_bound = sigma**2/np.abs(r - 0.5*sigma**2)

    if k > k_bound:
        print("ERROR: k non-negativity condition breached. k = " + str(k) + " is more than " + str(k_bound) + ".")
        return -1
    
    if h > h_bound:
        print("ERROR: h non-negativity condition breached. h = " + str(h) + " is more than " + str(h_bound) + ".")
        return -1
    
    # p-, p, p+
    p_minus = 0.5*(k/h) * ((sigma**2/h) - (r - (sigma**2/2))) 
    p = 1 - k*(sigma/h)**2 
    p_plus = 0.5*(k/h) * ((sigma**2/h) + (r - (sigma**2/2))) 

    # calculate potential values of the log of the stock price at expiry
    y_T = np.array([y + (n - N)* h for n in range(2*N+1)])

    # calculate value log value of the option
    if option_type == "call":
        V_T = np.array([np.maximum(np.exp(y_T[n])-E, 0) for n in range(2*N+1)])
    else:
        V_T = np.array([np.maximum(E-np.exp(y_T[n]), 0) for n in range(2*N+1)])

    V_new = np.zeros(2*N-1)
    V_prev = V_T.copy()


    # back recurs
    for n in range(N):
        for m in range(2*(N-n)-1):
            if option_type == "call":
                V_new[m] = np.maximum((1/(1 + r*k)) * (p_minus*V_prev[m] + p*V_prev[m+1] + p_plus*V_prev[m+2]), np.exp(y + (m - (N-1-n))*h) - E)  
            else:
                V_new[m] = np.maximum((1/(1 + r*k)) * (p_minus*V_prev[m] + p*V_prev[m+1] + p_plus*V_prev[m+2]), E-np.exp(y + (m - (N-1-n))*h))
        
        V_prev = V_new.copy()
    
    return V_new[0]

def monte_carlo(S, E, r, sigma, T, N, option_type = "call"):
    """
    Computes Monte Carlo value of a European Call option for the given parameters.
    """

    N = int(N)

    # generate N random variables
    x_tilde = np.random.randn(N)

    # calculate S at expiry from random variables
    S_T = S*np.e**((r - 0.5*sigma**2)*T + sigma * np.sqrt(T) * x_tilde)

    # calculate the discounted expected value of all the generated values
    if option_type == "call":
        C_t_estimate = np.exp(-r*T) * np.mean(np.maximum(S_T - E, 0))
    else:
        C_t_estimate = np.exp(-r*T) * np.mean(np.maximum(E - S_T, 0))

    return C_t_estimate

def monte_carlo_asian(S, E, r, sigma, T, N, time_steps, option_type = "call"):
    """
    Computes Monte Carlo value of a European Call option for the given parameters.
    """

    N = int(N)
    delta_t = T/time_steps
    S_t_next = S
    path_sum = 0

    for step in range(time_steps):
        # generate N random variables
        x_tilde = np.random.randn(N)

        # calculate S at expiry from random variables
        S_t_next = S_t_next*np.exp((r - 0.5*sigma**2)*delta_t + sigma * np.sqrt(delta_t) * x_tilde)

        path_sum += S_t_next

    S_average = path_sum/time_steps

    # calculate the discounted expected value of all the generated values
    if option_type == "call":
        C_t_estimate = np.exp(-r*T) * np.mean(np.maximum(S_average - E, 0))
    else:
        C_t_estimate = np.exp(-r*T) * np.mean(np.maximum(E - S_average, 0))

    return C_t_estimate


# test params 100, 100, 0.05, 0.1, 1, 100