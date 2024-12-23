import numpy as np

def forward_algorithm(A, B, pi, O):
    N = len(A)  # Number of states
    T = len(O)  # Length of observation sequence
    alpha = np.zeros((T, N))
    
    # Initialization step
    alpha[0, :] = pi * B[:, O[0]]
    
    # Recursion step
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, O[t]]
    
    return alpha

def backward_algorithm(A, B, O):
    N = len(A)  # Number of states
    T = len(O)  # Length of observation sequence
    beta = np.zeros((T, N))
    
    # Initialization step
    beta[T-1, :] = 1
    
    # Recursion step
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :])
    
    return beta

def baum_welch(O, N, M, max_iter=100):
    T = len(O)
    A = np.random.rand(N, N)
    B = np.random.rand(N, M)
    pi = np.random.rand(N)
    
    # Normalize to make valid probabilities
    A /= A.sum(axis=1, keepdims=True)
    B /= B.sum(axis=1, keepdims=True)
    pi /= pi.sum()
    
    for _ in range(max_iter):
        alpha = forward_algorithm(A, B, pi, O)
        beta = backward_algorithm(A, B, O)
        
        # Calculate gamma and xi
        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))
        
        for t in range(T-1):
            denom = np.sum(alpha[t, :] * beta[t, :])
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / denom
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O[t+1]] * beta[t+1, j]) / denom
        
        # Update A, B, pi
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0, keepdims=True)
        B_num = np.zeros_like(B)
        B_denom = np.sum(gamma, axis=0)
        
        for t in range(T):
            B_num[:, O[t]] += gamma[t, :]
        
        B = B_num / B_denom[:, None]
        pi = gamma[0, :]
    
    return A, B, pi

# Example usage
O = [0, 1, 0]  # Observation sequence (mapped to integers)
N = 2  # Number of states
M = 2  # Number of observation symbols

A, B, pi = baum_welch(O, N, M)
print("Updated Transition Probabilities (A):\n", A)
print("Updated Emission Probabilities (B):\n", B)
print("Updated Initial Probabilities (pi):\n", pi)
