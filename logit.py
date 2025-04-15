import numpy as np
def log_odds(p):
    # Ensure p is between 0 and 1 (exclusive)
    p=np.clip(p, 1e-15, 1-1e-15) #Avoid division by zero or log(0)
    return np.log(p/(1-p))

#Example usage
probabilities=np.array([0.1, 0.5, 0.9])
log_odds_values=log_odds(probabilities)
print(log_odds_values)