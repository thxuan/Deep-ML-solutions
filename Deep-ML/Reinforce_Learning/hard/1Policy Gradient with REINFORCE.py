import numpy as np
def softmax(x):
    x = x - np.max(x,axis=-1,keepdims=True)
    return np.exp(x) / (np.sum( np.exp(x), axis=-1, keepdims=True ) + 1e-9)

def compute_policy_gradient(theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    # Your code here
    rein_gradient = np.zeros(theta.shape)
    e_gradient = []    
    for i in range(len(episodes)):
       #episodes[i] = np.array(episodes[i])
        a = np.sum(np.array(episodes[i]),axis=0)
        gt = a[2]
        for episode in episodes[i]:
            scores = softmax(theta[episode[0]])
            lp_gradient = np.array( [ 1-scores[j] if j == episode[1] else  - scores[j] for j in range(theta.shape[-1]) ] )
            lp_gradient *= gt 
            gt -= episode[2]
            e_gradient.append(np.round(lp_gradient,4))
        rein_gradient += e_gradient
        e_gradient = []
    return rein_gradient/len(episodes)

theta = np.zeros((2,2))
episodes = [[(0,1,0), (1,0,1)], [(0,0,0)]]
print(compute_policy_gradient(theta,episodes))