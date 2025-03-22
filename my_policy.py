# import numpy as np

# def policy_action(policy, observation):
#     """
#     Given a numeric policy array (shape [36]) and an observation (shape [8]),
#     compute the action by applying a linear mapping:
#       W = policy[:32].reshape(8,4)
#       b = policy[32:].reshape(4)
#       logits = observation @ W + b
#       action = argmax(logits)
#     """
#     W = policy[:32].reshape(8, 4)
#     b = policy[32:].reshape(4)
#     logits = np.dot(observation, W) + b
#     return int(np.argmax(logits))


import numpy as np

def policy_action(policy, observation):
    """
    Given a numeric policy array (shape [36]) and an observation (shape [8]),
    compute the action by applying a linear mapping:
      W = policy[:32].reshape(8,4)
      b = policy[32:].reshape(4)
      logits = observation @ W + b
      action = argmax(logits)
    """
    W = policy[:32].reshape(8, 4)
    b = policy[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return int(np.argmax(logits))
