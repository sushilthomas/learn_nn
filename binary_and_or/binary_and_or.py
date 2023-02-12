import numpy as np

inputs = np.array([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ])

outputsOR = np.array([
          0,
          1,
          1,
          1,
        ])

outputsAND = np.array([
          0,
          0,
          0,
          1,
        ])

def activate1(sum):
  if sum >= 1:
    return 1
  return 0

activate = np.vectorize(activate1)

class Context:
  def __init__(self):
    self.learningRate = 0.1

def train(context, outputTruths):
  weights = [0.0, 0.0]

  for i in range(0, 100):
    sums = inputs.dot(weights)
    outputs = activate(sums)
    errors = outputs - outputTruths
    error = np.sum(np.abs(errors))
    if error == 0:
      return weights

    for i in range(0, len(errors)):
      if errors[i] != 0:
        for j in range(0, len(weights)):
          weights[j] = weights[j] - (context.learningRate * errors[i] * inputs[i][j])

  return weights

weightsOR = train(Context(), outputsOR)
weightsAND = train(Context(), outputsAND)

print('OR', weightsOR)
print('AND', weightsAND)

