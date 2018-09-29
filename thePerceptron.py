import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])
weights = np.array([0.0, 0.0])

LEARNING_RATE = 0.1


def sumWeights(x, w):
    return x.dot(w) # Dot product


def activationFunction(sum):
    return 1 if sum >= 1 else 0


def error(target, obtained):
    return target - obtained


def newWeight(weight, input, error, learnigrate = 0.1):
    return weight + (learnigrate * input * error)


def runOutput(input):
    return activationFunction(input.dot(weights))


def train():

    totalError = 1
    while (totalError != 0):
        totalError = 0

        for i in range(len(outputs)):
            obtainedOutput = runOutput(np.asarray(inputs[i]))
            err = error(outputs[i], obtainedOutput)
            totalError += err

            for j in range(len(weights)):
                weights[j] = newWeight(weight=weights[j],
                                       input=inputs[i][j],
                                       error=err,
                                       learnigrate=0.1)

                print('New Weight = ' + str(weights[j]))

            print('Total Error = ' + str(totalError))

train()

print(runOutput(inputs[0]))
print(runOutput(inputs[1]))
print(runOutput(inputs[2]))
print(runOutput(inputs[3]))
