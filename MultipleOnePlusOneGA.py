import numpy as np
import random
import matplotlib.pyplot as plt

# Length of initial bit sequence
l = 100
# Probability of switching bits
mu = 1/l
# Goal sequence
xg = [1 for i in range(l)]
# Generations
gen = 1500

for n in range(10):
    # Fitness score
    fitness = []

    # – Step 1: Randomly generate a bit sequence x
    x = [0 if rand<0.5 else 1 for rand in np.random.random_sample(size=l)]

    # – Step 4: Repeat steps 2 and 3 until the goal sequence is reached.
    for n in range(gen):

        # – Step 2: Create a copy of x and invert each bit with probability μ. Let xm be the result.
        xm = [0 if random.random() < mu and i==1 else  1 if random.random() < mu and i==0 else i for i in x]

        # – Step 3: If xm is closer to the goal sequence than x, replace x with xm.
        if sum(xm) > sum(x):
            x = xm

        fitness.append(sum(x))

    plt.plot(range(gen), fitness)

ax = plt.gca()
ax.set_ylim([30, l+5])
plt.axhline(y=l, color='gray', linestyle='--', alpha=0.5)
plt.title("Fitness over 1500 generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()