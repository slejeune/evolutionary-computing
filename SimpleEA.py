import matplotlib.pyplot as plt
import random

f = open('file-tsp.txt', 'r')
tsp = f.read()

# Number of cities
n = 50

# Coordinates parsed from the txt file
c = [[float(y) for y in x.split(' ') if y] for x in tsp.split('\n')]

# Tour ordering (random initialization)
t = [x for x in range(n)]
random.shuffle(t)

# Fitness = 1/(total distance travelled)
def fitness(cities, tour):

    total_distance = 0
    sorted_c = [cities[i] for i in tour]

    for i in range(len(sorted_c)):
        if i == len(sorted_c)-1:
            break
        cord = cities[tour[i]]
        next_cord = cities[tour[i+1]]
        total_distance += ((cord[0] - next_cord[0])**2 + (cord[1] - next_cord[1])**2)**0.5
        
    return 1/total_distance

plt.plot([x[0] for x in [c[i] for i in t]], 
         [x[1] for x in [c[i] for i in t]], 
         color='black',
         linewidth=1,
         alpha=0.75)
plt.scatter([x[0] for x in c],[x[1] for x in c], color='firebrick', s=100)
plt.show()