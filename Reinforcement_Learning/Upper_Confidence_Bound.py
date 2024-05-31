#UCB Implementation to check Ad that was most selected

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#Importing dataset
dataset = pd.read_csv('Reinforcement_Learning/Ads_CTR_Optimisation.csv')

# print(dataset)

#Implementation of UCB
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# print(number_of_selections)
for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (number_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward


#Visualisation of results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


