import numpy as np
import matplotlib.pyplot as plt

def average(l):
    return sum(l) / len(l)

def plot(id, scores):

    # average scores of all threads
   # scores = [average(scores) for scores in thread_scores]

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("Thread: %d" % id)
    plt.show()
