from IPython import parallel
from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


# Create a function which will do a 1-dimensional grid search in parallel.
# Think about how you will manage the data (i.e. try to send the data to
# the nodes just once, rather than over and over)

# Use this grid search to create and plot a curve showing accuracy vs.
# alpha (log-spaced alphas are probably best). What is the best alpha
# for this problem?

# Apply this code to the SGDClassifier using penalty="elasticnet"
# This is a much slower classifier, so the parallel computation will
# be very important!



def grid_search(alphas):
    results = []
    for alpha in alphas:
        clf = MultinomialNB(alpha)
        results.append(np.mean(cross_val_score(clf, data.data, data.target)))
    return results


def create_plot_curve():
    data = fetch_20newsgroups_vectorized(remove=('headers', 'footers', 'quotes'))
    clients = parallel.Client()
    lview = clients.load_balanced_view()
    lview.block = True
    alphas = [1E-4, 1E-3, 1E-2, 1E-1]
    lview.apply(data)

    res = lview.map(grid_search(alphas), range(10))
    return res

if __name__ == '__main__':
    results = create_plot_curve()
    best_result = (0, 0)
    for result in results:
        if result[1] > best_result[1]:
            best_result = result
    print "\nThe best result is:"
    print "alpha = {}".format(best_result[0])
    print "score = {}%".format(round(best_result[1] * 100, 2))
