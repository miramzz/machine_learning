from IPython import parallel
from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np



# Create a function which will do a 1-dimensional grid search in parallel.
# Think about how you will manage the data (i.e. try to send the data to
# the nodes just once, rather than over and over)

# Use this grid search to create and plot a curve showing accuracy vs.
# alpha (log-spaced alphas are probably best). What is the best alpha
# for this problem?

# Apply this code to the SGDClassifier using penalty="elasticnet"
# This is a much slower classifier, so the parallel computation will
# be very important!

def grid_search(alpha):
    clf = MultinomialNB(alpha)
    results = numpy.mean(cross_val_score(clf, data.data, data.target))
    return results



def create_plot_curve():
    clients = parallel.Client()
    lview = clients.load_balanced_view()
    dview = clients[:]
    dview['data']= fetch_20newsgroups_vectorized(remove=('headers', 'footers', 'quotes'))
    lview.block = True
    alphas = [1E-4, 1E-3, 1E-2, 1E-1]

    with dview.sync_imports():
        import numpy
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.cross_validation import cross_val_score

    res = lview.map(grid_search, alphas)
    return res


if __name__ == '__main__':
    results = create_plot_curve()
    best_ = 0
    for res in results:
        if res > best_:
            best_ = res
    print "best result", best_


