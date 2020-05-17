# ------------------------------------------------------------------------------
# Module for easy sampling.
# ------------------------------------------------------------------------------

from scipy.stats import norm
from scipy.stats import uniform
import seaborn as sns

def sample_normal(mean, std, min_value=0, max_value=1, nr_samples=100):
    data = norm.rvs(size=nr_samples, loc=mean, scale=std)
    data = [x if x>=min_value else min_value for x in data]
    data = [x if x<=max_value else max_value for x in data]
    return data

def sample_uniform(min_value=1, max_value=1, nr_samples=200):
    data = uniform.rvs(size=nr_samples, loc=min_value, scale=max_value)
    return data

def plot_samples(data):
    ax = sns.distplot(data,
                    bins=100,
                    kde=True,
                    color='skyblue',
                    hist_kws={"linewidth": 15,'alpha':1})
    ax.set(xlabel='Distribution', ylabel='Frequency')
