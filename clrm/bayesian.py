import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import time
import logging
import random
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CLRM')


class Result(object):
    def __init__(self, file_name, model, trace):
        self.file_name = file_name
        self.model = model
        self.trace = trace
        self.mean_estimates = {}
        self.stddev_estimates = {}

        keys = trace[0].keys()
        for k in keys:
            self.mean_estimates[k] = np.mean([t[k] for t in trace])
            self.stddev_estimates[k] = np.std([t[k] for t in trace])


class CLRM(object):
    def __init__(self, file_name, variance_fn=None):
        """
        Classical Linear Regression Model
        which supports both homoskedastic and heteroskastic
        error variables.
        The variance function can be used to specify the relationship
        of \lambda parameter of the error distribution on the
        input variable x
        example
        ```
        CLRM(f, lambda w, x: w)
        or
        CLRM(f, lambda w, x: w/np.pow(x, 2))
        ```
        """
        self.file_name = file_name
        self.x, self.y = self.xy_op(file_name)
        self.model = None
        self.trace = None
        self.ppc = None
        self.result = None
        self.variance_fn = lambda w, x: w / np.abs(x) or variance_fn

    def generate_ppc(self, plot_ppc=False):
        """
        Generates the ppc for the given model and data
        """
        if self.trace is None or self.model is None:
            raise ValueError(
                'model must be initialized and trace must be generated')
        ppc = pm.sample_ppc(trace=self.trace, model=self.model)
        if plot_ppc:
            # plot the ppc and observed points
            ax = plt.subplot()
            x, y = self.x, self.y
            y_ppc = np.array([random.choice(e) for e in ppc['y_obs'].T])
            ax.scatter(x[:], y_ppc, s=10, c='g')
            ax.scatter(x[:], y, s=10, c='b')
        self.ppc = ppc
        return self.ppc

    def fit_linear_model(self, number_of_samples=5000):
        # load class variables
        file_name = self.file_name
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        """
        Student T distribution

        Parameters
        ----------
        nu : int
                Degrees of freedom (nu > 0).
        mu : float
                Location parameter.
        lam : float
                Scale parameter (lam > 0).

        """
        logger.info("Fitting linear model for filename {}".format(file_name))
        model = pm.Model()
        start_time = time.time()
        with model:
            # Define priors
            b = pm.Normal("b", mu=0., sd=100**2)
            a = pm.Normal("a", mu=0., sd=100**2)
            w = pm.Uniform("w", lower=0.0, upper=10.0)
            nu = pm.Uniform("nu", lower=1.0, upper=10.0)

            # Identity Link Function
            mu = b + a * x
            y_obs = pm.StudentT(
                "y_obs", mu=mu, lam=variance_fn(w, x), nu=nu, observed=y)

        with model:
            start = pm.find_MAP(model=model, fmin=scipy.optimize.fmin_powell)
            print('MAP Estimate', start)
            trace = pm.sample(
                number_of_samples, step=pm.Metropolis(), start=start)

        logger.info("Time taken = {}".format(time.time() - start_time))

        # Set class variables
        self.model = model
        self.trace = trace
        self.result = Result(file_name, model, trace)
        return self

    def plot_input_data(self):
        """
        Plots the input data
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(self.x[:], self.y[:], s=20, c='g')
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}]".format(self.file_name))
        ax.legend()

    def rmsd(self):
        """
        Calculates the RMSD for the given point estimates
        and training data
        """
        if self.result is None:
            raise ValueError('Linear Model must be fit to calculate RMSD')
        x, y = self.x, self.y
        ppc = self.ppc or self.generate_ppc()
        rmsd_ = 0
        y_ppc = ppc['y_obs'].T
        for y_obs, y_ppc_sample in zip(y, y_ppc):
            closest_y_delta = min([np.abs(e - y_obs) for e in y_ppc_sample])
            rmsd_ += np.power(closest_y_delta, 2)
        rmsd_ /= len(y_ppc)
        rmsd_ = np.power(rmsd_, 0.5)
        return rmsd_

    def plot_fit(self):
        """
        Plots the dataset with the fitted line
        """
        if self.result is None:
            raise ValueError(
                'Linear Model must be fit before plotting best fit')
        # load class objects
        file_name = self.file_name
        x, y = self.x, self.y
        result = self.result

        # best fit
        a_est = result.mean_estimates['a']
        b_est = result.mean_estimates['b']

        # other fits
        ax = plt.subplots(figsize=(10, 10))
        for item in result.trace[:50:]:
            a_p = item['a']
            b_p = item['b']
            ax.plot(x[:], (b_p + a_p * x)[:], 'g', alpha=1, lw=.01)
        ax.plot(x[:], (b_est + a_est * x)[:], 'y', alpha=1, lw=1)
        # observed
        ax.scatter(
            x[:],
            y[:],
            alpha=1,
            color='k',
            marker='.',
            s=50,
            label='original data')
        # labelling
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}] with linear fit".format(file_name))

    @staticmethod
    def xy_op(file_name):
        """
        Loads the x, y values in a dataframe from the file_name
        """
        logger.info("reading file {}".format(file_name))
        df = pd.read_csv(file_name)
        exog = df['x'].as_matrix()
        endo = df['y'].as_matrix()

        x = exog
        logger.info("shape of X is {}".format(x.shape))
        logger.debug("X is {}".format(x[:3]))

        y = endo
        logger.info("shape of Y is {}".format(y.shape))
        logger.debug("Y is {}".format(y[:3]))

        return (x, y)


if __name__ == '__main__':

    # Variable which has the file_name
    f = None
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        raise ValueError(
            'specify file_name as the first argument on commandline')

    clrm = CLRM(f, lambda w, x: w)
    clrm.fit_linear_model(10000)
    print(clrm.result.__dict__)
