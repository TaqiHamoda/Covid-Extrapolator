# some general imports
from time import perf_counter
import datetime
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # for plotting dates


# loads the data from the provided csv file
# the data is stored in data, which will be
# a numpy array - the first column is the number of recovered individuals
#              - the second column is the number of infected individuals
# the data starts on '2020/03/05' (March 5th) and is daily.
data = []
with open("ONdata.csv", "r") as f:
    f.readline()
    # print(l.split(','))  # skip the column names from the csv
    l = f.readline()
    while '2020/03/05' not in l:
        l = f.readline()
    e = l.split(',')
    data.append([float(e[5]) - float(e[12]), float(e[12])])
    for l in f:
        e = l.split(',')
        data.append([float(e[5]) - float(e[12]), float(e[12])])
data = np.array(data)

# the 3 main parameters for the model, we'll use them as
# global variables, so you can refer to them anywhere
beta0 = 0.32545622
gamma = 0.09828734
w = 0.75895019

# simulation basic scenario (see the simulation function later)
base_ends = [28, 35, 49, 70, 94, 132]
base_beta_factors = [1, 0.57939203, 0.46448341, 0.23328388, 0.30647815, 0.19737586]

# we'll also use beta as a global variable
beta = beta0

# We are assuming no birth or death rate, so N is a constant
N = 15e6

# assumed initial conditions
E = 0  # assumed to be zero
I = 18  # initial infected, based on data
S = N - E - I  # everyone else is susceptible
R = 0
x0 = np.array([S, E, I, R])


# The right hand side function for the SEIR model ODE, x'(t) = F(x(t))
# Feel free to use this when implementing the 3 methods, but you don't have to.
def F(x):
    return np.array([-x[0]*(beta/N * x[2]),
                     (beta/N * x[0]*x[2] - w*x[1]),
                     (w*x[1] - gamma*x[2]),
                     x[2]*gamma])


# your three numerical methods for performing a single time step
def method_I(x, h):
    s = x[0] - h*(beta/N)*x[0]*x[2]
    e = x[1] + h*(beta/N)*x[0]*x[2] - w*h*x[1]
    i = x[2] + w*h*x[1] - h*gamma*x[2]
    r = x[3] + h*gamma*x[2]

    return np.array([s, e, i, r])


def _method_II(x, previous_x, h):
    s = previous_x[0] - h*(beta/N)*x[0]*x[2]
    e = previous_x[1] + h*(beta/N)*x[0]*x[2] - w*h*x[1]
    i = previous_x[2] + w*h*x[1] - h*gamma*x[2]
    r = previous_x[3] + h*gamma*x[2]

    return np.array([s, e, i, r])


def jacobian_II(x, previous_x, h):
    jac = [[h*(beta/N)*x[2], 0, h*(beta/N)*x[0], 0],
           [-h*(beta/N)*x[2], w*h, -h*(beta/N)*x[0], 0],
           [0, -w*h, h*gamma, 0],
           [0, 0, -h*gamma, 0]]

    return np.array(jac)


def method_II(x, h):
    func = lambda x_1, x_0, h_0: x_1 - _method_II(x_1, x_0, h_0)
    jacobian = lambda x_1, x_0, h_0: jacobian_II(x_1, x_0, h_0) + np.identity(4)

    return fsolve(func, x, args=(x, h), fprime=jacobian)


def _method_III(x, previous_x, h):
    return (method_I(previous_x, h) + _method_II(x, previous_x, h))/2


def jacobian_III(x, previous_x, h):
    return jacobian_II(x, previous_x, h)/2


def method_III(x, h):
    func = lambda x_1, x_0, h_0: x_1 - _method_III(x_1, x_0, h_0)
    jacobian = lambda x_1, x_0, h_0: jacobian_III(x_1, x_0, h_0) + np.identity(4)

    return fsolve(func, x, args=(x, h), fprime=jacobian)


METHODS = {'I': method_I, 'II': method_II, 'III': method_III}


# take a step of length 1 using n smaller steps
# used in simulation (see below)
def step(x, n, method):
    # simulate step_length time unit using n small uniform length steps
    # and return the new state
    for i in range(n):
        x = method(x, 1/n)
    return x


def ode_solver(x, start, end):
    from scipy.integrate import solve_ivp as ode
    fun = lambda t, z: F(z)
    sol = ode(fun, [start, end], x, t_eval=range(start, end+1),
              method='LSODA', rtol=1e-8, atol=1e-5)
    solution = []
    for y in sol.y.T[1:, :]:
        solution.append(y.T)
    return solution


# The main simulation code:
# The simulation starts at time 0 and goes up to time ends[-1],
# the state x(t) is returned at each time 0,1,...,ends[-1].
# Inputs:
# x = initial conditions
# n = integer number of steps of method used to advance one time step
# method = 1,2, or 3 to specify which method to use. If None,
#          the builtin ODE solver is used.
# ends = list of times to break the simulate
# beta_factors = list of factors to multiply beta0 by to
#                obtain beta. E.g. on the first segment of the simulation
#                from time t=0 up to ends[0], beta = beta0 * beta_factors[0].
def simulation(x=x0, n=1, method=None, ends=base_ends, beta_factors=base_beta_factors):
    cur_time = 0
    xs = [x]

    for i, end in enumerate(ends):
        global beta
        beta = beta0 * beta_factors[i]

        if method is None:
            xs.extend(ode_solver(xs[-1], cur_time, end))
            cur_time = end
        else:
            while cur_time < end:
                xs.append(step(xs[-1], n, METHODS[method]))
                cur_time += 1

    return np.array(xs)


# some helper code to plot simulation trajectories,
# feel free to modify as needed.
def plot_trajectories(xs=data, sty='--k', label="data"):
    start_date = datetime.datetime.strptime("2020-03-05", "%Y-%m-%d")
    dates = [start_date]
    while len(dates) < len(xs):
        dates.append(dates[-1] + datetime.timedelta(1))

    # code to get matplotlib to display dates on the x-axis
    ax = plt.gca()
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)

    plt.plot(dates, xs, sty, linewidth=1, label=label)
    plt.xticks(rotation=45)


# example code to plot trajectories of I and R for each method
def plot_all_methods():
    plot_trajectories(data[:, 0], "--k", "Data")
    plot_trajectories(data[:, 1], "--k", "")

    xs = simulation()[:, 2:]
    plot_trajectories(xs[:, 0], "--m", "ODE")
    plot_trajectories(xs[:, 1], "--m", "")

    xs = simulation(n=10, method='I')[:, 2:]
    plot_trajectories(xs[:, 0], "--b", "Method I")
    plot_trajectories(xs[:, 1], "--b", "")

    xs = simulation(n=10, method="II")[:, 2:]
    plot_trajectories(xs[:, 0], "--r", "Method II")
    plot_trajectories(xs[:, 1], "--r", "")

    xs = simulation(n=10, method="III")[:, 2:]
    plot_trajectories(xs[:, 0], "--g", "Method III")
    plot_trajectories(xs[:, 1], "--g", "")

    plt.title("Ontario Covid-19 Data")
    plt.ylabel("# of People")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()


##########
# Hypothetical experiment where mask-use decreases after june 2020.
# This only models the data until october 2020
##########
def extrapolate():
    
    # base_ends = [28, 35, 49, 70, 94, 212]
    # base_beta_factors = [1, 0.57939203, 0.46448341, 0.23328388, 0.30647815, 0.19737586]

    base_ends = [28, 35, 49, 70, 94, 132, 212]
    base_beta_factors = [1, 0.57939203, 0.46448341, 0.23328388, 0.30647815, 0.19737586, 0.29606379]

    xs = simulation(ends=base_ends, beta_factors=base_beta_factors)[:, 2:]
    plot_trajectories(xs[:, 1], "--b", "Recovered")
    plot_trajectories(xs[:, 0], "--r", "Infected")

    plt.title("Ontario Covid-19 Data (Hypothetical)")
    plt.ylabel("# of People")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    extrapolate()
