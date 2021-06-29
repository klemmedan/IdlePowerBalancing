import matplotlib.pyplot as plt
import numpy as np
import math

increase_rate = 1.015110095975038
start_time = 300
end_time = 2*365*24*3600  # Two years in seconds.


def add_randomness(per_time_list):
    return [t + (np.random.randn() - 0.5)*t/6 for t in per_time_list]


# This function is meant to reduce the time taken to get to early upgrades so that the game progresses faster earlier
# It also needs to make sure the first upgrade is not achieved *too* fast since we want that to be a little ways into
# the gameplay.
def massage_early_upgrade_times(times):
    new_list = [t / 5 if i < 25 else t for i, t in enumerate(times)]
    new_list = [t / 3 if 40 > i >= 25 else t for i, t in enumerate(new_list)]
    new_list = [t / 1.5 if 80 > i >= 40 else t for i, t in enumerate(new_list)]
    new_list[0] = 1200  # Take 20 minutes before getting to first upgrade.
    return new_list


def add_quick_rewards(cumulative_time_list):
    out_list = []
    for i, x in enumerate(cumulative_time_list):
        if i in [200, 201, 202, 203, 204]:
            out_list.append(out_list[-1] + 240)
        elif i in [300, 301, 302, 303, 304]:
            out_list.append(out_list[-1] + 240)
        elif i in [400, 401, 402, 403, 404]:
            out_list.append(out_list[-1] + 240)
        elif i in [500, 501, 502, 503, 504]:
            out_list.append(out_list[-1] + 240)
        elif i in [600, 601, 602, 603, 604]:
            out_list.append(out_list[-1] + 240)
        else:
            out_list.append(x)
    return out_list


class GenerateUpgradeTimes:
    def __init__(self, num_upgrades):
        self.start = 3600
        self.end = 2*365*24*3600  # Two years in seconds.
        self.num_upgrades = num_upgrades
        self.increase_rate = 1.1

    def calculate_upgrade_times(self):
        return_times = []
        self.increase_rate = CalculateIncreaseRate(self.start, self.end, self.num_upgrades).calculate_increase_rate()
        for x in range(0, self.num_upgrades):
            return_times.append(self.calculate_single_upgrade_time(x))
        return return_times

    def calculate_single_upgrade_time(self, upgrade_index):
        upgrade_time = self.calculate_exponent_time(upgrade_index)
        upgrade_time = add_randomness(upgrade_time)
        return upgrade_time

    def calculate_exponent_time(self, upgrade_index):
        return self.start * self.increase_rate ** upgrade_index

    def calculate_parabolic_times(self, b=0):
        c = self.start
        a = (self.end - c)/self.num_upgrades**2 - b/self.num_upgrades
        parabolic_list = [a*x**2 + b*x + c for x in range(self.num_upgrades)]
        return np.diff(parabolic_list)

    def calculate_cumulative_parabolic_times(self, b=0):
        c = self.start
        a = (self.end - c)/self.num_upgrades**2 - b/self.num_upgrades
        parabolic_list = [a*x**2 + b*x + c for x in range(self.num_upgrades)]
        return parabolic_list


class CalculateIncreaseRate:
    def __init__(self, start, end, n):
        self.start = start
        self.end = end
        self.num_upgrades = n
        self.min_bound = 1.0
        self.max_bound = 1.1

    def calculate_increase_rate(self):
        guess = self.max_bound
        sum = self.geometric_formula(guess)
        while not is_close(sum, self.end):
            if sum > self.end:
                self.max_bound = guess
            else:
                self.min_bound = guess
            guess = 0.5*(self.max_bound + self.min_bound)
            sum = self.geometric_formula(guess)
        return guess

    def geometric_formula(self, r):
        return self.start*(1 - r**(self.num_upgrades))/(1-r)


def is_close(a, b, rel_tol=1e-12, abs_tol=0.00000000001):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


if __name__ == "__main__":
    num_upgrades = 681
    time_maker = GenerateUpgradeTimes(num_upgrades)
    times = time_maker.calculate_parabolic_times(b=300)
    times = add_randomness(times)
    # plt.plot(np.cumsum([x/(3600*24) for x in times]))
    plt.plot([x/3600 for x in times])
    # plt.plot([x/(3600) for x in times])
    plt.axes()
    plt.grid(True)
    plt.show()
