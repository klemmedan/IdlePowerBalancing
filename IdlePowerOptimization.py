import matplotlib.pyplot as plt
import numpy
import math
import copy
from typing import List, Optional
import DataLoader

purchase_factor = 1.05


class Optimizer:

    def __init__(self):
        self.stat_tracker = StatTracker()
        self.power_value = list()  # type: List[float]
        self.income = list()  # type: List[float]
        self.cumulativeProduction = list()  # type: List[float]
        self.cumulativeIncome = list()  # type: List[float]

        self.prod_resources = list()  # type: List[Resource]
        self.demand_resources = list()  # type: List[Resource]
        self.prodMultiUnlockFactors = list()  # type: List[float]
        self.prodMultiUnlockThresholds = list()  # type: List[float]
        self.demandMultiUnlockFactors = list()  # type: List[float]
        self.demandMultiUnlockThresholds = list()  # type: List[float]
        self.prodMultiUpgrades = UpgradeManager()
        self.demandMultiUpgrades = UpgradeManager()
        self.prod_upgrade_managers = list()  # type: List[UpgradeManager]
        self.demand_upgrade_managers = list()  # type: List[UpgradeManager]

    def run_optimization(self, num_steps):
        for i in range(num_steps):
            prod_income = self.total_prod_income()
            demand_income = self.total_demand_income()
            if prod_income < demand_income:
                self.improve_prod()
            else:
                self.improve_demand()

            self.stat_tracker.update_stats(self)

    def improve_prod(self):
        (count_index, count_score) = self.max_prod_resource_score()
        (upgrade_index, upgrade_score) = self.max_prod_upgrade_score()
        if upgrade_score >= count_score:
            self.stat_tracker.update_time(self.prod_upgrade_managers[upgrade_index].current_cost(), self)
            self.prod_upgrade_managers[upgrade_index].make_purchase()
        else:
            self.stat_tracker.update_time(self.prod_resources[count_index].cost(), self)
            self.prod_resources[count_index].make_purchase()

    def improve_demand(self):
        (count_index, count_score) = self.max_demand_resource_score()
        (upgrade_index, upgrade_score) = self.max_demand_upgrade_score()
        if upgrade_score >= count_score:
            self.stat_tracker.update_time(self.demand_upgrade_managers[upgrade_index].current_cost(), self)
            self.demand_upgrade_managers[upgrade_index].make_purchase()
        else:
            self.stat_tracker.update_time(self.demand_resources[count_index].cost(), self)
            self.demand_resources[count_index].make_purchase()

    def total_prod_income(self):
        total = 0
        index = 0
        for resource in self.prod_resources:
            total = total + resource.increment() * self.prod_upgrade_managers[index].current_factor()
            index += 1
        return total

    def total_demand_income(self):
        total = 0
        index = 0
        for resource in self.demand_resources:
            total = total + resource.increment() * self.demand_upgrade_managers[index].current_factor()
            index += 1
        return total

    def total_income(self):
        return min(self.total_demand_income(), self.total_prod_income())

    def max_prod_resource_score(self):
        index = 0
        max_score = 0
        max_index = index
        for resource in self.prod_resources:
            cost = resource.cost()
            current_increment_value = resource.increment()
            increment_value_if_purchased = resource.increment_if_purchased()
            score = (increment_value_if_purchased - current_increment_value) / cost

            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score

    def max_prod_upgrade_score(self):
        index = 0
        max_index = 0
        max_score = 0

        for upgrade_manager in self.prod_upgrade_managers:
            resource_increment = self.prod_resources[index].increment()
            cost = upgrade_manager.next_cost()
            current_factor = upgrade_manager.current_factor()
            next_factor = upgrade_manager.next_factor()
            score = resource_increment * (next_factor/current_factor - 1) / cost
            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score

    def max_demand_resource_score(self):
        index = 0
        max_score = 0
        max_index = index
        for resource in self.demand_resources:
            cost = resource.cost()
            current_increment_value = resource.increment()
            increment_value_if_purchased = resource.increment_if_purchased()
            score = (increment_value_if_purchased - current_increment_value) / cost

            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score

    def max_demand_upgrade_score(self):
        index = 0
        max_index = 0
        max_score = 0

        for upgrade_manager in self.demand_upgrade_managers:
            resource_increment = self.demand_resources[index].increment()
            cost = upgrade_manager.next_cost()
            current_factor = upgrade_manager.current_factor()
            next_factor = upgrade_manager.next_factor()
            score = resource_increment * (next_factor/current_factor - 1) / cost
            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score


class PowerValue:

    def __init__(self):
        self.amount_growth_rate = 1
        self.amount_second_growth_rate = 0.1
        self.count = 0
        self.base_cost = 50000
        self.cost_growth_rate = 1.2

    def factor(self):
        return 1 + self.count*self.amount_growth_rate + self.count*(self.count - 1)*self.amount_second_growth_rate/2

    def cost(self):
        n = self.num_to_purchase()
        total_cost = self.cost_growth_rate * (1 - self.cost_growth_rate ** (n + self.count - 1)) / (1 - self.cost_growth_rate)
        previous_cost = self.cost_growth_rate * (1 - self.cost_growth_rate ** (self.count - 1)) / (1 - self.cost_growth_rate)
        diff_cost = self.base_cost * (total_cost - previous_cost)
        return diff_cost

    def num_to_purchase(self):
        base_increase = math.floor(self.count*(purchase_factor - 1))
        return max(1, base_increase)

    def factor_if_purchased(self):
        bought_resource = copy.copy(self)
        bought_resource.count = bought_resource.count + self.num_to_purchase()
        return bought_resource.factor()

    def make_purchase(self):
        self.count = self.count + self.num_to_purchase()


class Resource:

    def __init__(self):
        self.amount = 0
        self.base_increment = 1
        self.base_cost = 10
        self.growth_rate = 1.1
        # Threshold to achieve an unlock. Start at 0 (corresponds to 1x factor, and go from there)
        self.unlock_thresholds = list()
        # Factor is cumulative and starts at 1 (i.e. 1, 2, 4, 12, 36, etc.)
        self.unlock_factors = list()

    def increment(self):
        unlock_index = self.unlock_index()
        unlock_factor = self.unlock_factors[unlock_index]
        return self.amount * self.base_increment * unlock_factor

    def cost(self):
        n = self.num_to_purchase()
        total_cost = self.growth_rate * (1 - self.growth_rate ** (n + self.amount-1)) / (1 - self.growth_rate)
        previous_cost = self.growth_rate * (1 - self.growth_rate ** (self.amount-1)) / (1 - self.growth_rate)
        diff_cost = self.base_cost * (total_cost - previous_cost)
        return diff_cost

    def unlock_index(self):
        index = -1
        for threshold in self.unlock_thresholds:
            if self.amount >= threshold:
                index = index + 1
            else:
                break
        return index

    def next_unlock_threshold(self):
        unlock_index = self.unlock_index()
        if unlock_index + 1 < len(self.unlock_thresholds):
            return self.unlock_thresholds[self.unlock_index() + 1]
        else:
            return 1e6

    def num_to_purchase(self):
        to_next_threshold = self.next_unlock_threshold() - self.amount
        base_increase = math.floor(self.amount*(purchase_factor - 1))
        return max(1, min(base_increase, to_next_threshold))

    def increment_if_purchased(self):
        bought_resource = copy.copy(self)
        bought_resource.amount = bought_resource.amount + self.num_to_purchase()
        return bought_resource.increment()

    def make_purchase(self):
        self.amount = self.amount + self.num_to_purchase()


class UpgradeManager:

    def __init__(self):
        self.current_index = 0

        # Cost to achieve an upgrade. Start at 0 and then go up (0 is for current/initial cost)
        self.upgrade_costs = list()

        # Cumulative factor, start at 1, then multiples all the way up.
        self.upgrade_factors = list()

    def current_cost(self):
        return self.upgrade_costs[self.current_index]

    def current_factor(self):
        return self.upgrade_factors[self.current_index]

    def next_cost(self):
        if self.current_index + 1 < len(self.upgrade_costs):
            return self.upgrade_costs[self.current_index + 1]
        else:
            return math.inf

    def next_factor(self):
        if self.current_index + 1 < len(self.upgrade_factors):
            return self.upgrade_factors[self.current_index + 1]
        else:
            return 0

    def make_purchase(self):
        self.current_index += 1


class StatTracker:

    def __init__(self):
        self.demand = list()  # type: List[float]
        self.production = list()  # type: List[float]
        self.income = list()
        self.delta_time = list()  # type: List[float]
        self.time = list()  # type: List[float]
        self.prod_counts = list()  # type: List[List[int]]
        self.demand_counts = list()  # type: List[List[int]]

    def update_stats(self, optimizer):
        # type: (Optimizer) -> None
        self.demand.append(optimizer.total_demand_income())
        self.production.append(optimizer.total_prod_income())
        self.income.append(optimizer.total_income())
        self.update_demand_resource_counts(optimizer)
        self.update_prod_resource_counts(optimizer)

    def update_demand_resource_counts(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.demand_counts) == 0:
            self.demand_counts = [[] for _ in optimizer.demand_resources]
        index = 0
        for r in optimizer.demand_resources:
            self.demand_counts[index].append(r.amount)
            index += 1

    def update_prod_resource_counts(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.prod_counts) == 0:
            self.prod_counts = [[] for _ in optimizer.prod_resources]
        index = 0
        for r in optimizer.prod_resources:
            self.prod_counts[index].append(r.amount)
            index += 1

    def update_time(self, cost, optimizer):
        # type: (float, Optimizer) -> None
        time_taken = cost / optimizer.total_income()
        self.delta_time.append(time_taken)
        if len(self.time) > 0:
            self.time.append(self.time[-1] + time_taken)
        else:
            self.time.append(time_taken)

    def time_hours(self):
        return [t/3600 for t in self.time]


if __name__ == "__main__":
    data_loader = DataLoader.DataLoader()
    data_loader.load_optimizer()
    opt = data_loader.load_optimizer()
    opt.run_optimization(1000)

    stat_tracker = opt.stat_tracker

    plt.semilogy(stat_tracker.time_hours(), [d/3600 for d in stat_tracker.delta_time])
    plt.figure()

    time_hours = stat_tracker.time_hours()
    index = 0
    for count_list in stat_tracker.demand_counts:
        plt.plot(time_hours, count_list, label="Demand Resource {0} Count".format(index))
        index += 1
    plt.legend()
    plt.figure()

    time_hours = stat_tracker.time_hours()
    index = 0
    for count_list in stat_tracker.prod_counts:
        plt.plot(time_hours, count_list, label="Prod Resource {0} Count".format(index))
        index += 1
    plt.legend()
    plt.show()

    power_value = PowerValue()
    power_value.amount_second_growth_rate = 200
    power_value.base_cost = 10

    factors = list()
    costs = list()
    counts = list()
    for _ in range(100):
        factors.append(power_value.factor())
        costs.append(power_value.cost())
        counts.append(power_value.count)
        power_value.make_purchase()

    plt.semilogy(factors)
    plt.semilogy(costs)
    # plt.plot(counts)
    plt.show()

