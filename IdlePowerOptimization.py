import matplotlib.pyplot as plt
import numpy
import math
import copy
from typing import List, Optional
import DataLoader
import cProfile
import timeit

purchase_factor = 1.15


class Optimizer:

    def __init__(self):
        self.stat_tracker = StatTracker()

        self.prod_resources = list()  # type: List[Resource]
        self.demand_resources = list()  # type: List[Resource]
        self.prod_multi_unlock_factors = list()  # type: List[float]
        self.prod_multi_unlock_thresholds = list()  # type: List[float]
        self.demand_multi_unlock_factors = list()  # type: List[float]
        self.demand_multi_unlock_thresholds = list()  # type: List[float]
        self.prod_multi_upgrades = UpgradeManager()
        self.demand_multi_upgrades = UpgradeManager()
        self.prod_upgrade_managers = list()  # type: List[UpgradeManager]
        self.demand_upgrade_managers = list()  # type: List[UpgradeManager]
        self.power_value = PowerValue()
        self.power_value_upgrade_manager = UpgradeManager()

        self.current_demand = 1
        self.current_prod = 1
        self.current_income = 1
        self.current_income_multiplier = 1

        self.prestige_manager = Prestige()

    def run_optimization(self, num_steps):
        self.current_prod = self.total_prod_income()
        self.current_demand = self.total_demand_income()
        self.current_income_multiplier = self.income_multiplier()
        self.current_income = self.total_income()
        for i in range(num_steps):

            if i > 0 and self.check_prestige():
                self.do_prestige()
            elif self.current_prod < self.current_demand:
                self.improve_prod()
            else:
                self.improve_demand()

            self.current_prod = self.total_prod_income()
            self.current_demand = self.total_demand_income()
            self.current_income_multiplier = self.income_multiplier()
            self.current_income = self.total_income()
            self.stat_tracker.update_stats(self)

    def improve_prod(self):
        (count_index, count_score) = self.max_prod_resource_score()
        (upgrade_index, upgrade_score) = self.max_prod_upgrade_score()
        power_value_score = self.power_value_score()
        power_value_upgrade_score = self.power_value_upgrade_score()
        multi_upgrade_score = self.multi_prod_upgrade_score()

        max_score = max(count_score, upgrade_score, power_value_score, power_value_upgrade_score, multi_upgrade_score)

        if upgrade_score == max_score:
            self.stat_tracker.update_time(self.prod_upgrade_managers[upgrade_index].next_cost(), self)
            self.stat_tracker.prod_purchase_made(self.prod_upgrade_managers[upgrade_index].next_cost())
            self.prod_upgrade_managers[upgrade_index].make_purchase()
        elif count_score == max_score:
            self.stat_tracker.update_time(self.prod_resources[count_index].current_cost, self)
            self.stat_tracker.prod_purchase_made(self.prod_resources[count_index].current_cost)
            self.prod_resources[count_index].make_purchase()
        elif power_value_score == max_score:
            self.make_power_value_purchase()
        elif power_value_upgrade_score == max_score:
            self.make_power_value_upgrade_purchase()
        elif multi_upgrade_score == max_score:
            self.stat_tracker.update_time(self.prod_multi_upgrades.next_cost(), self)
            self.stat_tracker.prod_purchase_made(self.prod_multi_upgrades.next_cost())
            self.prod_multi_upgrades.make_purchase()

    def improve_demand(self):
        (count_index, count_score) = self.max_demand_resource_score()
        (upgrade_index, upgrade_score) = self.max_demand_upgrade_score()
        power_value_score = self.power_value_score()
        power_value_upgrade_score = self.power_value_upgrade_score()
        multi_upgrade_score = self.multi_demand_upgrade_score()
        # print(count_score, upgrade_score, power_value_score, power_value_upgrade_score, multi_upgrade_score)

        max_score = max(upgrade_score, count_score, power_value_score, power_value_upgrade_score, multi_upgrade_score)

        if upgrade_score == max_score:
            self.stat_tracker.update_time(self.demand_upgrade_managers[upgrade_index].next_cost(), self)
            self.stat_tracker.demand_purchase_made(self.demand_upgrade_managers[upgrade_index].next_cost())
            self.demand_upgrade_managers[upgrade_index].make_purchase()
        elif count_score == max_score:
            self.stat_tracker.update_time(self.demand_resources[count_index].cost(), self)
            self.stat_tracker.demand_purchase_made(self.demand_resources[count_index].cost())
            self.demand_resources[count_index].make_purchase()
        elif power_value_score == max_score:
            self.make_power_value_purchase()
        elif power_value_upgrade_score == max_score:
            self.make_power_value_upgrade_purchase()
        elif multi_upgrade_score == max_score:
            self.stat_tracker.update_time(self.demand_multi_upgrades.next_cost(), self)
            self.stat_tracker.demand_purchase_made(self.demand_multi_upgrades.next_cost())
            self.demand_multi_upgrades.make_purchase()

    def make_power_value_purchase(self):
        self.stat_tracker.update_time(self.power_value.cost(), self)
        self.stat_tracker.power_value_purchase_made(self.power_value.cost())
        self.power_value.make_purchase()

    def make_power_value_upgrade_purchase(self):
        self.stat_tracker.update_time(self.power_value_upgrade_manager.next_cost(), self)
        self.stat_tracker.power_value_purchase_made(self.power_value_upgrade_manager.next_cost())
        self.power_value_upgrade_manager.make_purchase()

    def total_prod_income(self):
        total = sum([resource.increment() * self.prod_upgrade_managers[i].current_factor()
                     for i, resource in enumerate(self.prod_resources)])
        total = total * self.prod_multi_upgrades.current_factor() * self.current_prod_multi_unlock_factor()
        total = total * self.prestige_manager.prod_factor()
        return total

    def total_demand_income(self):
        total = sum([resource.increment() * self.demand_upgrade_managers[i].current_factor()
                     for i, resource in enumerate(self.demand_resources)])
        total = total * self.demand_multi_upgrades.current_factor() * self.current_demand_multi_unlock_factor()
        total = total * self.prestige_manager.prod_factor()
        return total

    def total_income(self):
        return min(self.current_demand, self.current_prod)*self.current_income_multiplier

    def income_multiplier(self):
        return self.power_value.current_factor*self.power_value_upgrade_manager.current_factor()

    def max_prod_resource_score(self):
        prestige_prod_factor = self.prestige_manager.prod_factor()

        score_list = [(r.increment_if_purchased() - r.increment()) * self.current_income_multiplier
                      * prestige_prod_factor * self.prod_upgrade_managers[i].current_factor() / r.cost()
                      for i, r in enumerate(self.prod_resources)]
        max_score = max(score_list)
        max_index = score_list.index(max_score)
        if self.purchase_will_trigger_prod_multi_unlock(max_index):
            multi_factor = self.next_prod_multi_unlock_factor() / self.current_prod_multi_unlock_factor() - 1
            max_score = max_score + multi_factor * self.current_income / self.prod_resources[max_index].cost()

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
            score = resource_increment * (next_factor/current_factor - 1) * self.current_income_multiplier / cost
            score = score * self.prestige_manager.prod_factor()
            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score

    def multi_prod_upgrade_score(self):
        total_income = self.current_income
        cost = self.prod_multi_upgrades.next_cost()
        factor = self.prod_multi_upgrades.next_factor() / self.prod_multi_upgrades.current_factor() - 1
        return total_income*factor/cost

    def max_demand_resource_score(self):
        prestige_demand_factor = self.prestige_manager.demand_factor()

        score_list = [(r.increment_if_purchased() - r.increment()) * self.current_income_multiplier
                      * prestige_demand_factor * self.demand_upgrade_managers[i].current_factor() / r.cost()
                      for i, r in enumerate(self.demand_resources)]
        max_score = max(score_list)
        max_index = score_list.index(max_score)
        if self.purchase_will_trigger_demand_multi_unlock(max_index):
                multi_factor = self.next_demand_multi_unlock_factor()/self.current_demand_multi_unlock_factor() - 1
                max_score = max_score + multi_factor*self.current_income / self.demand_resources[max_index].cost()

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
            score = resource_increment * (next_factor/current_factor - 1) * self.current_income_multiplier / cost
            score = score * self.prestige_manager.demand_factor()
            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        return max_index, max_score

    def multi_demand_upgrade_score(self):
        cost = self.demand_multi_upgrades.next_cost()
        factor = self.demand_multi_upgrades.next_factor() / self.demand_multi_upgrades.current_factor() - 1
        return self.current_income*factor/cost

    def power_value_score(self):
        cost = self.power_value.cost()
        improvement_factor = self.power_value.factor_if_purchased()/self.power_value.factor() - 1
        return self.current_income*improvement_factor / cost

    def power_value_upgrade_score(self):
        cost = self.power_value_upgrade_manager.next_cost()
        factor = self.power_value_upgrade_manager.next_factor()/self.power_value_upgrade_manager.current_factor() - 1
        return self.current_income * factor / cost

    def current_demand_multi_unlock_factor(self):
        min_amount = min([r.amount for r in self.demand_resources])
        current_unlock_index = [i for i, v in enumerate(self.demand_multi_unlock_thresholds) if v <= min_amount][-1]
        return self.demand_multi_unlock_factors[current_unlock_index]

    def next_demand_multi_unlock_factor(self):
        min_amount = min([r.amount for r in self.demand_resources])
        current_unlock_index = [i for i, v in enumerate(self.demand_multi_unlock_thresholds) if v <= min_amount][-1]
        if len(self.demand_multi_unlock_factors) > current_unlock_index + 1:
            return self.demand_multi_unlock_factors[current_unlock_index + 1]
        else:
            return self.demand_multi_unlock_factors[current_unlock_index]

    def purchase_will_trigger_demand_multi_unlock(self, resource_index):
        resource_amount = self.demand_resources[resource_index].amount
        min_amount = min([r.amount for r in self.demand_resources])
        if resource_amount > min_amount:
            return False
        current_unlock_index = [i for i, v in enumerate(self.demand_multi_unlock_thresholds) if v <= min_amount][-1]
        if len(self.demand_multi_unlock_thresholds) > current_unlock_index + 1:
            next_threshold = self.demand_multi_unlock_thresholds[current_unlock_index + 1]
            num_to_purchase = self.demand_resources[resource_index].num_to_purchase()
            return resource_amount + num_to_purchase >= next_threshold
        else:
            return False

    def current_prod_multi_unlock_factor(self):
        min_amount = min([r.amount for r in self.prod_resources])
        current_unlock_index = [i for i, v in enumerate(self.prod_multi_unlock_thresholds) if v <= min_amount][-1]
        return self.prod_multi_unlock_factors[current_unlock_index]

    def next_prod_multi_unlock_factor(self):
        min_amount = min([r.amount for r in self.prod_resources])
        current_unlock_index = [i for i, v in enumerate(self.prod_multi_unlock_thresholds) if v <= min_amount][-1]
        if len(self.prod_multi_unlock_factors) > current_unlock_index + 1:
            return self.prod_multi_unlock_factors[current_unlock_index + 1]
        else:
            return self.prod_multi_unlock_factors[current_unlock_index]

    def purchase_will_trigger_prod_multi_unlock(self, resource_index):
        resource_amount = self.prod_resources[resource_index].amount
        min_amount = min([r.amount for r in self.prod_resources])
        if resource_amount > min_amount:
            return False
        current_unlock_index = [i for i, v in enumerate(self.prod_multi_unlock_thresholds) if v <= min_amount][-1]
        if len(self.prod_multi_unlock_thresholds) > current_unlock_index + 1:
            next_threshold = self.prod_multi_unlock_thresholds[current_unlock_index + 1]
            num_to_purchase = self.prod_resources[resource_index].num_to_purchase()
            return resource_amount + num_to_purchase >= next_threshold
        else:
            return False

    def check_prestige(self):
        points_on_prestige = self.prestige_manager.points_available_on_prestige(self.stat_tracker.cumulative_prod[-1])
        return points_on_prestige > 100 and points_on_prestige > 2 * self.prestige_manager.available_points

    def do_prestige(self):
        new_prestige_points = self.prestige_manager.points_available_on_prestige(self.stat_tracker.cumulative_prod[-1])
        self.prestige_manager.available_points = new_prestige_points
        self.prod_multi_upgrades.prestige()
        self.demand_multi_upgrades.prestige()

        for manager in self.demand_upgrade_managers:
            manager.prestige()
        for manager in self.prod_upgrade_managers:
            manager.prestige()
        for resource in self.prod_resources:
            resource.prestige()
        self.prod_resources[0].amount = 1
        self.prod_resources[0].update_cost()
        for resource in self.demand_resources:
            resource.prestige()
        self.demand_resources[0].amount = 1
        self.demand_resources[0].update_cost()

        self.power_value.count = 0
        self.power_value.update_factor()
        self.power_value_upgrade_manager.prestige()
        self.stat_tracker.update_time(0, self)
        self.stat_tracker.update_prestige_time_stamp()

        # print(self.income_multiplier())


class PowerValue:

    def __init__(self):
        self.amount_growth_rate = 1
        self.amount_second_growth_rate = 0.1
        self.count = 0
        self.base_cost = 50000
        self.cost_growth_rate = 1.2

        self.current_factor = 1
        self.update_factor()

    def factor(self):
        return self.current_factor

    def update_factor(self):
        self.current_factor = 1 + self.count*self.amount_growth_rate + \
                              self.count*(self.count - 1)*self.amount_second_growth_rate/2

    def cost(self):
        n = self.num_to_purchase()
        gr = self.cost_growth_rate
        total_cost = gr * (1 - gr ** (n + self.count - 1)) / (1 - gr)
        previous_cost = gr * (1 - gr ** (self.count - 1)) / (1 - gr)
        diff_cost = self.base_cost * (total_cost - previous_cost)
        return diff_cost

    def num_to_purchase(self):
        base_increase = math.floor(self.count*(purchase_factor - 1))
        return max(1, base_increase)

    def factor_if_purchased(self):
        bought_resource = copy.copy(self)
        bought_resource.make_purchase()
        return bought_resource.factor()

    def make_purchase(self):
        self.count = self.count + self.num_to_purchase()
        self.update_factor()


class Resource:

    def __init__(self):
        self.amount = 0
        self.base_increment = 1
        self.base_cost = 10
        self.growth_rate = 1.1
        self.current_cost = self.base_cost
        self.current_unlock_index = 0
        self.current_num_to_purchase = 1
        # Threshold to achieve an unlock. Start at 0 (corresponds to 1x factor, and go from there)
        self.unlock_thresholds = list()
        # Factor is cumulative and starts at 1 (i.e. 1, 2, 4, 12, 36, etc.)
        self.unlock_factors = list()

    def increment(self):
        unlock_factor = self.unlock_factors[self.current_unlock_index]
        return self.amount * self.base_increment * unlock_factor

    def cost(self):
        return self.current_cost

    def unlock_index(self):
        return max([i for i, v in enumerate(self.unlock_thresholds) if v <= self.amount])

    def next_unlock_threshold(self):
        if self.current_unlock_index + 1 < len(self.unlock_thresholds):
            return self.unlock_thresholds[self.current_unlock_index + 1]
        else:
            return 1e9

    def num_to_purchase(self):
        # type: () -> int
        return self.current_num_to_purchase

    def update_num_to_purchase(self):
        to_next_threshold = self.next_unlock_threshold() - self.amount
        base_increase = math.floor(self.amount*(purchase_factor - 1))
        return max(1, min(base_increase, to_next_threshold))

    def increment_if_purchased(self):
        new_amount = self.amount + self.num_to_purchase()
        unlock_factor = self.unlock_factors[self.current_unlock_index]
        if self.next_unlock_threshold() == new_amount and self.current_unlock_index + 1 < len(self.unlock_factors):
            unlock_factor = self.unlock_factors[self.current_unlock_index + 1]
        return new_amount * self.base_increment * unlock_factor

    def make_purchase(self):
        self.amount = self.amount + self.num_to_purchase()
        self.current_num_to_purchase = self.update_num_to_purchase()
        self.update_cost()
        self.current_unlock_index = self.unlock_index()

    def update_cost(self):
        n = self.num_to_purchase()
        total_cost = self.growth_rate * (1 - self.growth_rate ** (n + self.amount-1)) / (1 - self.growth_rate)
        previous_cost = self.growth_rate * (1 - self.growth_rate ** (self.amount-1)) / (1 - self.growth_rate)
        diff_cost = self.base_cost * (total_cost - previous_cost)
        self.current_cost = diff_cost

    def prestige(self):
        self.amount = 0
        self.current_num_to_purchase = self.update_num_to_purchase()
        self.update_cost()
        self.current_unlock_index = self.unlock_index()

    def initialize(self):
        self.update_cost()


class UpgradeManager:

    def __init__(self):
        self.current_index = 0
        self.factor = 1

        # Cost to achieve an upgrade. Start at 0 and then go up (0 is for current/initial cost)
        self.upgrade_costs = list()

        # Cumulative factor, start at 1, then multiples all the way up.
        self.upgrade_factors = list()

    def current_cost(self):
        return self.upgrade_costs[self.current_index]

    def current_factor(self):
        return self.factor

    def next_cost(self):
        if self.current_index + 1 < len(self.upgrade_costs):
            return self.upgrade_costs[self.current_index + 1]
        else:
            return math.inf

    def next_factor(self):
        if self.current_index + 1 < len(self.upgrade_factors):
            return self.upgrade_factors[self.current_index + 1]
        else:
            return self.upgrade_factors[-1]

    def make_purchase(self):
        self.current_index += 1
        self.factor = self.upgrade_factors[self.current_index]

    def prestige(self):
        self.current_index = 0
        self.factor = self.upgrade_factors[0]


class Prestige:
    def __init__(self):
        self.tipping_point = 1e15
        self.tipping_point_amount = 100
        self.available_points = 0
        self.bonus_per_point = .01
        self.count_bonus_max_out_magnitude = 200
        self.final_growth_rate = 1.01

    def points_available_on_prestige(self, cumulative_prod):
        return self.tipping_point_amount * math.sqrt(cumulative_prod/self.tipping_point)

    def prod_factor(self):
        return 1 + self.bonus_per_point * self.available_points/2

    def demand_factor(self):
        return 1 + self.bonus_per_point * self.available_points/2

    def growth_rate(self, base_growth_rate):
        pass


class StatTracker:

    def __init__(self):
        self.demand = list()  # type: List[float]
        self.production = list()  # type: List[float]
        self.income_multiplier = list()
        self.income = list()
        self.delta_time = list()  # type: List[float]
        self.time = list()  # type: List[float]
        self.prod_counts = list()  # type: List[List[int]]
        self.demand_counts = list()  # type: List[List[int]]
        self.prod_costs = list()  # type: List[List[float]]
        self.demand_costs = list()  # type: List[List[float]]
        self.cumulative_prod = list()
        self.cumulative_demand = list()
        self.cumulative_income = list()

        self.prestige_time_stamps = list()
        self.times_to_prestige = list()

        self.demand_purchase_indices = []
        self.production_purchase_indices = []
        self.power_value_purchase_indices = []
        self.demand_purchase_flat_costs = []
        self.production_purchase_flat_costs = []
        self.power_value_purchase_flat_costs = []

    def update_stats(self, optimizer):
        # type: (Optimizer) -> None
        self.demand.append(optimizer.current_demand)
        self.production.append(optimizer.current_prod)
        self.income.append(optimizer.current_income)
        self.update_demand_resource_counts(optimizer)
        self.update_prod_resource_counts(optimizer)
        self.update_demand_resource_costs(optimizer)
        self.update_prod_resource_costs(optimizer)
        self.income_multiplier.append(optimizer.current_income_multiplier)
        self.update_cumulative_values()

    def update_demand_resource_counts(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.demand_counts) == 0:
            self.demand_counts = [[] for _ in optimizer.demand_resources]
        [self.demand_counts[i].append(r.amount) for i, r in enumerate(optimizer.demand_resources)]

    def update_prod_resource_counts(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.prod_counts) == 0:
            self.prod_counts = [[] for _ in optimizer.prod_resources]
        [self.prod_counts[i].append(r.amount) for i, r in enumerate(optimizer.prod_resources)]

    def update_demand_resource_costs(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.demand_costs) == 0:
            self.demand_costs = [[] for _ in optimizer.demand_resources]
        [self.demand_costs[i].append(r.cost()) for i, r in enumerate(optimizer.demand_resources)]

    def update_prod_resource_costs(self, optimizer):
        # type: (Optimizer) -> None
        if len(self.prod_costs) == 0:
            self.prod_costs = [[] for _ in optimizer.prod_resources]
        [self.prod_costs[i].append(r.cost()) for i, r in enumerate(optimizer.prod_resources)]

    def update_time(self, cost, optimizer):
        # type: (float, Optimizer) -> None
        time_taken = cost / optimizer.current_income
        self.delta_time.append(time_taken)
        if len(self.time) > 0:
            self.time.append(self.time[-1] + time_taken)
        else:
            self.time.append(time_taken)

    def update_prestige_time_stamp(self):
        # NOTE: This must be called after update time to work properly.
        self.prestige_time_stamps.append(self.time[-1])
        if len(self.times_to_prestige) == 0:
            self.times_to_prestige.append(self.prestige_time_stamps[0])
        else:
            self.times_to_prestige.append(self.prestige_time_stamps[-1] - self.prestige_time_stamps[-2])

    def update_cumulative_values(self):
        # Warning: This function makes assumption that update_time has already been called this iteration.
        if len(self.cumulative_prod) > 0:
            self.cumulative_demand.append(self.cumulative_demand[-1] + self.demand[-1] * self.delta_time[-1])
            self.cumulative_prod.append(self.cumulative_prod[-1] + self.production[-1] * self.delta_time[-1])
            self.cumulative_income.append(self.cumulative_income[-1] + self.income[-1] * self.delta_time[-1])
        else:
            self.cumulative_demand.append(self.demand[-1] * self.delta_time[-1])
            self.cumulative_prod.append(self.production[-1] * self.delta_time[-1])
            self.cumulative_income.append(self.income[-1] * self.delta_time[-1])

    def time_hours(self):
        return [t/3600 for t in self.time]

    def demand_purchase_made(self, cost):
        self.demand_purchase_indices.append(len(self.demand))
        self.demand_purchase_flat_costs.append(cost)

    def prod_purchase_made(self, cost):
        self.production_purchase_indices.append(len(self.demand))
        self.production_purchase_flat_costs.append(cost)

    def power_value_purchase_made(self, cost):
        self.power_value_purchase_indices.append(len(self.demand))
        self.power_value_purchase_flat_costs.append(cost)


if __name__ == "__main__":
    data_loader = DataLoader.DataLoader()
    data_loader.load_optimizer()
    opt = data_loader.load_optimizer()

    print([].append(1))
    # cProfile.run('opt.run_optimization(10000)')
    opt.run_optimization(1500)

    stat_tracker = opt.stat_tracker

    # plt.loglog(stat_tracker.time_hours(), stat_tracker.income)
    # plt.loglog(stat_tracker.time_hours(), stat_tracker.income_multiplier)
    # plt.loglog(stat_tracker.time_hours(), stat_tracker.demand)
    # plt.loglog(stat_tracker.time_hours(), stat_tracker.production)
    # plt.semilogy([t/3600 for t in stat_tracker.times_to_prestige])
    # plt.semilogy([t/3600 for t in stat_tracker.prestige_time_stamps])

    # total_prod_costs = [sum(i) for i in zip(*stat_tracker.prod_costs)]
    # total_demand_costs = [sum(i) for i in zip(*stat_tracker.demand_costs)]
    # min_prod_costs = [min(i) for i in zip(*stat_tracker.prod_costs)]
    # min_demand_costs = [min(i) for i in zip(*stat_tracker.demand_costs)]
    # max_prod_costs = [max(i) for i in zip(*stat_tracker.prod_costs)]
    # max_demand_costs = [max(i) for i in zip(*stat_tracker.demand_costs)]

    demand_purchase_times = [stat_tracker.time[i] for i in stat_tracker.demand_purchase_indices]
    prod_purchase_time = [stat_tracker.time[i] for i in stat_tracker.production_purchase_indices]

    plt.loglog(demand_purchase_times, stat_tracker.demand_purchase_flat_costs)
    plt.loglog(prod_purchase_time, stat_tracker.production_purchase_flat_costs)
    plt.figure()
    print(min(stat_tracker.demand_purchase_flat_costs))

    time_hours = stat_tracker.time_hours()
    index = 0
    for count_list in stat_tracker.demand_counts:
        plt.plot(count_list, label="Demand Resource {0} Count".format(index))
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
