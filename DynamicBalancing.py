import numpy as np
import math
import copy
from typing import List
import DynamicBalanceDataLoader
from PlottingStuff import *
from GenerateUpgradeTimes import GenerateUpgradeTimes
from GenerateUpgradeTimes import add_randomness, massage_early_upgrade_times
import DataExport
import random

prod_single_key = 'prod_single_upgrades'
demand_single_key = 'demand_single_upgrades'
prod_multi_key = 'prod_multi_upgrades'
demand_multi_key = 'demand_multi_upgrades'
power_value_key = 'power_value_upgrades'

purchase_factor = 1.05


class DynamicBalancer:

    def __init__(self):
        self.stat_tracker = StatTracker()
        self.factor_generator = FactorManager()  # type: FactorManager
        self.factor_generator.balancer = self
        self.battery_generator = BatteryEnergyValue()

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
        # self.power_value_upgrade_manager = UpgradeManager()
        self.current_multi_demand_unlock_index = 0
        self.current_multi_prod_unlock_index = 0

        self.current_demand = 1
        self.current_prod = 1
        self.current_income = 1
        self.current_income_multiplier = 1

        self.prestige_manager = Prestige()

    def add_factor_info_to_resources(self):
        for i, r in enumerate(self.demand_resources):
            r.resource_index = i
            r.factor_generator = self.factor_generator
        for i, r in enumerate(self.prod_resources):
            r.resource_index = i
            r.factor_generator = self.factor_generator

    def run_optimization_definite_times(self):
        self.initialize_battery_value()
        num_upgrades = self.num_upgrades()
        time_generator = GenerateUpgradeTimes(num_upgrades + 1)
        times = time_generator.calculate_parabolic_times(b=0)
        times = massage_early_upgrade_times(times)
        times = add_randomness(times)
        ordered_upgrades = self.generate_ordered_list()
        self.remove_upgrade_costs()
        self.current_prod = self.total_prod_income()
        self.current_demand = self.total_demand_income()
        self.current_income_multiplier = self.income_multiplier()
        self.current_income = self.total_income()
        i = 0
        while i < num_upgrades:

            next_upgrade_cost = self.total_income() * times[i]
            prod_resource_cost = min([r.cost() for r in self.prod_resources])
            demand_resource_cost = min([r.cost() for r in self.demand_resources])
            # power_value_cost = self.power_value.cost()
            min_cost = min(prod_resource_cost, demand_resource_cost)

            if min_cost <= next_upgrade_cost / 5:
                self.stat_tracker.update_time(min_cost, self)
                times[i] = times[i] - min_cost / self.current_income
                if min_cost == prod_resource_cost:
                    self.purchase_prod_resource()
                else:
                    self.purchase_demand_resource()
            else:
                self.purchase_upgrade(ordered_upgrades[i], next_upgrade_cost)
                i += 1

            self.current_prod = self.total_prod_income()
            self.current_demand = self.total_demand_income()
            self.current_income_multiplier = self.income_multiplier()
            self.current_income = self.total_income()

            self.stat_tracker.update_stats(self)
            self.handle_batteries()
            new_prestige_points = \
                self.prestige_manager.points_available_on_prestige(self.stat_tracker.cumulative_prod[-1])
            self.prestige_manager.available_points = new_prestige_points
            self.update_resource_costs()
            # self.update_resource_unlock_thresholds_from_prestige()

    def initialize_battery_value(self):
        num_tiers = self.battery_generator.num_battery_tiers
        time_generator = GenerateUpgradeTimes(num_tiers)
        times = time_generator.calculate_parabolic_times(b=0)
        times[0] = 6000
        times[1] = 10000
        times[2] = 15000
        times[3] = 20000
        times[4] = 25000
        times[5] = 30000
        times[6] = 70000
        times[7] = 160000
        times = add_randomness(times)
        self.battery_generator.synthesis_upgrade_times = times

    def handle_batteries(self):
        current_index = self.battery_generator.synthesis_tier
        cost = self.battery_generator.synthesis_upgrade_times[current_index] * self.total_income()
        self.battery_generator.update_elapsed_time(self.stat_tracker.delta_time[-1])
        if self.battery_generator.synthesis_tier > current_index:
            self.stat_tracker.update_battery_tier_upgrade_costs(cost)
            self.stat_tracker.update_battery_tier_synthesis_energy(self)

    def purchase_upgrade(self, upgrade_packet, cost):
        upgrade_type = upgrade_packet[0]
        upgrade_index = upgrade_packet[1]
        if upgrade_type == prod_single_key:
            manager = self.prod_upgrade_managers[upgrade_index]
            factor = self.factor_generator.factor_for_prod_resource(upgrade_index)
        elif upgrade_type == demand_single_key:
            manager = self.demand_upgrade_managers[upgrade_index]
            factor = self.factor_generator.factor_for_demand_resource(upgrade_index)
        elif upgrade_type == prod_multi_key:
            manager = self.prod_multi_upgrades
            factor = self.factor_generator.factor_for_multi_prod()
        elif upgrade_type == demand_multi_key:
            manager = self.demand_multi_upgrades
            factor = self.factor_generator.factor_for_multi_demand()
        # else:
        #     manager = self.power_value_upgrade_manager
        #     factor = 2
        manager.upgrade_costs.append(cost)
        self.stat_tracker.update_time(cost, self)
        manager.make_purchase(factor)

    def num_upgrades(self):
        num_upgrades = 0
        # -1 to all because they all start off with a '0' cost upgrade for starting at.
        for manager in self.demand_upgrade_managers:
            num_upgrades += len(manager.upgrade_costs) - 1
        for manager in self.prod_upgrade_managers:
            num_upgrades += len(manager.upgrade_costs) - 1
        num_upgrades += len(self.demand_multi_upgrades.upgrade_costs) - 1
        num_upgrades += len(self.prod_multi_upgrades.upgrade_costs) - 1
        # num_upgrades += len(self.power_value_upgrade_manager.upgrade_costs) - 1
        return num_upgrades

    def remove_upgrade_costs(self):
        self.demand_multi_upgrades.upgrade_costs = []
        self.prod_multi_upgrades.upgrade_costs = []
        # self.power_value_upgrade_manager.upgrade_costs = []
        for i in range(len(self.prod_upgrade_managers)):
            self.prod_upgrade_managers[i].upgrade_costs = []
        for i in range(len(self.demand_upgrade_managers)):
            self.demand_upgrade_managers[i].upgrade_costs = []

    def generate_ordered_list(self):
        cost_list = []
        for i, manager in enumerate(self.prod_upgrade_managers):
            cost_list = cost_list + [(prod_single_key, i, c) for c in manager.upgrade_costs if c > 0]
        for i, manager in enumerate(self.demand_upgrade_managers):
            cost_list = cost_list + [(demand_single_key, i, c) for c in manager.upgrade_costs if c > 0]
        cost_list = cost_list + [(prod_multi_key, 0, c) for c in self.prod_multi_upgrades.upgrade_costs if c > 0]
        cost_list = cost_list + [(demand_multi_key, 0, c) for c in self.demand_multi_upgrades.upgrade_costs if c > 0]
        # cost_list = cost_list + [(power_value_key, 0, c)
        #                          for c in self.power_value_upgrade_manager.upgrade_costs if c > 0]
        cost_list.sort(key=lambda x: x[2])
        return cost_list

    def run_optimization_no_prestige(self, num_steps):
        self.current_prod = self.total_prod_income()
        self.current_demand = self.total_demand_income()
        self.current_income_multiplier = self.income_multiplier()
        self.current_income = self.total_income()
        for i in range(num_steps):
            self.purchase_lowest_cost_item()

            self.current_prod = self.total_prod_income()
            self.current_demand = self.total_demand_income()
            self.current_income_multiplier = self.income_multiplier()
            self.current_income = self.total_income()
            self.stat_tracker.update_stats(self)
            new_prestige_points = \
                self.prestige_manager.points_available_on_prestige(self.stat_tracker.cumulative_prod[-1])
            self.prestige_manager.available_points = new_prestige_points
            self.update_resource_costs()

    def update_resource_costs(self):
        # This method is to update all resource costs to new one after prestige (or with the no-prestige mode, to update
        #  them all every tick so that we get the full bonus.
        for r in self.prod_resources:
            r.update_cost()
        for r in self.demand_resources:
            r.update_cost()

    def update_resource_unlock_thresholds_from_prestige(self):
        for i, r in enumerate(self.prod_resources):
            if r.unlock_threshold_is_exceeded():
                factor = self.factor_generator.factor_for_prod_resource(i)
                self.stat_tracker.update_single_prod_unlock_factor(i, factor)
                r.unlock_factors.append(r.unlock_factors[-1] * factor)
                r.current_unlock_index = r.unlock_index()
        for i, r in enumerate(self.demand_resources):
            if r.unlock_threshold_is_exceeded():
                factor = self.factor_generator.factor_for_demand_resource(i)
                self.stat_tracker.update_single_demand_unlock_factor(i, factor)
                r.unlock_factors.append(r.unlock_factors[-1] * factor)
                r.current_unlock_index = r.unlock_index()

    def purchase_lowest_cost_item(self):
        demand_multi_upgrade_cost = self.demand_multi_upgrades.next_cost() / 10
        demand_single_upgrade_cost = min([um.next_cost() for um in self.demand_upgrade_managers]) / 4
        prod_multi_upgrade_cost = self.prod_multi_upgrades.next_cost() / 10
        prod_single_upgrade_cost = min([um.next_cost() for um in self.prod_upgrade_managers]) / 4
        prod_resource_cost = min([r.cost() for r in self.prod_resources])
        demand_resource_cost = min([r.cost() for r in self.demand_resources])
        # power_value_cost = self.power_value.cost() / 2
        # power_value_upgrade_cost = self.power_value_upgrade_manager.next_cost() / 15
        min_cost = min(demand_multi_upgrade_cost, demand_single_upgrade_cost, prod_multi_upgrade_cost,
                       prod_single_upgrade_cost, prod_resource_cost, demand_resource_cost)

        self.stat_tracker.update_time(min_cost, self)
        if min_cost == demand_multi_upgrade_cost:
            self.purchase_multi_demand_upgrade()
        elif min_cost == demand_single_upgrade_cost:
            self.purchase_single_demand_upgrade()
        elif min_cost == prod_multi_upgrade_cost:
            self.purchase_multi_prod_upgrade()
        elif min_cost == prod_single_upgrade_cost:
            self.purchase_single_prod_upgrade()
        elif min_cost == prod_resource_cost:
            self.purchase_prod_resource()
        elif min_cost == demand_resource_cost:
            self.purchase_demand_resource()
        # elif min_cost == power_value_cost:
        #     self.purchase_power_value_resource()
        # elif min_cost == power_value_upgrade_cost:
        #     self.purchase_power_value_upgrade()

    def purchase_multi_demand_upgrade(self):
        factor = self.factor_generator.factor_for_multi_demand()
        self.stat_tracker.demand_multi_upgrade_factors.append(factor)
        self.demand_multi_upgrades.make_purchase(factor)

    def purchase_multi_prod_upgrade(self):
        factor = self.factor_generator.factor_for_multi_prod()
        self.stat_tracker.prod_multi_upgrade_factors.append(factor)
        self.prod_multi_upgrades.make_purchase(factor)

    def purchase_single_demand_upgrade(self):
        cost_list = [um.next_cost() for um in self.demand_upgrade_managers]
        index = cost_list.index(min(cost_list))
        factor = self.factor_generator.factor_for_demand_resource(index)
        self.stat_tracker.update_single_demand_upgrade_factor(index, factor)
        self.demand_upgrade_managers[index].make_purchase(factor)

    def purchase_single_prod_upgrade(self):
        cost_list = [um.next_cost() for um in self.prod_upgrade_managers]
        index = cost_list.index(min(cost_list))
        factor = self.factor_generator.factor_for_prod_resource(index)
        self.stat_tracker.update_single_prod_upgrade_factor(index, factor)
        self.prod_upgrade_managers[index].make_purchase(factor)

    def purchase_prod_resource(self):
        cost_list = [r.cost() for r in self.prod_resources]
        index = cost_list.index(min(cost_list))
        resource = self.prod_resources[index]
        factor = self.factor_generator.factor_for_prod_resource(index)
        if self.prod_purchase_will_trigger_multi_unlock(index):
            self.generate_prod_multi_unlock_factor()
        if resource.purchase_will_reach_unlock_threshold():
            self.stat_tracker.update_single_prod_unlock_factor(index, factor)
        resource.make_purchase(factor)

    def purchase_demand_resource(self):
        cost_list = [r.cost() for r in self.demand_resources]
        index = cost_list.index(min(cost_list))
        resource = self.demand_resources[index]
        factor = self.factor_generator.factor_for_demand_resource(index)
        if self.demand_purchase_will_trigger_multi_unlock(index):
            self.generate_demand_multi_unlock_factor()
        if resource.purchase_will_reach_unlock_threshold():
            self.stat_tracker.update_single_demand_unlock_factor(index, factor)
        resource.make_purchase(factor)

    def purchase_power_value_resource(self):
        self.power_value.make_purchase()

    def purchase_power_value_upgrade(self):
        pass
        # self.stat_tracker.update_time(self.power_value_upgrade_manager.next_cost(), self)
        # self.stat_tracker.power_value_purchase_made(self.power_value_upgrade_manager.next_cost())
        # self.power_value_upgrade_manager.make_purchase(2)

    def make_power_value_purchase(self):
        pass
        # self.stat_tracker.update_time(self.power_value.cost(), self)
        # self.stat_tracker.power_value_purchase_made(self.power_value.cost())
        # self.power_value.make_purchase()

    def total_prod_income(self):
        total = sum([resource.increment() * self.prod_upgrade_managers[i].current_factor()
                     for i, resource in enumerate(self.prod_resources)])
        total = total * self.prod_multi_upgrades.current_factor() * self.current_prod_multi_unlock_factor()
        total = total * self.prestige_manager.prod_factor()
        return total

    def single_resource_production(self, index):
        resource = self.prod_resources[index]
        prod = resource.increment() * self.prod_upgrade_managers[index].current_factor()
        prod = prod * self.prod_multi_upgrades.current_factor() * self.current_prod_multi_unlock_factor()
        prod = prod * self.prestige_manager.prod_factor()
        return prod

    def total_demand_income(self):
        total = sum([resource.increment() * self.demand_upgrade_managers[i].current_factor()
                     for i, resource in enumerate(self.demand_resources)])
        total = total * self.demand_multi_upgrades.current_factor() * self.current_demand_multi_unlock_factor()
        total = total * self.prestige_manager.demand_factor()
        return total

    def single_resource_demand(self, index):
        resource = self.demand_resources[index]
        demand = resource.increment() * self.demand_upgrade_managers[index].current_factor()
        demand = demand * self.demand_multi_upgrades.current_factor() * self.current_demand_multi_unlock_factor()
        demand = demand * self.prestige_manager.demand_factor()
        return demand

    def total_income(self):
        return min(self.current_demand, self.current_prod)*self.current_income_multiplier

    def income_multiplier(self):
        cumulative_energy = 0
        if len(self.stat_tracker.cumulative_prod) > 0:
            cumulative_energy = self.stat_tracker.cumulative_prod[-1]
        return self.battery_generator.power_value(cumulative_energy)

    def current_demand_multi_unlock_factor(self):
        # min_amount = min([r.amount + r.base_amount() for r in self.demand_resources])
        # current_unlock_index = [i for i, v in enumerate(self.demand_multi_unlock_thresholds) if v <= min_amount][-1]
        return self.demand_multi_unlock_factors[self.current_multi_demand_unlock_index]

    def next_demand_multi_unlock_factor(self):
        min_amount = min([r.amount + r.base_amount() for r in self.demand_resources])
        current_unlock_index = [i for i, v in enumerate(self.demand_multi_unlock_thresholds) if v <= min_amount][-1]
        if len(self.demand_multi_unlock_factors) > current_unlock_index + 1:
            return self.demand_multi_unlock_factors[current_unlock_index + 1]
        else:
            return self.demand_multi_unlock_factors[current_unlock_index]

    def prod_purchase_will_trigger_multi_unlock(self, resource_index):
        amounts = [r.amount + r.base_amount() for r in self.prod_resources]
        min_index = amounts.index(min(amounts))
        if min_index != resource_index:
            return False  # Give an early exit condition for resources that will for sure not trigger.
        if self.current_multi_prod_unlock_index + 1 >= len(self.prod_multi_unlock_thresholds):
            return False  # Exit early if all multi unlocks have been acquired.
        next_threshold = self.prod_multi_unlock_thresholds[self.current_multi_prod_unlock_index + 1]
        for i, a in enumerate(amounts):
            if i != resource_index and a < next_threshold:
                return False  # Exit if any other resource does not meet the threshold criterion.

        resource = self.prod_resources[resource_index]
        amount_after_buying = resource.amount + resource.base_amount() + resource.num_to_purchase()
        return amount_after_buying >= next_threshold

    def demand_purchase_will_trigger_multi_unlock(self, resource_index):
        amounts = [r.amount + r.base_amount() for r in self.demand_resources]
        min_index = amounts.index(min(amounts))
        if min_index != resource_index:
            return False  # Give an early exit condition for resources that will for sure not trigger.
        if self.current_multi_demand_unlock_index + 1 >= len(self.demand_multi_unlock_thresholds):
            return False  # Exit early if all multi unlocks have been acquired.
        next_threshold = self.demand_multi_unlock_thresholds[self.current_multi_demand_unlock_index + 1]
        for i, a in enumerate(amounts):
            if i != resource_index and a < next_threshold:
                return False  # Exit if any other resource does not meet the threshold criterion.

        resource = self.demand_resources[resource_index]
        amount_after_buying = resource.amount + resource.base_amount() + resource.num_to_purchase()
        return amount_after_buying >= next_threshold

    def generate_prod_multi_unlock_factor(self):
        factor = self.factor_generator.factor_for_multi_prod()
        self.stat_tracker.prod_multi_unlock_factors.append(factor)
        self.prod_multi_unlock_factors.append(self.prod_multi_unlock_factors[-1] * factor)
        self.current_multi_prod_unlock_index += 1

    def generate_demand_multi_unlock_factor(self):
        factor = self.factor_generator.factor_for_multi_demand()
        self.stat_tracker.demand_multi_unlock_factors.append(factor)
        self.demand_multi_unlock_factors.append(self.demand_multi_unlock_factors[-1] * factor)
        self.current_multi_demand_unlock_index += 1

    def current_prod_multi_unlock_factor(self):
        # min_amount = min([r.amount for r in self.prod_resources])
        # current_unlock_index = [i for i, v in enumerate(self.prod_multi_unlock_thresholds) if v <= min_amount][-1]
        return self.prod_multi_unlock_factors[self.current_multi_prod_unlock_index]

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
        self.production_type = True
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
        self.unlock_factors = [1]
        self.prestige_manager = Prestige() # type: Prestige
        self.factor_generator = FactorManager  # type: FactorManager
        self.resource_index = 0
        self.stat_tracker = StatTracker()

    def increment(self):
        unlock_factor = self.unlock_factors[self.current_unlock_index]
        return (self.amount + self.base_amount()) * self.base_increment * unlock_factor

    def cost(self):
        return self.current_cost / self.cost_reduction_factor()

    def unlock_index(self):
        total_amount = self.base_amount() + self.amount
        return max([i for i, v in enumerate(self.unlock_thresholds) if v <= total_amount])

    def next_unlock_threshold(self):
        if self.current_unlock_index + 1 < len(self.unlock_thresholds):
            return self.unlock_thresholds[self.current_unlock_index + 1]
        else:
            return 1e9

    def num_to_purchase(self):
        # type: () -> int
        return self.current_num_to_purchase

    def update_num_to_purchase(self):
        to_next_threshold = self.next_unlock_threshold() - (self.amount + self.base_amount())
        base_increase = math.floor(self.amount*(purchase_factor - 1))
        return max(1, min(base_increase, to_next_threshold))

    def purchase_will_reach_unlock_threshold(self):
        next_threshold = self.next_unlock_threshold()
        amount_after_purchase = self.current_num_to_purchase + self.amount + self.base_amount()
        return amount_after_purchase >= next_threshold

    def make_purchase(self, factor):
        unlock_reached = self.purchase_will_reach_unlock_threshold()
        if unlock_reached:
            self.unlock_factors.append(self.unlock_factors[-1] * factor)
            self.current_unlock_index += 1
        self.amount = self.amount + self.num_to_purchase()
        self.current_num_to_purchase = self.update_num_to_purchase()
        self.update_cost()
        # if unlock_reached:
        #     self.current_unlock_index = self.unlock_index()

    def update_cost(self):
        upgraded_growth_rate = self.growth_rate_after_prestige()
        n = self.num_to_purchase()
        total_cost = upgraded_growth_rate * (1 - upgraded_growth_rate ** (n + self.amount-1)) / (1 - upgraded_growth_rate)
        previous_cost = upgraded_growth_rate * (1 - upgraded_growth_rate ** (self.amount-1)) / (1 - upgraded_growth_rate)
        diff_cost = self.base_cost * (total_cost - previous_cost)
        self.current_cost = diff_cost

    def prestige(self):
        self.amount = 0
        self.current_num_to_purchase = self.update_num_to_purchase()
        self.update_cost()
        self.current_unlock_index = self.unlock_index()

    def initialize(self):
        self.update_cost()

    def growth_rate_after_prestige(self):
        if self.production_type:
            return self.prestige_manager.prod_growth_rate(self.growth_rate)
        else:
            return self.prestige_manager.demand_growth_rate(self.growth_rate)

    def cost_reduction_factor(self):
        if self.production_type:
            return self.prestige_manager.prod_cost_reduction_factor()
        else:
            return self.prestige_manager.demand_cost_reduction_factor()

    def base_amount(self):
        if self.production_type:
            return self.prestige_manager.prod_resource_start()
        else:
            return self.prestige_manager.demand_resource_start()

    def unlock_threshold_is_exceeded(self):
        test_index = self.unlock_index()
        return test_index > self.current_unlock_index


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

    def make_purchase(self, new_factor):
        self.current_index += 1
        self.upgrade_factors.append(self.upgrade_factors[-1] * new_factor)
        if self.current_index < len(self.upgrade_factors):
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
        self.final_growth_rate = 1.01
        self.growth_rate_orders_of_magnitude = 125
        self.cost_reduction_orders_of_magnitude = 125
        self.cost_reduction_max_factor = 1000
        self.start_resource_max_factor = 1000
        self.start_resource_orders_of_magnitude = 125
        self.spreading_factor = 3

    def points_available_on_prestige(self, cumulative_prod):
        return math.floor(self.tipping_point_amount * math.sqrt(cumulative_prod/self.tipping_point))

    def prod_factor(self):
        return 1 + self.bonus_per_point * self.available_points / self.spreading_factor

    def demand_factor(self):
        return 1 + self.bonus_per_point * self.available_points / self.spreading_factor

    def prod_growth_rate(self, base_growth_rate):
        # return base_growth_rate
        factor = min([1, 1/self.growth_rate_orders_of_magnitude * math.log10(1+self.available_points / self.spreading_factor)])
        growth_rate_diff = base_growth_rate - self.final_growth_rate
        return base_growth_rate - factor * growth_rate_diff

    def demand_growth_rate(self, base_growth_rate):
        # return base_growth_rate
        factor = min([1, 1/self.growth_rate_orders_of_magnitude * math.log10(1+self.available_points / self.spreading_factor)])
        growth_rate_diff = base_growth_rate - self.final_growth_rate
        return base_growth_rate - factor * growth_rate_diff

    def prod_cost_reduction_factor(self):
        factor = 1+self.cost_reduction_max_factor * (1/self.cost_reduction_orders_of_magnitude) * \
                 math.log10(1+self.available_points / self.spreading_factor)
        return min([self.cost_reduction_max_factor, factor])

    def demand_cost_reduction_factor(self):
        factor = 1+self.cost_reduction_max_factor * (1/self.cost_reduction_orders_of_magnitude) * \
                 math.log10(1+self.available_points / self.spreading_factor)
        return min([self.cost_reduction_max_factor, factor])

    def prod_resource_start(self):
        factor = self.start_resource_max_factor * (1/self.start_resource_orders_of_magnitude) * \
                 math.log10(1+self.available_points / self.spreading_factor)
        return math.floor(min([self.start_resource_max_factor, factor]))

    def demand_resource_start(self):
        factor = self.start_resource_max_factor * (1/self.start_resource_orders_of_magnitude) * \
                 math.log10(1+self.available_points / self.spreading_factor)
        return math.floor(min([self.start_resource_max_factor, factor]))


class BatteryValues:

    def __init__(self):
        self.num_batteries = 120
        self.num_capacity_upgrades = 200
        self.base_capacity = 1e5
        self.prod_capacity_factor = 1
        self.min_battery_capacity_factor = 3.5
        self.income_synthesis_factor = 0.5
        self.income_upgrade_factor = 0.5
        self.capacity_synthesis_factor = 10
        self.battery_times = self.generate_battery_times()
        self.battery_upgrade_times = self.generate_capacity_upgrade_times()
        self.current_battery_index = 0
        self.current_upgrade_index = 0
        self.current_capacity = self.base_capacity

        self.individual_capacities = list()
        self.synthesis_values = list()
        self.synthesis_upgrade_costs = list()
        self.factors = list()
        self.capacity_upgrade_costs = list()
        self.current_factor = 1

    def generate_battery_times(self):
        time_generator = GenerateUpgradeTimes(self.num_batteries + 1)
        times = [2000+0.9*t for t in time_generator.calculate_cumulative_parabolic_times(b=self.num_batteries/4)]
        print(times)
        return times

    def generate_capacity_upgrade_times(self):
        time_generator = GenerateUpgradeTimes(self.num_capacity_upgrades + 1)
        times = [2000+0.9*t for t in time_generator.calculate_cumulative_parabolic_times(b=self.num_capacity_upgrades/4)]
        return times

    def generate_battery(self, balancer):
        # type: (DynamicBalancer) -> None
        previous_capacity = 0
        if len(self.individual_capacities) > 0:
            previous_capacity = self.individual_capacities[-1]
        new_min_capacity = self.min_battery_capacity_factor * previous_capacity

        capacity = max([balancer.current_prod * self.prod_capacity_factor/self.current_factor, new_min_capacity])
        synthesis_energy = self.capacity_synthesis_factor*capacity
        synthesis_upgrade_cost = balancer.stat_tracker.cumulative_income[-1] * self.income_synthesis_factor

        self.current_capacity = capacity * self.current_factor
        self.individual_capacities.append(capacity)
        self.synthesis_values.append(synthesis_energy)
        self.synthesis_upgrade_costs.append(synthesis_upgrade_cost)
        self.current_battery_index += 1

    def generate_upgrade(self, balancer):
        # type: (DynamicBalancer) -> None
        factor = random.choice([2, 3, 4, 5])
        cost = self.income_upgrade_factor * balancer.stat_tracker.cumulative_income[-1]

        self.factors.append(factor)
        self.current_factor *= factor
        self.capacity_upgrade_costs.append(cost)
        self.current_capacity = self.individual_capacities[-1] * self.current_factor
        self.current_upgrade_index += 1

    def next_upgrade_time(self):
        if self.current_upgrade_index < self.num_capacity_upgrades:
            return self.battery_upgrade_times[self.current_upgrade_index]
        else:
            return 1e500

    def next_battery_time(self):
        if self.current_battery_index< self.num_batteries:
            return self.battery_times[self.current_battery_index]
        else:
            return 1e500


class BatteryEnergyValue:

    def __init__(self):
        self.base_power_value = 0.01
        self.base_battery_power_value = 0.001
        self.battery_combine_increase_factor = 2.5
        self.model_battery_tier_combine_start = 9
        self.model_battery_tiers_combine_end = 13
        self.num_battery_tiers = 90
        self.synthesis_tier = 0
        self.elapsed_time = 0
        self.cumulative_time = 0
        self.time_orders_of_magnitude = math.log10(2*365*24*3600)
        self.total_time = 2*365*24*3600
        self.synthesis_upgrade_times = list()
        t = GenerateUpgradeTimes(self.num_battery_tiers)
        t.start = 10
        t.end = 5*3600
        self.synthesis_times = t.calculate_cumulative_parabolic_times()

    def set_times(self, times):
        self.synthesis_upgrade_times = times
        end_time = times[-1]
        self.time_orders_of_magnitude = math.log10(end_time)

    def power_value(self, cumulative_energy):
        if cumulative_energy <= 500000:
            return self.base_power_value
        # combine_tier = self.model_battery_tiers_combined_to * (math.log10(1 + self.cumulative_time) / self.time_orders_of_magnitude)
        combine_tier = self.model_battery_tier_combine_start + (self.model_battery_tiers_combine_end -
                                                                self.model_battery_tier_combine_start)*\
                       math.sqrt(self.cumulative_time / self.total_time)
        effective_tier = self.synthesis_tier + combine_tier
        effective_tier = min(self.num_battery_tiers, effective_tier)
        power_value = self.base_battery_power_value + self.base_battery_power_value * (self.battery_combine_increase_factor ** effective_tier)
        return power_value

    def update_elapsed_time(self, time):
        self.elapsed_time += time
        self.cumulative_time += time
        if self.synthesis_tier < len(self.synthesis_upgrade_times) - 1:
            time_for_next_upgrade = self.synthesis_upgrade_times[self.synthesis_tier]
            if self.elapsed_time >= time_for_next_upgrade:
                self.synthesis_tier += 1
                self.elapsed_time = 0


class StatTracker:

    def __init__(self):
        self.demand = list()  # type: List[float]
        self.production = list()  # type: List[float]
        self.single_prod_values = list()  # type: List[List[float]]
        self.single_demand_values = list()  # type: List[List[float]]
        self.single_prod_values_cumulative = list()  # type: List[List[float]]
        self.single_demand_values_cumulative = list()  # type: List[List[float]]
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
        # self.power_value_purchase_flat_costs = []

        self.ordered_demand_purchase_costs = []
        self.ordered_production_purchase_costs = []
        self.ordered_production_indices = []
        self.ordered_demand_indices = []

        self.prod_single_unlock_factors = dict()
        self.demand_single_unlock_factors = dict()
        self.prod_multi_unlock_factors = []
        self.demand_multi_unlock_factors = []
        self.prod_single_upgrade_factors = dict()
        self.demand_single_upgrade_factors = dict()
        self.prod_multi_upgrade_factors = []
        self.demand_multi_upgrade_factors = []
        # self.power_value_upgrade_factors = []

        self.demand_single_upgrade_costs = []
        self.demand_multi_upgrade_costs = []
        self.prod_single_upgrade_costs = []
        self.prod_multi_upgrade_costs = []
        self.power_value_upgrade_costs = []

        self.battery_upgrade_tier_costs = []
        self.battery_synthesis_tier_energies = []

    def dump_upgrade_info(self, optimizer):
        # type: (DynamicBalancer) -> None
        f = optimizer.prod_multi_upgrades.upgrade_factors
        self.prod_multi_upgrade_factors = [f[i] / f[i-1] for i in range(1, len(f))]
        f = optimizer.demand_multi_upgrades.upgrade_factors
        self.demand_multi_upgrade_factors = [f[i] / f[i-1] for i in range(1, len(f))]
        # f = optimizer.power_value_upgrade_manager.upgrade_factors
        # self.power_value_upgrade_factors = [f[i] / f[i-1] for i in range(1, len(f))]
        self.prod_single_upgrade_factors = []
        for manager in optimizer.prod_upgrade_managers:
            f = manager.upgrade_factors
            self.prod_single_upgrade_factors.append([f[i] / f[i-1] for i in range(1, len(f))])
        self.demand_single_upgrade_factors = []
        for manager in optimizer.demand_upgrade_managers:
            f = manager.upgrade_factors
            self.demand_single_upgrade_factors.append([f[i] / f[i-1] for i in range(1, len(f))])

        self.demand_multi_upgrade_costs = optimizer.demand_multi_upgrades.upgrade_costs
        self.prod_multi_upgrade_costs = optimizer.prod_multi_upgrades.upgrade_costs
        # self.power_value_upgrade_costs = optimizer.power_value_upgrade_manager.upgrade_costs
        self.prod_single_upgrade_costs = [um.upgrade_costs for um in optimizer.prod_upgrade_managers]
        self.demand_single_upgrade_costs = [um.upgrade_costs for um in optimizer.demand_upgrade_managers]

    def update_stats(self, optimizer):
        # type: (DynamicBalancer) -> None
        self.demand.append(optimizer.current_demand)
        self.production.append(optimizer.current_prod)
        self.income.append(optimizer.current_income)
        self.update_demand_resource_counts(optimizer)
        self.update_prod_resource_counts(optimizer)
        self.update_demand_resource_costs(optimizer)
        self.update_prod_resource_costs(optimizer)
        self.income_multiplier.append(optimizer.current_income_multiplier)
        self.update_cumulative_values()
        self.update_single_production(optimizer)
        self.update_single_demand(optimizer)

    def update_single_production(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.single_prod_values) == 0:
            self.single_prod_values= [[] for _ in optimizer.prod_resources]
        if len(self.single_prod_values_cumulative) == 0:
            self.single_prod_values_cumulative = [[] for _ in optimizer.prod_resources]
        for i in range(len(optimizer.prod_resources)):
            prod = optimizer.single_resource_production(i)
            self.single_prod_values[i].append(prod)
            if len(self.single_prod_values_cumulative[i]) == 0:
                self.single_prod_values_cumulative[i].append(prod)
            else:
                self.single_prod_values_cumulative[i].append(prod + self.single_prod_values_cumulative[i][-1])

    def update_single_demand(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.single_demand_values) == 0:
            self.single_demand_values = [[] for _ in optimizer.demand_resources]
        if len(self.single_demand_values_cumulative) == 0:
            self.single_demand_values_cumulative = [[] for _ in optimizer.demand_resources]
        for i in range(len(optimizer.demand_resources)):
            demand = optimizer.single_resource_demand(i)
            self.single_demand_values[i].append(demand)
            if len(self.single_demand_values_cumulative[i]) == 0:
                self.single_demand_values_cumulative[i].append(demand)
            else:
                self.single_demand_values_cumulative[i].append(demand + self.single_demand_values_cumulative[i][-1])

    def update_demand_resource_counts(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.demand_counts) == 0:
            self.demand_counts = [[] for _ in optimizer.demand_resources]
        [self.demand_counts[i].append(r.amount) for i, r in enumerate(optimizer.demand_resources)]

    def update_prod_resource_counts(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.prod_counts) == 0:
            self.prod_counts = [[] for _ in optimizer.prod_resources]
        [self.prod_counts[i].append(r.amount) for i, r in enumerate(optimizer.prod_resources)]

    def update_demand_resource_costs(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.demand_costs) == 0:
            self.demand_costs = [[] for _ in optimizer.demand_resources]
        [self.demand_costs[i].append(r.cost()) for i, r in enumerate(optimizer.demand_resources)]

    def update_prod_resource_costs(self, optimizer):
        # type: (DynamicBalancer) -> None
        if len(self.prod_costs) == 0:
            self.prod_costs = [[] for _ in optimizer.prod_resources]
        [self.prod_costs[i].append(r.cost()) for i, r in enumerate(optimizer.prod_resources)]

    def update_time(self, cost, optimizer):
        # type: (float, DynamicBalancer) -> None
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

    def time_seconds(self):
        return self.time

    def time_minutes(self):
        return [t/60 for t in self.time]

    def time_days(self):
        return [t/(3600*24) for t in self.time]

    def demand_purchase_made(self, cost):
        self.demand_purchase_indices.append(len(self.demand))
        self.demand_purchase_flat_costs.append(cost)
        if len(self.ordered_demand_purchase_costs) == 0:
            self.ordered_demand_indices.append(len(self.demand))
            self.ordered_demand_purchase_costs.append(cost)
        elif cost >= self.ordered_demand_purchase_costs[-1]:
            self.ordered_demand_indices.append(len(self.demand))
            self.ordered_demand_purchase_costs.append(cost)

    def prod_purchase_made(self, cost):
        self.production_purchase_indices.append(len(self.demand))
        self.production_purchase_flat_costs.append(cost)
        if len(self.ordered_production_purchase_costs) == 0:
            self.ordered_production_indices.append(len(self.demand))
            self.ordered_production_purchase_costs.append(cost)
        elif cost >= self.ordered_production_purchase_costs[-1]:
            self.ordered_production_indices.append(len(self.demand))
            self.ordered_production_purchase_costs.append(cost)

    def power_value_purchase_made(self, cost):
        self.power_value_purchase_indices.append(len(self.demand))
        self.power_value_purchase_flat_costs.append(cost)

    def update_single_demand_upgrade_factor(self, index, factor):
        if index in self.demand_single_upgrade_factors:
            self.demand_single_upgrade_factors[index].append(factor)
        else:
            self.demand_single_upgrade_factors[index] = [factor]

    def update_single_prod_upgrade_factor(self, index, factor):
        if index in self.prod_single_upgrade_factors:
            self.prod_single_upgrade_factors[index].append(factor)
        else:
            self.prod_single_upgrade_factors[index] = [factor]

    def update_single_demand_unlock_factor(self, index, factor):
        if index in self.demand_single_unlock_factors:
            self.demand_single_unlock_factors[index].append(factor)
        else:
            self.demand_single_unlock_factors[index] = [factor]

    def update_single_prod_unlock_factor(self, index, factor):
        if index in self.prod_single_unlock_factors:
            self.prod_single_unlock_factors[index].append(factor)
        else:
            self.prod_single_unlock_factors[index] = [factor]

    def update_battery_tier_upgrade_costs(self, cost):
        self.battery_upgrade_tier_costs.append(cost)

    def update_battery_tier_synthesis_energy(self, balancer):
        # type: (DynamicBalancer) -> None
        tier_index = balancer.battery_generator.synthesis_tier
        time = balancer.battery_generator.synthesis_times[tier_index]
        energy = time * balancer.total_prod_income()
        self.battery_synthesis_tier_energies.append(energy)


class FactorManager:

    def __init__(self):
        self.balancer = None  # type: DynamicBalancer

    def factor_for_demand_resource(self, resource_index):
        resource_rate = self.single_demand_rate(resource_index)
        total_rate = self.total_demand_rate()
        # Reduce ratio further if prod > demand, (lower ratio -> Higher average factor)
        prod_demand_factor = self.demand_over_prod_factor()
        rates = [self.single_demand_rate(i) for i in range(len(self.balancer.demand_resources))]
        ratios = [r / total_rate for r in rates]
        ratio = prod_demand_factor * (resource_rate / total_rate)
        target_ratio = 0.5 * 1/len(self.balancer.demand_resources)
        return self.factor_from_ratio(ratio, target_ratio)

    def factor_for_prod_resource(self, resource_index):
        resource_rate = self.single_prod_rate(resource_index)
        total_rate = self.total_prod_rate()
        # Reduce ratio further if prod > demand, (lower ratio -> Higher average factor)
        prod_demand_factor = self.prod_over_demand_factor()
        ratio = prod_demand_factor * (resource_rate / total_rate)
        target_ratio = 0.75*1/len(self.balancer.prod_resources)
        return self.factor_from_ratio(ratio, target_ratio)

    def factor_for_multi_prod(self):
        ratio = self.prod_over_demand_factor()
        return self.factor_from_ratio(ratio, 0.5)

    def factor_for_multi_demand(self):
        ratio = self.demand_over_prod_factor()
        return self.factor_from_ratio(ratio, 0.5)

    def prod_over_demand_factor(self):
        demand = self.total_demand_rate()
        prod = self.total_prod_rate()
        return prod / (demand + prod)

    def demand_over_prod_factor(self):
        demand = self.total_demand_rate()
        prod = self.total_prod_rate()
        return demand / (demand + prod)

    def total_demand_rate(self):
        return self.balancer.total_demand_income()

    def single_demand_rate(self, resource_index):
        resource = self.balancer.demand_resources[resource_index]
        rate = resource.increment() * self.balancer.demand_upgrade_managers[resource_index].current_factor()
        rate = rate * self.balancer.demand_multi_upgrades.current_factor() * \
            self.balancer.current_demand_multi_unlock_factor()
        rate = rate * self.balancer.prestige_manager.demand_factor()
        return rate

    def total_prod_rate(self):
        return self.balancer.total_prod_income()

    def single_prod_rate(self, resource_index):
        resource = self.balancer.prod_resources[resource_index]
        rate = resource.increment() * self.balancer.prod_upgrade_managers[resource_index].current_factor()
        rate = rate * self.balancer.prod_multi_upgrades.current_factor() * \
               self.balancer.current_prod_multi_unlock_factor()
        rate = rate * self.balancer.prestige_manager.prod_factor()
        return rate

    def factor_from_ratio(self, ratio, target_ratio):
        weights = self.manual_weights(ratio, target_ratio)
        results = np.random.multinomial(1, weights, size=1)
        for index, result in enumerate(results[0]):
            if result == 1:  # This indicates desired upgrade factor
                return index + 2  # +2 because index starts at 0, upgrade factor starts at 2

    @staticmethod
    def manual_weights(ratio, target_ratio):
        # This function takes the so-called "ratio" (how productive a certain item is relative to all else) and returns
        # a set of "weights" which assign a probability for what factor to generate.

        if ratio > 5*target_ratio:
            weights = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ratio > target_ratio:
            weights = [0.85, 0.15, 0, 0, 0, 0, 0, 0, 0]
        elif ratio > 0.75*target_ratio:
            weights = [0.6, 0.3, 0.10, 0, 0, 0, 0, 0, 0]
        elif ratio > 0.5*target_ratio:
            weights = [0.4, 0.4, 0.2, 0, 0, 0, 0, 0, 0]
        elif ratio > .25*target_ratio:
            weights = [0.3, 0.3, 0.2, 0.2, 0, 0, 0, 0, 0]
        elif ratio > .1*target_ratio:
            weights = [2/9, 3/9, 2/9, 1/9, 1/9, 0, 0, 0, 0]
        elif ratio > 0.01*target_ratio:
            weights = [0, 0, 0.3, .3, .20, .10, .1, 0, 0]
        elif ratio > 0.001*target_ratio:
            weights = [0, 0, 0, .2, .10, .10, .3, .2, .1]
        elif ratio > 0.0001*target_ratio:
            weights = [0, 0, 0, 0, 0, .1, .4, .3, .2]
        elif ratio > 0.00001*target_ratio:
            weights = [0, 0, 0, 0, 0, 0, .3, .4, .3]
        elif ratio > 0.00001*target_ratio:
            weights = [0, 0, 0, 0, 0, 0, .05, .2, .75]
        else:
            weights = [0, 0, 0, 0, 0, 0, 0, .2, .8]
        return weights


if __name__ == "__main__":

    exporter = DataExport.DataExporter()
    data_loader = DynamicBalanceDataLoader.DataLoader()
    opt = data_loader.load_optimizer()
    opt.run_optimization_definite_times()
    opt.stat_tracker.dump_upgrade_info(opt)
    stats = opt.stat_tracker
    exporter.export_data(opt)
    exporter.add_demand_flavor_texts_to_final_document()
    exporter.add_prod_flavor_texts_to_final_document()
    #
    # plt.semilogy(stats.time_days(), stats.income_multiplier)
    # plt.show()
    #
    # plotter = DataPlotter()
    # plotter.stat_tracker = stats
    # plotter.do_plots()

    cum_prod = stats.cumulative_prod[-1]
    prestige_points = opt.prestige_manager.points_available_on_prestige(cum_prod)
    print("Cumulative Production: " + str(cum_prod))
    print("Prestige Points: " + str(float(prestige_points)))
    print("Cumulative Money: " + str(stats.cumulative_income[-1]))

    # num_upgrades = opt.num_upgrades()
    # time_generator = GenerateUpgradeTimes(num_upgrades + 1)
    # times = time_generator.calculate_parabolic_times(b=0)
    # times = massage_early_upgrade_times(times)
    # end_index = len(times)
    # plt.plot(times[0:end_index], 'o')
    # plt.show()

