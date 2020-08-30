from DataLoader import *
from IdlePowerOptimization import *


prestige_manager = Prestige()
prestige_manager.start_resource_orders_of_magnitude = 100
prestige_manager.cost_reduction_orders_of_magnitude = 100
prestige_manager.growth_rate_orders_of_magnitude = 100

print("resource start should be 0 with no points:")
print(prestige_manager.demand_resource_start())

print("growth rate should be 1.1 (no change):")
print(prestige_manager.prod_growth_rate(1.1))

print("cost reduction factor should be 1 (no points)")
print(prestige_manager.prod_cost_reduction_factor())

print("factor should be 1 with no points:")
print(prestige_manager.prod_factor())

prestige_manager.available_points = 1e5

print("Have 1e5 points, should have some resource start now:")
print(prestige_manager.demand_resource_start())

print("Have 1e5 points now should have some cost reduction:")
print(prestige_manager.prod_cost_reduction_factor())

print("growth rate should be below 1.1 (1e5 points):")
print(prestige_manager.prod_growth_rate(1.1))

print("factor should be 1001 (1e5 points)")
print(prestige_manager.prod_factor())


prestige_manager.available_points = 1e99

print("Have 1e99 points, should have resource start close to max")
print(prestige_manager.demand_resource_start())

print("Have 1e99 points now should have some cost reduction close to max:")
print(prestige_manager.prod_cost_reduction_factor())

print("growth rate should be almost 1.01 (1e99 points):")
print(prestige_manager.prod_growth_rate(1.1))

print("factor should be 1e97 (1e99 points)")
print(prestige_manager.prod_factor())


prestige_manager.available_points = 1e108

print("Have 1e108 points, should have resource start at 1000")
print(prestige_manager.demand_resource_start())

print("Have 1e108 points now should have 1000:")
print(prestige_manager.prod_cost_reduction_factor())

print("growth rate should be 1.01 (1e108 points):")
print(prestige_manager.prod_growth_rate(1.1))

print("factor should be 1e106 (1e108 points)")
print(prestige_manager.prod_factor())


