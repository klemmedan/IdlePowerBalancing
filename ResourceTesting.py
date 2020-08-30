from DataLoader import *
from IdlePowerOptimization import *

resource = Resource()
resource.unlock_factors = [1]
resource. unlock_thresholds = [0]

print(resource.num_to_purchase())

print("original cost:")
print(resource.cost())

n = 100
for i in range(n):
    print(resource.cost())
    resource.make_purchase()

print("cost after incrementing:")
old_cost = resource.cost()
print(resource.cost())

resource.prestige_manager.available_points = 1e20
print("prestige cost factor:")
print(resource.prestige_manager.prod_cost_reduction_factor())
print("expected cost if growth rate didn't change:")
print(old_cost / resource.prestige_manager.prod_cost_reduction_factor())

print("Actual after prestige bonuses:")
print(resource.cost())
resource.update_cost()
print("Actual cost after calling 'update_cost':")
print(resource.cost())

print("new growth rate:")
print(resource.growth_rate_after_prestige())
