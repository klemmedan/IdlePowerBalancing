from DynamicBalancing import *
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self):
        self.stat_tracker = None  # type: StatTracker

    def do_plots(self):
        self.plot_separated_demand()
        self.plot_demand_counts()
        self.plot_demand_costs()
        self.plot_separated_prod()
        self.plot_prod_counts()
        self.plot_prod_costs()
        self.plot_combined_income()
        self.plot_cumulative_income()
        plt.show()

    def plot_separated_demand(self):
        data = self.stat_tracker.single_demand_values
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Demand")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Demand")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Demand")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.semilogy(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Demand")

    def plot_demand_counts(self):
        data = self.stat_tracker.demand_counts
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Demand Resource Count")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Demand Resource Count")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Demand Resource Count")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.plot(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Demand Resource Count")

    def plot_demand_costs(self):
        data = self.stat_tracker.demand_costs
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Demand Resource Cost")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Demand Resource Cost")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Demand Resource Cost")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.semilogy(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Demand Resource Cost")

    def plot_separated_prod(self):
        data = self.stat_tracker.single_prod_values
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Production")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Production")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Production")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.semilogy(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Production")

    def plot_prod_counts(self):
        data = self.stat_tracker.prod_counts
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Prod Resource Count")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Prod Resource Count")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.plot(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Prod Resource Count")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.plot(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Prod Resource Count")

    def plot_prod_costs(self):
        data = self.stat_tracker.prod_costs
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Prod Resource Cost")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Prod Resource Cost")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Prod Resource Cost")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        for i, d in enumerate(data):
            plt.semilogy(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Prod Resource Cost")

    def plot_cumulative_income(self):
        prod = self.stat_tracker.cumulative_prod
        demand = self.stat_tracker.cumulative_demand
        income = self.stat_tracker.cumulative_income
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Cumulative")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Cumulative")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Cumulative")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Cumulative")

    def plot_combined_income(self):
        prod = self.stat_tracker.production
        demand = self.stat_tracker.demand
        income = self.stat_tracker.income
        plt.figure()

        plt.subplot(2, 2, 1)
        time = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Current")

        plt.subplot(2, 2, 2)
        time = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel("Current")

        plt.subplot(2, 2, 3)
        time = [t for t in self.stat_tracker.time_hours() if t <= 300]
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Current")

        plt.subplot(2, 2, 4)
        time = self.stat_tracker.time_days()
        plt.semilogy(time, [x for i, x in enumerate(prod) if i < len(time)], label='Production')
        plt.semilogy(time, [x for i, x in enumerate(demand) if i < len(time)], label='Demand')
        plt.semilogy(time, [x for i, x in enumerate(income) if i < len(time)], label='Income')
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Current")