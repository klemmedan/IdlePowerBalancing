from DynamicBalancing import *
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self):
        self.stat_tracker = None  # type: StatTracker
        self.time_seconds = []
        self.time_minutes = []
        self.time_hours = []
        self.time_days = []

    def do_plots(self):
        self.time_seconds = [t for t in self.stat_tracker.time_seconds() if t <= 300]
        self.time_minutes = [t for t in self.stat_tracker.time_minutes() if t <= 300]
        self.time_hours = [t for t in self.stat_tracker.time_hours() if t <= 300]
        self.time_days = self.stat_tracker.time_days()

        self.plot_separated_demand()
        self.plot_demand_counts()
        self.plot_demand_costs()
        self.plot_separated_prod()
        self.plot_prod_counts()
        self.plot_prod_costs()
        self.plot_combined_income()
        self.plot_cumulative_income()
        self.plot_upgrade_factors()
        self.plot_upgrade_costs()
        plt.show()

    def plot_separated_demand(self):
        data = self.stat_tracker.single_demand_values
        y_label = 'Demand'
        self.plot_multiple_resources(data, y_label)

    def plot_demand_counts(self):
        data = self.stat_tracker.demand_counts
        y_label = 'Demand Resource Count'
        self.plot_multiple_resources(data, y_label)

    def plot_demand_costs(self):
        data = self.stat_tracker.demand_costs
        y_label = 'Demand Resource Cost'
        self.plot_multiple_resources(data, y_label)

    def plot_separated_prod(self):
        data = self.stat_tracker.single_prod_values
        y_label = 'Production'
        self.plot_multiple_resources(data, y_label)

    def plot_prod_counts(self):
        data = self.stat_tracker.prod_counts
        y_label = 'Prod Resource Count'
        self.plot_multiple_resources(data, y_label)

    def plot_prod_costs(self):
        data = self.stat_tracker.prod_costs
        y_label = 'Prod Resource Cost'
        self.plot_multiple_resources(data, y_label)

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

    def plot_upgrade_factors(self):
        data = self.stat_tracker.prod_single_upgrade_factors
        plt.figure()

        ylabel = 'Prod Single Upgrade Factors'
        plt.subplot(2, 2, 1)
        for i, d in enumerate(data):
            plt.plot(d, label='Resource {}'.format(i))
        plt.legend()
        plt.ylabel(ylabel)

        data = self.stat_tracker.prod_multi_upgrade_factors
        ylabel = 'Prod Multi Upgrade Factors'
        plt.subplot(2, 2, 2)
        plt.plot(data)
        plt.ylabel(ylabel)

        data = self.stat_tracker.demand_single_upgrade_factors
        ylabel = 'Demand Single Upgrade Factors'
        plt.subplot(2, 2, 3)
        for i, d in enumerate(data):
            plt.plot(d, label='Resource {}'.format(i))
        plt.legend()
        plt.ylabel(ylabel)

        data = self.stat_tracker.demand_multi_upgrade_factors
        ylabel = 'Demand Multi Upgrade Factors'
        plt.subplot(2, 2, 4)
        plt.plot(data)
        plt.ylabel(ylabel)

    def plot_upgrade_costs(self):
        data = self.stat_tracker.prod_single_upgrade_costs
        plt.figure()

        ylabel = 'Prod Single Upgrade Costs'
        plt.subplot(2, 2, 1)
        for i, d in enumerate(data):
            plt.semilogy(d, label='Resource {}'.format(i))
        plt.legend()
        plt.ylabel(ylabel)

        data = self.stat_tracker.prod_multi_upgrade_costs
        ylabel = 'Prod Multi Upgrade Costs'
        plt.subplot(2, 2, 2)
        plt.semilogy(data)
        plt.ylabel(ylabel)

        data = self.stat_tracker.demand_single_upgrade_costs
        ylabel = 'Demand Single Upgrade Costs'
        plt.subplot(2, 2, 3)
        for i, d in enumerate(data):
            plt.semilogy(d, label='Resource {}'.format(i))
        plt.legend()
        plt.ylabel(ylabel)

        data = self.stat_tracker.demand_multi_upgrade_costs
        ylabel = 'Demand Multi Upgrade Costs'
        plt.subplot(2, 2, 4)
        plt.semilogy(data)
        plt.ylabel(ylabel)

    def plot_multiple_resources(self, data, ylabel):
        plt.figure()

        plt.subplot(2, 2, 1)
        time = self.time_seconds
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel(ylabel)

        plt.subplot(2, 2, 2)
        time = self.time_minutes
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (minutes)")
        plt.ylabel(ylabel)

        plt.subplot(2, 2, 3)
        time = self.time_hours
        for i, d in enumerate(data):
            plt.semilogy(time, [x for i, x in enumerate(d) if i < len(time)], label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel(ylabel)

        plt.subplot(2, 2, 4)
        time = self.time_days
        for i, d in enumerate(data):
            plt.semilogy(time, d, label='Resource {}'.format(i))
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel(ylabel)