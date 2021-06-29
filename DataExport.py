from DynamicBalanceDataLoader import *
from typing import List
import csv
import os


class DataExporter:

    def __init__(self):
        self.export_folder = 'export_data_files'

    def export_data(self, balancer):
        # type: (DynamicBalancer) -> None
        self.export_prod_list_upgrades(balancer)
        self.export_demand_list_upgrades(balancer)
        self.export_demand_single_auto_unlocks(balancer)
        self.export_demand_manual_unlocks(balancer)
        self.export_prod_single_unlocks(balancer)
        self.export_demand_multi_unlocks(balancer)
        self.export_prod_multi_unlocks(balancer)
        # self.export_battery_capacity_upgrades(balancer)
        self.export_battery_tier_upgrades(balancer)

    def export_prod_list_upgrades(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'production_upgrade_values.csv'
        path = os.path.join(self.export_folder, file_name)
        balance_data = self.generate_prod_list_upgrade_data(balancer)
        write_data = [['Cost', 'Factor', 'Upgradable Type', 'Upgradable Index', 'Upgrade Effect',
                       'Upgrade Name', 'Flavor Text']]
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            for i, data in enumerate(balance_data):
                # Want 6th and 7th position from existing file for upgrade name/flavor text.
                # keep it as empty string if it doesn't exist in the file.
                reader_data = next(reader, [None, None, None, None, None, 'Prod Upgrade {}'.format(i), ''])
                line_data = list()
                line_data.append('{:.6e}'.format(data[0]))  # cost
                line_data.append(data[1])  # factor
                line_data.append('powerSpinner')  # upgradable Type
                line_data.append(data[2])  # upgradable index
                line_data.append('multiplyGains')  # upgrade effect
                line_data.append(reader_data[5])  # upgrade name
                line_data.append(reader_data[6])  # flavor text
                write_data.append(line_data)
        self.write_csv_data(write_data, path)

    def add_prod_flavor_texts_to_final_document(self):
        file_name = 'production_upgrade_values.csv'
        path = os.path.join(self.export_folder, file_name)
        # First get the index data from the file.
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            indices = list()
            for row in reader:
                indices.append(int(row[3]))
                # Want 6th and 7th position from existing file for upgrade name/flavor text.
                # keep it as empty string if it doesn't exist in the file.
            names, flavor_texts = self.extract_prod_flavor_text_and_name_data(indices)

        with open(path, 'r', newline='') as f:  # open reader again to extract all the values and insert flavor text
            write_data = [['Cost', 'Factor', 'Upgradable Type', 'Upgradable Index', 'Upgrade Effect',
                           'Upgrade Name', 'Flavor Text']]
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            for i, flavor_data in enumerate(flavor_texts):
                # Want 6th and 7th position from existing file for upgrade name/flavor text.
                # keep it as empty string if it doesn't exist in the file.
                reader_data = next(reader)
                line_data = list()
                line_data.append(reader_data[0])  # cost
                line_data.append(reader_data[1])  # factor
                line_data.append(reader_data[2])  # upgradable Type
                line_data.append(reader_data[3])  # upgradable index
                line_data.append(reader_data[4])  # upgrade effect
                line_data.append(names[i])  # upgrade name
                line_data.append(flavor_texts[i])  # flavor text
                write_data.append(line_data)

        self.write_csv_data(write_data, path)

    def extract_prod_flavor_text_and_name_data(self, index_data):
        file_name = 'FlavorTextValues.xlsx'
        prod_sheet = 'prod upgrade names'
        flavor_text_index_tracker = dict()
        names = []
        flavor_texts = []
        wb = openpyxl.load_workbook(filename=file_name, data_only=True)
        ws = wb[prod_sheet]
        for i in index_data:
            name_position = 2*i+1
            flavor_text_position = 2*i + 2
            if i == -1:
                name_position = 13  # Magic numbers for multi upgrade names
                flavor_text_position = 14  # Magic numbers for multi upgrade flavor texts.
            if i not in flavor_text_index_tracker:
                flavor_text_index_tracker[i] = 3

            name = ws.cell(row=flavor_text_index_tracker[i], column=name_position).value
            flavor_text = ws.cell(row=flavor_text_index_tracker[i], column=flavor_text_position).value
            names.append(name)
            if flavor_text is None:
                flavor_text = ''
            else:
                flavor_text = str(flavor_text)
                flavor_text = flavor_text.replace('"', '')
                flavor_text = '"' + flavor_text + '"'
            flavor_texts.append(flavor_text)
            flavor_text_index_tracker[i] += 1
        return names, flavor_texts

    def export_demand_list_upgrades(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'power_seller_list_upgrade_values.csv'
        path = os.path.join(self.export_folder, file_name)
        balance_data = self.generate_demand_list_upgrade_data(balancer)
        write_data = [['Cost', 'Factor', 'Upgradable Type', 'Upgradable Index', 'Upgrade Effect',
                       'Upgrade Name', 'Flavor Text']]
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            for i, data in enumerate(balance_data):
                # Want 6th and 7th position from existing file for upgrade name/flavor text.
                # keep it as empty string if it doesn't exist in the file.
                reader_data = next(reader, [None, None, None, None, None, 'Demand Upgrade {}'.format(i), ''])
                line_data = list()
                line_data.append('{:.6e}'.format(data[0]))  # cost
                line_data.append(data[1])  # factor
                line_data.append(data[2])  # upgradable Type
                line_data.append(data[3])  # upgradable index
                line_data.append(data[4])  # upgrade effect
                line_data.append(reader_data[5])  # upgrade name
                line_data.append(reader_data[6])  # flavor text
                write_data.append(line_data)
        self.write_csv_data(write_data, path)

    def add_demand_flavor_texts_to_final_document(self):
        file_name = 'power_seller_list_upgrade_values.csv'
        path = os.path.join(self.export_folder, file_name)
        # First get the index data from the file.
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            indices = list()
            for row in reader:
                if str(row[2]) == 'manualPowerSeller':
                    indices.append(int(row[3]))
                elif int(row[3]) == -1:
                    indices.append(11)
                else:
                    indices.append(int(row[3]) + 1)

            names, flavor_texts = self.extract_demand_flavor_text_and_name_data(indices)

        with open(path, 'r', newline='') as f:  # open reader again to extract all the values and insert flavor text
            write_data = [['Cost', 'Factor', 'Upgradable Type', 'Upgradable Index', 'Upgrade Effect',
                           'Upgrade Name', 'Flavor Text']]
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            for i, flavor_data in enumerate(flavor_texts):
                # Want 6th and 7th position from existing file for upgrade name/flavor text.
                # keep it as empty string if it doesn't exist in the file.
                reader_data = next(reader)
                line_data = list()
                line_data.append(reader_data[0])  # cost
                line_data.append(reader_data[1])  # factor
                line_data.append(reader_data[2])  # upgradable Type
                line_data.append(reader_data[3])  # upgradable index
                line_data.append(reader_data[4])  # upgrade effect
                line_data.append(names[i])  # upgrade name
                line_data.append(flavor_texts[i])  # flavor text
                write_data.append(line_data)

        self.write_csv_data(write_data, path)

    def extract_demand_flavor_text_and_name_data(self, index_data):
        file_name = 'FlavorTextValues.xlsx'
        prod_sheet = 'demand upgrade names'
        flavor_text_index_tracker = dict()
        names = []
        flavor_texts = []
        wb = openpyxl.load_workbook(filename=file_name, data_only=True)
        ws = wb[prod_sheet]
        for i in index_data:
            name_position = 2 * i + 1
            flavor_text_position = 2 * i + 2
            if i not in flavor_text_index_tracker:
                flavor_text_index_tracker[i] = 3

            name = ws.cell(row=flavor_text_index_tracker[i], column=name_position).value
            flavor_text = ws.cell(row=flavor_text_index_tracker[i], column=flavor_text_position).value
            names.append(name)
            if flavor_text is None:
                flavor_text = ''
            else:
                flavor_text = str(flavor_text)
                flavor_text = flavor_text.replace('"', '')
                flavor_text = '"' + flavor_text + '"'
            flavor_texts.append(flavor_text)
            flavor_text_index_tracker[i] += 1
        return names, flavor_texts

    def export_demand_single_auto_unlocks(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'demand_auto_single_unlock_values.csv'
        path = os.path.join(self.export_folder, file_name)
        stats = balancer.stat_tracker
        factors = stats.demand_single_unlock_factors
        thresholds = balancer.demand_resources[1].unlock_thresholds[1:]  # index 1 because 0 is for manual resource.
        export_data = [['Threshold'] + [str(j - 1) for j in range(1, len(balancer.demand_resources))]]  # skip manual
        for i, threshold in enumerate(thresholds):
            next_line = [threshold]
            next_line = next_line + [factors[j][i] if i < len(factors[j]) else 2 for j in range(1, len(factors))]
            export_data.append(next_line)
        self.write_csv_data(export_data, path)

    def export_demand_manual_unlocks(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'demand_manual_single_unlock_values.csv'
        path = os.path.join(self.export_folder, file_name)
        stats = balancer.stat_tracker
        factors = stats.demand_single_unlock_factors[0]  # Manual unlock factors
        thresholds = balancer.demand_resources[0].unlock_thresholds[1:]  # Skip dummy threshold 0
        export_data = [['Threshold', 'Factor']]  # Header for csv file.
        for i, (threshold, factor) in enumerate(zip(thresholds, factors)):
            export_data.append([threshold, factor])
        self.write_csv_data(export_data, path)

    def export_prod_single_unlocks(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'production_single_unlock_values.csv'
        path = os.path.join(self.export_folder, file_name)
        stats = balancer.stat_tracker
        factors = stats.prod_single_unlock_factors
        thresholds = balancer.prod_resources[0].unlock_thresholds[1:]  # Skip first threshold (equals 0)
        export_data = [['Threshold'] + [str(j) for j in range(len(balancer.prod_resources))]]
        for i, threshold in enumerate(thresholds):
            next_line = [threshold]
            next_line = next_line + [factors[j][i] if i < len(factors[j]) else 2 for j in range(len(factors))]
            export_data.append(next_line)
        self.write_csv_data(export_data, path)

    def export_prod_multi_unlocks(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'production_multi_unlock_values.csv'
        path = os.path.join(self.export_folder, file_name)
        stats = balancer.stat_tracker
        factors = stats.prod_multi_unlock_factors
        thresholds = balancer.prod_multi_unlock_thresholds[1:]
        export_data = [['Threshold', 'Factor']]  # Header for csv file.
        for i, (threshold, factor) in enumerate(zip(thresholds, factors)):
            export_data.append([threshold, factor])
        self.write_csv_data(export_data, path)

    def export_demand_multi_unlocks(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'demand_multi_unlock_values.csv'
        path = os.path.join(self.export_folder, file_name)
        stats = balancer.stat_tracker
        factors = stats.demand_multi_unlock_factors
        thresholds = balancer.demand_multi_unlock_thresholds[1:]
        export_data = [['Threshold', 'Factor']]  # Header for csv file.
        for i, (threshold, factor) in enumerate(zip(thresholds, factors)):
            export_data.append([threshold, factor])
        self.write_csv_data(export_data, path)

    def export_battery_capacity_upgrades(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'battery_capacity_upgrades.csv'
        battery_generator = balancer.battery_generator
        costs = balancer.stat_tracker.battery_upgrade_tier_costs
        # factors = battery_generator.factors
        path = os.path.join(self.export_folder, file_name)
        write_data = [['Cost', 'Factor', 'Name', 'Flavor Text']]
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            for i, cost in enumerate(costs):
                reader_data = next(reader, [None, None, 'Capacity Upgrade {}'.format(i), ''])
                line_data = list()
                line_data.append('{:.6e}'.format(costs[i]))  # cost
                line_data.append(1)  # factor
                line_data.append(reader_data[2])  # upgrade name
                line_data.append(reader_data[3])  # flavor text
                write_data.append(line_data)
        self.write_csv_data(write_data, path)

    def export_battery_tier_upgrades(self, balancer):
        # type: (DynamicBalancer) -> None
        file_name = 'battery_tier_upgrades.csv'
        battery_generator = balancer.battery_generator
        costs = balancer.stat_tracker.battery_upgrade_tier_costs
        # capacities = battery_generator.individual_capacities
        synthesis_energies = balancer.stat_tracker.battery_synthesis_tier_energies

        path = os.path.join(self.export_folder, file_name)
        write_data = [['Cost', 'Storage Capacity', 'Energy to Synthesize', 'Battery Name', 'Flavor Text']]
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # get rid of the headers.
            write_data.append(next(reader))  # Add in the first "0 value" starting battery.
            for i, cost in enumerate(costs):
                reader_data = next(reader, [None, None, None, 'Synthesis Upgrade {}'.format(i), ''])
                line_data = list()
                line_data.append('{:.6e}'.format(costs[i]))  # cost
                line_data.append(0)  # Battery Capacity
                line_data.append(synthesis_energies[i])  # Energy to synthesize
                line_data.append(reader_data[3])  # upgrade name
                line_data.append(reader_data[4])  # flavor text
                write_data.append(line_data)
        self.write_csv_data(write_data, path)

    @staticmethod
    def write_csv_data(write_data, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for wd in write_data:
                writer.writerow(wd)

    @staticmethod
    def generate_prod_list_upgrade_data(balancer):
        # type: (DynamicBalancer) -> List[List]
        data = []
        stats = balancer.stat_tracker
        for i, (costs, factors) in enumerate(zip(stats.prod_single_upgrade_costs, stats.prod_single_upgrade_factors)):
            data = data + [(c, f, i) for c, f in zip(costs, factors) if c > 0]
        costs = stats.prod_multi_upgrade_costs
        factors = stats.prod_multi_upgrade_factors
        data = data + [(c, f, -1) for c, f in zip(costs, factors) if c > 0]
        data.sort(key=lambda x: x[0])
        return data

    @staticmethod
    def generate_demand_list_upgrade_data(balancer):
        # type: (DynamicBalancer) -> List[List]
        data = []
        stats = balancer.stat_tracker
        for i, (costs, factors) in enumerate(zip(stats.demand_single_upgrade_costs,
                                                 stats.demand_single_upgrade_factors)):
            if i == 0:
                data = data + [(c, f, 'manualPowerSeller', 0, 'multiplyGains') for c, f in zip(costs, factors) if c > 0]
            else:  # These are the auto demand resources. Just subtract the index by 1 and we are good.
                data = data + [(c, f, 'autoDemandResource', i - 1, 'multiplyGains')
                               for c, f in zip(costs, factors) if c > 0]
        # Multi demand upgrades:
        costs = stats.demand_multi_upgrade_costs
        factors = stats.demand_multi_upgrade_factors
        data = data + [(c, f, 'autoDemandResource', -1, 'multiplyGains') for c, f in zip(costs, factors) if c > 0]
        # Power value upgrades (which get lumped into the demand list upgrades sector)
        # costs = stats.power_value_upgrade_costs
        # factors = stats.power_value_upgrade_factors
        # data = data + [(c, f, 'gameController', 0, 'multiplyRevenuePerPower') for c, f in zip(costs, factors) if c > 0]
        data.sort(key=lambda x: x[0])
        return data


if __name__ == '__main__':
    pass
