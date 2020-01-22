import numpy as np
import pandas as pd
import copy
import time
from Santas_Helpers.Schedule import Family, Schedule, CSchedule
from Santas_Helpers.Search import Search
from Santas_Helpers.Cost import CostMatrices
import matplotlib.pyplot as plt


def family_list_generate(filename):
    data = pd.read_csv(filename)
    family_list = []
    for itr in data.iterrows():
        itr_array = itr[1].values
        family = Family(itr_array[0], itr_array[11], itr_array[1:11])
        family_list.append(family)
    return family_list


def schedule_from_csv(filename, family_list):
    schedule_df = pd.read_csv(filename)
    test_schedule = Schedule(family_list)
    for row in schedule_df.iterrows():
        test_schedule.add_family(row[1]['family_id'], row[1]['assigned_day'])
    return test_schedule


def test_move_family(schedule):
    # picks a family at random from a valid schedule, and moves the family to a valid day
    # calculates the initial cost with a full calculation, and calculates the cost after moving
    # compares the costs from full calculations with the result of move_family() and self.acounting_cost and choice_cost
    days = np.arange(100) + 1
    families = np.arange(5000)
    days_above_min = [n_people > 135 for n_people in schedule.n_people_pairing[1:]]
    days_above_min = days[days_above_min]

    days_below_max = [n_people < 290 for n_people in schedule.n_people_pairing[1:]]
    days_below_max = days[days_below_max]

    remove_day = np.random.choice(days_above_min)
    add_day = np.random.choice(days_below_max)

    remove_day_families = schedule.family_day_pairing[:, remove_day]
    remove_day_families = families[remove_day_families]
    remove_day_family = np.random.choice(remove_day_families)

    d_choice, d_accounting = schedule.delta_move_cost(remove_day_family, remove_day, add_day)

    schedule.calculate_cost()
    print(
        f"{d_choice + d_accounting + schedule.schedule_cost}, {d_accounting + sum(schedule.accounting_cost)}, {d_choice + schedule.choice_cost}")

    print(f"{schedule.schedule_cost}, {sum(schedule.accounting_cost)}, {schedule.choice_cost}")

    schedule.move_family(remove_day_family, remove_day, add_day)
    print(f"{schedule.schedule_cost}, {sum(schedule.accounting_cost)}, {schedule.choice_cost}")

    schedule.calculate_cost()
    print(f"{schedule.schedule_cost}, {sum(schedule.accounting_cost)}, {schedule.choice_cost}")


def main():
    np.random.seed(23423)
    fdata = "/home/george/Documents/learning/kaggle_data/Santa_data/family_data.csv"
    family_list = family_list_generate(fdata)

    # load the schedule with score 86174
    schedule = schedule_from_csv("~/Downloads/submission.csv", family_list)
    # schedule = schedule_from_csv("~/Documents/learning/power_tools/SantasHelpers/out/testing_2277_83071.csv",
    #                              family_list)
    search_op = Search(schedule)
    search_op.schedule.calculate_cost()
    search_op.update_state('Valid')

    # search_op.circuit_move(5)

    # queue = search_op.circuit_move(steps=1)
    # print(len(queue))

    # get_cost = lambda x: x[1].h_cost
    # cost_list = list(map(get_cost, queue))
    # print(cost_list)

    # iterate through different gradient descent methods
    # for _ in range(25):
    #     search_op.greedy_sgd_1(steps=10**5)
    #     # in this format, sgd_3 makes a lot of progress
    #     search_op.greedy_sgd_3(steps=1000)
    #     search_op.sgd_choice(steps=1000)
    #     search_op.family_swapping(steps=1)

    for _ in range(100):
        cp = copy.deepcopy(search_op)
        cp.family_swapping(steps=1)
        if cp.schedule.schedule_cost < search_op.schedule.schedule_cost:
            search_op = cp

    print(f"Schedule Cost: {search_op.schedule.schedule_cost}")
    print(f"final accounting cost: {sum(search_op.schedule.accounting_cost)} final choice cost: "
          f"{search_op.schedule.choice_cost}")
    search_op.schedule.write_csv("out/testing")


if __name__ == '__main__':
    main()
