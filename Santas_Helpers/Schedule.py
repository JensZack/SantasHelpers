import numpy as np
import pandas as pd
import time
import copy
from Santas_Helpers.Cost import CostMatrices


class Family:
    def __init__(self, family_id, n_people, choices):
        self.family_id = family_id
        self.day = 0
        self.n_people = n_people
        self.choice_array = np.array(choices)
        self.current_choice = -1

    def update_choice(self):
        if self.day in self.choice_array:
            self.current_choice = np.where(self.choice_array == self.day)[0][0]

        else:
            self.current_choice = 10

    def get_choice_from_day(self, day):
        if day in self.choice_array:
            choice = np.where(self.choice_array == day)[0][0]

        else:
            choice = 10

        return choice


class Schedule:
    def __init__(self, family_list):
        # NEW STRUCTURE
        # schedule will contain a dictionary of family objects such that fam_dict[fam_id] = fam_obj
        # the actual schedule will be in the form of a matrix 101 x 5000 either containing n_people or a boolean
        # value for each [day, family] pair, with day 0 meaning the family is unassigned
        # This matrix can be used to replace family_id_bool, and maybe n_people_pairing, depending on performance delta
        #
        # still need to keep track of costs, and redefine calculations for delta cost
        # need to
        # -------------------------------------------------------------------------------------
        # stores family objects for the day they are scheduled
        # store the family id as the key, and day of pairing as the value

        # the order of the matrix is optimized to make it fastest to find a day given a family
        # the other option would be transposing the order to make it faster to sum over a given day to find n_people
        # the matrix is boolean to just hold true or false for whether a given family is on that day
        N_FAMILIES = len(family_list)
        N_DAYS = 101
        self.family_day_pairing = np.zeros((N_FAMILIES, N_DAYS), dtype=np.bool)
        self.family_day_pairing[:, 0] = np.ones(N_FAMILIES, dtype=np.bool)

        self.family_choice_cost = np.zeros(N_FAMILIES)

        # because the pairing only contains the relationship between family id and schedule day,
        # there is also a dict of families that is indexed by their family id
        self.family_dict = {family.family_id: family for family in family_list}

        # stores the sum total of people scheduled for a given day
        self.n_people_pairing = np.zeros(N_DAYS, dtype=np.int)
        self.n_people_pairing[0] = sum([fam.n_people for fam in family_list])

        self.accounting_cost = np.zeros(N_DAYS)
        self.choice_cost = 0
        self.schedule_cost = 0

        # loading the cost matrices into the schedule object once, to avoid recalculating
        self.costMatrices = CostMatrices()

    def day_of_week(self, day):
        return (100 - day) % 7

    def add_family(self, family_id, day):
        if not self.family_day_pairing[family_id, 0]:
            raise EnvironmentError('Family already added')
        # no updates to accounting cost because it can't be calculated until schedule is valid
        # all family id's start on day=0, so move to the new day
        self.family_dict[family_id].day = day
        self.family_dict[family_id].update_choice()

        fnp = self.family_dict[family_id].n_people

        self.family_day_pairing[family_id, 0] = False
        self.family_day_pairing[family_id, day] = True

        self.n_people_pairing[0] -= fnp
        self.n_people_pairing[day] += fnp

        current_choice_cost = self.costMatrices.get_choice_cost(fnp, self.family_dict[family_id].current_choice)
        self.family_choice_cost[family_id] = current_choice_cost
        self.choice_cost += current_choice_cost

    def is_valid(self):
        for idx, n in enumerate(self.n_people_pairing[1:]):
            if n > 300 or n < 125:
                return False
        return True

    def calculate_accounting_cost(self):
        # this function makes the full calculation on the accounting cost, not for updating
        if self.is_valid():
            accounting_function = lambda nt, nt1: self.costMatrices.get_accounting_cost(nt, nt1)
            accounting_function = np.vectorize(accounting_function)

            self.accounting_cost[1:] = accounting_function(self.n_people_pairing[1:],
                                                           np.concatenate([self.n_people_pairing[2:],
                                                                           np.array([self.n_people_pairing[-1]])],
                                                                          axis=0))
            return True
        else:
            return False

    def calculate_choice_cost(self):
        self.choice_cost = 0
        for fam in list(self.family_dict.values()):
            self.choice_cost += self.costMatrices.get_choice_cost(fam.n_people, fam.current_choice)

    def calculate_cost(self):
        self.schedule_cost = 0
        accounting_bool = self.calculate_accounting_cost()
        if not accounting_bool:
            raise ValueError("Accounting cost doesn't exist for current schedule")

        if accounting_bool:
            self.schedule_cost += np.sum(self.accounting_cost)

        self.calculate_choice_cost()
        self.schedule_cost += self.choice_cost

    def delta_accounting(self, n_people, day, new_day):
        # returns the full change in accounting cost for a given move
        # the method "calculate_acounting_cost" must have already been run
        # otherwise accounting cost will be predefined as 0

        if self.n_people_pairing[day] - n_people < 125 or self.n_people_pairing[new_day] + n_people > 300:
            return False
        accounting_function = lambda nt, nt1: self.costMatrices.get_accounting_cost(nt, nt1)
        accounting_function = np.vectorize(accounting_function)

        current_accounting = accounting_function(self.n_people_pairing[1:],
                                                 np.concatenate([self.n_people_pairing[2:],
                                                                np.array([self.n_people_pairing[-1]])],
                                                                axis=0))

        next_n_people = copy.deepcopy(self.n_people_pairing)
        next_n_people[day] -= n_people
        next_n_people[new_day] += n_people

        next_accounting = accounting_function(next_n_people[1:], np.concatenate([next_n_people[2:],
                                                                                 np.array([next_n_people[-1]])],
                                                                                axis=0))

        return sum(next_accounting) - sum(current_accounting)

    def delta_move_cost(self, family_id, current_day, new_day):
        # given a family, their currently scheduled day, and the day to move them to, return the change in cost fn
        # does not move the family, meant to be used to find possible next moves
        family = self.family_dict[family_id]
        delta_accounting = self.delta_accounting(family.n_people, current_day, new_day)

        choice_idx = np.where(family.choice_array == new_day)
        if choice_idx[0].size == 0:
            choice_idx = 10
        else:
            choice_idx = choice_idx[0][0]

        delta_choice = self.costMatrices.get_choice_cost(family.n_people, choice_idx) - \
            self.family_choice_cost[family.family_id]

        return delta_choice + delta_accounting

    def update_accounting(self, day):
        n_people_day_1 = min(max(125, self.n_people_pairing[day]), 300)
        if day < 100:
            n_people_day_2 = min(max(125, self.n_people_pairing[day+1]), 300)
        else:
            n_people_day_2 = n_people_day_1
        self.accounting_cost[day] = self.costMatrices.get_accounting_cost(n_people_day_1, n_people_day_2)

    def move_family(self, family_id, day, new_day):
        # this method needs to update: schedule cost, n_people_pairing, choice_cost, accounting_cost, family_pairing

        # n_people_pairing ----------------------------------------------------------------
        n_people = self.family_dict[family_id].n_people
        if self.n_people_pairing[new_day] + n_people <= 300 and self.n_people_pairing[day] - n_people >= 125:
            self.n_people_pairing[day] -= n_people
            self.n_people_pairing[new_day] += n_people

        else:
            raise ValueError(f"Can't move family from {day} to {new_day}")

        # choice cost --------------------------------------------------------------------
        self.family_dict[family_id].day = new_day

        prev_choice_cost = self.costMatrices.get_choice_cost(self.family_dict[family_id].n_people,
                                                             self.family_dict[family_id].current_choice)

        self.family_dict[family_id].update_choice()

        current_choice_cost = self.costMatrices.get_choice_cost(self.family_dict[family_id].n_people,
                                                                self.family_dict[family_id].current_choice)

        self.family_choice_cost[family_id] = current_choice_cost

        self.choice_cost += current_choice_cost - prev_choice_cost
        self.schedule_cost += current_choice_cost - prev_choice_cost

        # Accounting cost vector update ------------------------------------------------
        days_effected = np.unique(np.array([day-1, day, new_day-1, new_day]))
        prev_accounting_cost = np.sum(self.accounting_cost[days_effected])

        update_accounting = np.vectorize(self.update_accounting)
        update_accounting(days_effected)

        new_accounting_cost = np.sum(self.accounting_cost[days_effected])

        self.schedule_cost += new_accounting_cost - prev_accounting_cost

        # Family Pairing

        self.family_day_pairing[family_id, day] = False
        self.family_day_pairing[family_id, new_day] = True

    def swap_families(self, f1, f2):
        """
        Move family_1 to family_2.day and move family_2 to family_1.day
        :param f1:
        :param f2:
        :return:
        """
        # check if the swap is valid
        n1 = f1.n_people
        n2 = f2.n_people

        d1 = f1.day
        d2 = f2.day

        new_n1 = self.n_people_pairing[d1] + (n2 - n1)
        new_n2 = self.n_people_pairing[d2] + (n1 - n2)

        if max(new_n1, new_n2) > 300 or min(new_n1, new_n2) < 125:
            raise ValueError('Not a valid swap, exceeds daily bounds')

        # n people pairing
        self.n_people_pairing[d1] += (n2 - n1)
        self.n_people_pairing[d2] += (n1 - n2)

        # Update family objects and updating schedule choice and schedule cost
        self.family_dict[f1.family_id].day = d2
        self.family_dict[f2.family_id].day = d1

        f1_prev_choice_cost = self.family_choice_cost[f1.family_id]
        f2_prev_choice_cost = self.family_choice_cost[f2.family_id]

        self.family_dict[f1.family_id].update_choice()
        self.family_dict[f2.family_id].update_choice()

        f1_current_choice_cost = self.costMatrices.get_choice_cost(f1.n_people,
                                                               self.family_dict[f1.family_id].current_choice)
        f2_current_choice_cost = self.costMatrices.get_choice_cost(f2.n_people,
                                                               self.family_dict[f2.family_id].current_choice)

        self.family_choice_cost[f1.family_id] = f1_current_choice_cost
        self.family_choice_cost[f2.family_id] = f2_current_choice_cost

        self.choice_cost += f1_current_choice_cost + f2_current_choice_cost - \
        (f1_prev_choice_cost + f2_prev_choice_cost)

        self.schedule_cost += f1_current_choice_cost + f2_current_choice_cost - \
        (f1_prev_choice_cost + f2_prev_choice_cost)

        # Accounting cost vector
        days_effected = np.unique(np.array([d1-1, d1, d2-1, d2]))
        prev_accounting_cost = np.sum(self.accounting_cost[days_effected])

        update_accounting = np.vectorize(self.update_accounting)
        update_accounting(days_effected)

        new_accounting_cost = np.sum(self.accounting_cost[days_effected])
        self.schedule_cost += new_accounting_cost - prev_accounting_cost

        # Family Pairing

        self.family_day_pairing[f1.family_id, d1] = False
        self.family_day_pairing[f1.family_id, d2] = True

        self.family_day_pairing[f2.family_id, d2] = False
        self.family_day_pairing[f2.family_id, d1] = True

    def write_csv(self, file_out="test"):
        # use self.family_dict to make pandas df

        timestamp_filename = file_out + '_' + str(int(time.time()))[-4:] + '_' + str(int(self.schedule_cost)) + '.csv'

        assigned_days = [fam.day for fam in list(self.family_dict.values())]
        dict_out = {'family_id': list(self.family_dict.keys()), 'assigned_day': assigned_days}
        df_out = pd.DataFrame(dict_out)
        df_out.to_csv(timestamp_filename, index=False)


class CSchedule:
    def __init__(self, family_list):

        class CFamily(Family):

            def __init__(self, family_id, n_people, choices):
                Family.__init__(self, family_id, n_people, choices)
                self.current_density = np.zeros(101)
                self.cost_dot_array = np.zeros(101)

                costM = CostMatrices()
                for idx in (np.arange(100) + 1):
                    if idx in self.choice_array:
                        choice = np.where(self.choice_array == idx)[0][0]
                    else:
                        choice = 10

                    self.cost_dot_array[idx] = costM.get_choice_cost(self.n_people, choice)

            def update_choice(self):
                pass

            def update_density(self, delta):
                # should raise an error for sum(delta) != 0, but will leave alone for performance
                self.current_density = np.add(self.current_density, delta)

        N_FAMILIES = len(family_list)
        N_DAYS = 101

        self.family_density_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.float)
        self.choice_dot_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.float)
        self.costMatrices = CostMatrices()
        self.cost = 0

        self.c_familylist = []

        # not the most efficient, I would like to find a way to manipulate the list of Family objects into Family
        # objects instead of copying the child objects into  new list
        for idx, fam in enumerate(family_list):
            self.c_familylist.append(CFamily(fam.family_id, fam.n_people, fam.choice_array))
            self.choice_dot_matrix[idx, :] = self.c_familylist[idx].cost_dot_array

    def state_accounting_cost(self):
        accounting_fn = lambda nt1, diff: ((nt1 - 125)/400) * np.power(nt1, (.5 + abs(diff)/50))
        accounting_fn = np.vectorize(accounting_fn)

        n_vec = np.sum(self.family_density_matrix, axis=0)
        return n_vec

    def state_choice_cost(self):
        choice_cost = np.dot(self.family_density_matrix, self.choice_dot_matrix)
        return choice_cost

    def state_cost(self):
        self.cost = self.state_accounting_cost() + self.state_choice_cost()
        return self.cost

    def write_csv(self, filename):
        df_out = pd.Dataframe(columns=['family_id', 'assigned_day'])

        for idx, fam_density in enumerate(self.family_density_matrix):
            df_out.append([idx + 1, np.argmax(fam_density)], ignore_index=True)

        df_out.to_csv(path_or_buff=filename, index=False)
