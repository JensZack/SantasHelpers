import numpy as np
import pandas as pd
import time
import copy
from Santas_Helpers.Schedule import Family, Schedule, CSchedule
import matplotlib.pyplot as plt


class Search:
    def __init__(self, schedule, state='Initial'):
        self.N_DAYS = 100
        self.N_FAMILIES = 5000
        self.schedule = schedule

        """
        I want to define state to only have one of the possible state descriptions 
            Initial: At least on family still assigned "day 0"
            Assigned: No Families on "day 0"
            Valid_min: every day has 125+ people assigned
            Valid_max: every day has 300- people assigned
            Valid: every day is between 125 and 300 people assigned
        """

        self._state_dict = {'Initial': 'At least on family still assigned "day 0"',
                           'Assigned': 'No Families on "day 0"',
                           'Valid_min': 'Every day has 125+ people assigned',
                           'Valid_max': 'Every day has 300- people assigned',
                           'Valid': 'Every day has between 125 and 300 people assigned'}

        self._state = list(self._state_dict.keys())[0]

        self._inverse_day_density = np.zeros(self.N_FAMILIES + 1)
        self._day_density = np.zeros(self.N_DAYS + 1)

        # MAKE THE CODE FOR THE INVERSE CHOICE MATRIX
        # Dimensions 9, 10 with the first 2rows being offset to line up with n_people
        # should sum to 0 for any column, need to decide if it contains the inverse cost for no assignment
        # needs to be close to one for choice 1, and close to zero for choice 10
        self.pInvChoiceMatrix = -1 * np.ones((9, 10))

        lam = 10
        for n_people in np.arange(2, 9):
            choice_array = self.schedule.costMatrices.get_choice_array(n_people)[:10]
            inv_choice_array = 1 / (choice_array + lam)
            self.pInvChoiceMatrix[n_people, :] = inv_choice_array

        # ----------------adjustable functions that are used throughout the search process---------------------------
        self.day_density_heuristic = lambda n_p, choice: n_p * np.power(.9, choice)
        # weights the demand for a day based on the number of families that want that day, and how high of a choice
        # that day is

        self.initial_queue_function = lambda choice_cost, day_density: np.square(np.dot(choice_cost, day_density))
        # choice cost is a vector with the cost of putting a family on their 0-9th choice
        # day density is a vector with the density of families that want to be on the days of the families 10 choices
        # the currently described function is using the distance function for density vector and choice vector
        """
        The initial queue function is trying to order families from most likely to be expensive to place to lease
        expensive to place, the current function doesn't take into account the resources used by a family determined
        by the number of people in the family
        """
        # returns the probability vector of being assigned choice n
        mu = 10
        self.rand_fam_assignments = lambda carr, iday: (iday / (carr + mu)) / sum(iday / (carr + mu))

        # -----------------------------------------------------------------------------------------------------------

    def day_density(self, inverse=False):
        """
        returns the heuristic version of the demand or density for each day,
        has the option to return the inverse density which is basically the expected availability for a given day
        :param inverse:
        :return: a numpy array of size N_DAYS + 1 where the index is the day it stores, out[0] should always be 0
        """

        if np.all(self._day_density == 0):
            # if day density hasn't been calculated yet, calculate it now and store in the class
            self._day_density = np.zeros(self.N_DAYS + 1)
            for fam in list(self.schedule.family_dict.values()):
                n_people = fam.n_people

                for choice, day in enumerate(fam.choice_array):
                    self._day_density[day] += self.day_density_heuristic(n_people, choice)

            self._day_density = self._day_density / sum(self._day_density)

        if inverse:

            if np.all(self._inverse_day_density == 0):
                # if the inverse day density hasn't been calculated, do so now and store in the class
                self._inverse_day_density = np.zeros(self.N_DAYS + 1)
                self._inverse_day_density[1:] = 1 / self._day_density[1:]
                self._inverse_day_density = self._inverse_day_density / sum(self._inverse_day_density)

            return self._inverse_day_density

        return self._day_density

    def update_state(self, new_state):
        """
        takes new state in the form int[0:4] or string from state_dict keys
        :param new_state:
        :return:
        """
        if isinstance(new_state, int):
            if new_state in np.arange(len(self._state_dict.keys())):
                self._state = list(self._state_dict.keys())[new_state]
            else:
                raise ValueError(f"{new_state} is out of bounds for the state dict")

        if isinstance(new_state, str):
            if new_state in list(self._state_dict.keys()):
                self._state = new_state
            else:
                raise ValueError(f"{new_state} is not a valid state")

    def initial_family_queue(self):
        """
        take self.schedule.family_dict to return a family queue list, there's no point in
        storing this list as an attribute in the class
        :return:
        """
        family_list = list(self.schedule.family_dict.values())
        n = len(family_list)
        queue_probability = np.zeros(n)

        day_density = self.day_density()
        cost_array_dict = {}
        for idx, family in enumerate(family_list):
            if family.n_people not in list(cost_array_dict.keys()):
                cost_array_dict[family.n_people] = self.schedule.costMatrices.get_choice_array(family.n_people)[:10]

            queue_probability[idx] = self.initial_queue_function(cost_array_dict[family.n_people],
                                                                 day_density[family.choice_array])

        queue_probability = queue_probability / sum(queue_probability)
        family_queue = np.random.choice(n, n, p=queue_probability, replace=False)
        family_queue = [family_list[idx] for idx in family_queue]

        return family_queue

    def initial_to_valid_max(self, family_queue=None):
        i_day_density = self.day_density(inverse=True)

        # storing the n_people, choice costs with an offset of 2 to make np_cost_matrix[n_people, :] = choice_array
        np_cost_matrix = np.concatenate((-1 * np.ones((2, 11)), self.schedule.costMatrices.ChoiceMatrix), axis=0)

        if not family_queue:
            family_queue = list(self.schedule.family_dict.values())

        for fam in family_queue:
            p_fam_assignments = self.rand_fam_assignments(np_cost_matrix[fam.n_people, :10],
                                                          i_day_density[fam.choice_array[:10]])

            fam_assignments = np.random.choice(fam.choice_array, 10, replace=False, p=p_fam_assignments)

            family_added = False
            # try to add family to a preferred day
            for assignment in fam_assignments:
                if self.schedule.n_people_pairing[assignment] < 300 - fam.n_people:
                    family_added = True
                    self.schedule.add_family(fam.family_id, assignment)
                    break

            # if family can't be added to preferred day, then add to a day with low density
            while not family_added:
                best_days = np.random.choice(np.arange(100) + 1, 20, replace=False, p=i_day_density[1:])
                for day in best_days:
                    if self.schedule.n_people_pairing[day] < 300 - fam.n_people:
                        family_added = True
                        self.schedule.add_family(fam.family_id, day)
                        break

        self.update_state("Valid_max")

    def valid_max_to_valid(self):
        days = np.arange(100) + 1
        days_under_capacity = [self.schedule.n_people_pairing[day] < 125 for day in days]
        days_under_capacity = days[days_under_capacity]
        days_above_min = [day not in days_under_capacity for day in days]
        days_above_min = days[days_above_min]

        for day in days_under_capacity:
            while self.schedule.n_people_pairing[day] < 125:
                remove_day = np.random.choice(days_above_min)
                family_id = np.random.choice(np.where(self.schedule.family_day_pairing[:, remove_day])[0])

                temp_n_people = self.schedule.family_dict[family_id].n_people
                if (self.schedule.n_people_pairing[remove_day] - temp_n_people) > 125:
                    try:
                        self.schedule.move_family(family_id, remove_day, day)
                    except ValueError:
                        pass

        self.update_state('Valid')

    def initial_to_random_valid_state(self):
        """
        Takes the current initial state, and finds a random valid state.
        :heuristic: a heuristic function to score a family for order of insertion
        :return:
        """

        family_queue = self.initial_family_queue()
        self.initial_to_valid_max()
        self.valid_max_to_valid()

    def greedy_sgd_1(self, steps=10000):
        """
        Implementing a "greedy" SGD, moving one family at a time.
        Set cutoffs of time and steps to pass into the method

        I want this sgd to focus on optimizing accounting costs, as they seem to be the largest factor in cost

        Search Algorithm:

        Using self.schedule.accounting_cost, pick days with high cost

        Randomly pick a day with large accounting cost
            Compare this day to the n+1 day, and now choose the day with the most people
            Pick a family from this day with a high choice cost
            Move this family to a day using a heuristic of choice cost and day with day with low number of people

        :return:
        """
        if self._state != 'Valid':
            print('must start in a valid state')
            return False

        # initialize by calculating the valid starting cost
        self.schedule.calculate_cost()

        # variables that need to live just in the sgd algorithm
        days = np.arange(100) + 1
        family_ids = np.arange(5000)

        # parameter for weighting the choice cost
        mu = 10

        choose_day = lambda accounting_vector: np.power(accounting_vector, 3)
        check_delta_cost = np.vectorize(self.schedule.delta_move_cost)
        # ----------------------------------------------------

        ts = time.time()
        for step in range(steps):
            # pick a day with high accounting cost
            choose_day_p = np.sqrt(self.schedule.accounting_cost[1:])
            choose_day_p = choose_day_p / sum(choose_day_p)
            move_day = np.random.choice(days, p=choose_day_p)

            # make a list of families currently assigned to the move day
            families = self.schedule.family_day_pairing[:, move_day]
            families = family_ids[families]

            # list the families choice cost, and pick a family with a high choice cost
            family_choice_cost = np.array([x + 0.1 for x in self.schedule.family_choice_cost[families]])
            if len(family_choice_cost) == 0:
                break
            p_family_choice = family_choice_cost / sum(family_choice_cost)
            move_family_id = np.random.choice(families, p=p_family_choice)

            # go through the queue of families with high choice cost until you find one that can move
            # and improve the overall schedule cost
            open_days = days[self.schedule.n_people_pairing[1:] + self.schedule.family_dict[move_family_id].n_people < 300]
            new_day_p = 300 - self.schedule.n_people_pairing[open_days]
            new_day_p = new_day_p / sum(new_day_p)
            new_day = np.random.choice(open_days, p=new_day_p)
            move_cost = self.schedule.delta_move_cost(move_family_id, move_day, new_day)
            if move_cost < 0:
                try:
                    self.schedule.move_family(move_family_id, move_day, new_day)
                except ValueError:
                    pass

            if step % 10 == 0:
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} taken in {time.time() - ts} seconds")
        self.schedule.calculate_cost()
        print(f"final cost: {self.schedule.schedule_cost}")

    def fdf1(self):
        """
        family distribution function 1
        :return:
        """

        alpha = 1
        beta = 1
        sigma = .05

        fc = self.schedule.family_choice_cost
        ac = np.dot(self.schedule.family_day_pairing, self.schedule.accounting_cost)
        dif_day = self.schedule.n_people_pairing[1:] - np.append(self.schedule.n_people_pairing[2:],
                                                                 self.schedule.n_people_pairing[-1])

        dif = np.dot(self.schedule.family_day_pairing[:, 1:], dif_day)

        h = (alpha * fc) + (beta * ac) + np.exp(sigma * dif)
        h = h / sum(h)

        return h

    def ddf(self, family_id):

        alpha = 300
        beta = 1

        dd = self.day_density()[1:]
        fam_choice_array = self.schedule.family_dict[family_id].choice_array
        choice_cost_array = self.schedule.costMatrices.get_choice_array(self.schedule.family_dict[family_id].n_people)
        cc = choice_cost_array[-1] * np.ones(100)
        cc[fam_choice_array - 1] = choice_cost_array[:-1]

        inv_day_p = (alpha * dd) + cc
        day_p = 1 / inv_day_p
        day_p = day_p / sum(day_p)

        return day_p

    def greedy_sgd_generic(self, steps=1000, family_dist_func=fdf1, day_dist_func=ddf):
        """
        1. make probability distribution for picking a family
            if the loop comes back and the family hasn't been moved, reduce that families probability of being moved
        2. make a probability distribution for the new day for that family
        3. if the random new day chosen has a lower cost, move
        4. retry moving to a new day if less than 75% of the total new day probability has been explored
        :return:
        """
        families = np.arange(5000)
        days = np.arange(100) + 1
        prev_failed = False
        family_id = 0

        ts = time.time()
        for step in range(steps):
            family_p = family_dist_func(self)

            if prev_failed:
                family_p[family_id] = 0
                family_p = family_p / sum(family_p)

            family_id = np.random.choice(families, p=family_p)
            current_day = np.where(self.schedule.family_day_pairing[family_id, :])[0][0]

            day_p = day_dist_func(self, family_id)
            new_day = np.random.choice(days, p=day_p)

            # keep track of whether or not a family has been moved
            # keep track of the total day probability explored
            # keep track of whether or not the family insertion fails
            moved = False
            prev_failed = True
            day_p_static = copy.deepcopy(day_p)
            total_day_p = 0

            while not moved:
                if self.schedule.delta_move_cost(family_id, current_day, new_day) < 0:
                    self.schedule.move_family(family_id, current_day, new_day)
                    moved = True
                    prev_failed = False
                    break

                else:
                    total_day_p += day_p_static[new_day-1]
                    day_p[new_day-1] = 0
                    day_p = day_p / sum(day_p)

                if total_day_p > .75:
                    break

            if step % 10 == 0:
                print(f"{self.schedule.schedule_cost} cost after {step} steps, in a time of {time.time() - ts}")

    def greedy_gd_3(self, steps=1000):
        """
        Implementing a "greedy" GD, moving one family at a time.
        Does not implement any randomness, just slow moves
        Set cutoffs of time and steps to pass into the method

        gd_3 should focus on making smart, slow moves to optimize choice cost

        Search Algorithm:

        1. make a family queue of tuples with choice cost and family id
        2. sort this queue by cost
        3. take the family with the largest choice cost and attempt to move them
        4. make a random choice of days and check all of their delta move costs
        5. if the best of these days has a negative delta cost, move the family
        6. insert the family back into the correct position in the queue
        7. if the family fails to get moved, go to the second highest cost family in the queue

        :return:
        """
        if self._state != 'Valid':
            print('must start in a valid state')
            return False

        # initialize by calculating the valid starting cost
        self.schedule.calculate_cost()

        # variables that need to live just in the sgd algorithm -------------------
        days = np.arange(100) + 1
        family_ids = np.arange(5000)

        choose_day = lambda accounting_vector: np.power(accounting_vector, 3)
        check_delta_cost = np.vectorize(self.schedule.delta_move_cost)

        # function to re-insert families into the queue
        def recursive_find_slot(sl, key):
            total_len = len(sl)
            mid = int(total_len/2)
            if sl[mid] == key or total_len <= 2:
                return mid
            if sl[mid] > key:
                return mid + recursive_find_slot(sl[:mid], key)
            if sl[mid] < key:
                return recursive_find_slot(sl[mid+1:], key)

        # ----------------------------------------------------------------------------

        # initialize family choice cost queue
        family_choice_queue = []
        for family_id in family_ids:
            family_choice_queue.append((family_id, self.schedule.family_choice_cost[family_id]))

        cost_sort = lambda x: x[1]
        family_choice_queue = sorted(family_choice_queue, key=cost_sort, reverse=True)

        ts = time.time()
        for step in range(steps):

            step_completed = False
            failed_family_queue = []
            while not step_completed:
                if not family_choice_queue:
                    print('no possible moves')
                    break

                move_family_id, move_family_cost = family_choice_queue.pop(0)
                move_family = self.schedule.family_dict[move_family_id]
                move_day = move_family.day

                if self.schedule.n_people_pairing[move_day] - move_family.n_people < 125:
                    pass

                else:
                    available_days = [npp + move_family.n_people <= 300 for npp in self.schedule.n_people_pairing[1:]]
                    available_days = days[available_days]

                    for choice in move_family.choice_array:
                        if choice in available_days:
                            move_cost = self.schedule.delta_move_cost(move_family_id, move_day, choice)
                            if move_cost < 0:
                                self.schedule.move_family(move_family_id, move_day, choice)
                                new_choice_cost = self.schedule.family_choice_cost[move_family_id]
                                step_completed = True
                                break

                if not step_completed:
                    failed_family_queue.append((move_family_id, move_family_cost))

            family_choice_queue = failed_family_queue + family_choice_queue

            if step_completed:
                cost_queue = list(map(cost_sort, family_choice_queue))
                family_choice_queue.insert(recursive_find_slot(cost_queue, new_choice_cost),
                                           (move_family_id, new_choice_cost))

            if step % 10 == 0:
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} steps taken in {time.time() - ts} seconds")
        self.schedule.calculate_cost()
        print(f"final cost: {self.schedule.schedule_cost}")

    def greedy_sgd_3(self, steps=10000):
        """
        a random version of greedy_gd_3
        :param steps:
        :return:
        """

        if self._state != 'Valid':
            print('must start in a valid state')
            return False

        # initialize by calculating the valid starting cost
        self.schedule.calculate_cost()

        # variables that need to live just in the sgd algorithm -------------------
        days = np.arange(100) + 1
        family_ids = np.arange(5000)

        choose_day = lambda accounting_vector: np.power(accounting_vector, 3)
        check_delta_cost = np.vectorize(self.schedule.delta_move_cost)

        # function to re-insert families into the queue
        def recursive_find_slot(sl, key):
            total_len = len(sl)
            mid = int(total_len / 2)
            if sl[mid] == key or total_len <= 2:
                return mid
            if sl[mid] > key:
                return mid + recursive_find_slot(sl[:mid], key)
            if sl[mid] < key:
                return recursive_find_slot(sl[mid + 1:], key)

        # ----------------------------------------------------------------------------

        # initialize family choice cost queue
        family_choice_queue = []
        for family_id in family_ids:
            family_choice_queue.append((family_id, self.schedule.family_choice_cost[family_id]))

        cost_sort = lambda x: x[1]
        # family_choice_queue = sorted(family_choice_queue, key=cost_sort, reverse=True)

        ts = time.time()
        for step in range(steps):

            # pick a family at random from the choice queue
            cost_queue = np.array(list(map(cost_sort, family_choice_queue)))
            family_p = cost_queue / sum(cost_queue)
            move_family = np.random.choice(np.arange(len(family_choice_queue)), p=family_p)
            move_family = family_choice_queue[move_family]
            move_family_id = move_family[0]
            move_family = self.schedule.family_dict[move_family_id]

            for choice in move_family.choice_array:
                if self.schedule.delta_move_cost(move_family_id, move_family.day, choice) < 0:
                    try:
                        self.schedule.move_family(move_family_id, move_family.day, choice)
                        new_choice_cost = self.schedule.family_choice_cost[move_family_id]

                        family_choice_queue.append((move_family_id, new_choice_cost))
                    except ValueError:
                        pass

            if step % 10 == 0:
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} steps taken in {time.time() - ts} seconds")
        self.schedule.calculate_cost()
        print(f"final cost: {self.schedule.schedule_cost}")

    def family_swapping(self, steps=10):
        """
        Looking to swap families to reduce choice cost that can't be easily overcome in greedy sgd
        not a greedy algorithm

        Algorithm:
        1. make a queue of families, and their current choice cost,
        2. select the first family in the queue, and search the remaining queue in order
            for another family trying to move to the first families location
            ???maybe??? only swap families if it reduces global choice cost, making the algo greedy ???
        3. once a pairing is found, swap the families and remove them both from the list

        :param steps:
        :return:
        """

        if self._state != 'Valid':
            print('must start in a valid state')
            return False

        # initialize by calculating the valid starting cost
        self.schedule.calculate_cost()

        # variables that need to live just in the sgd algorithm -------------------
        days = np.arange(100) + 1
        family_ids = np.arange(5000)

        ts = time.time()
        report_time = time.time()
        for step in range(steps):
            # initialize family choice cost queue
            family_choice_queue = []
            for family_id in family_ids:
                family_choice_queue.append((family_id, self.schedule.family_choice_cost[family_id]))

            cost_sort = lambda x: x[1]
            family_choice_queue = sorted(family_choice_queue, key=cost_sort, reverse=True)

            while family_choice_queue:
                family = family_choice_queue.pop(0)
                family_obj = self.schedule.family_dict[family[0]]
                choice_days = family_obj.choice_array

                for idx, swap_family in enumerate(family_choice_queue):
                    swap_family_obj = self.schedule.family_dict[swap_family[0]]

                    if swap_family_obj.day in choice_days:
                        no_swap_cost = self.schedule.family_choice_cost[family[0]] + \
                                       self.schedule.family_choice_cost[swap_family[0]]

                        family_choice = family_obj.get_choice_from_day(swap_family_obj.day)
                        swap_family_choice = swap_family_obj.get_choice_from_day(family_obj.day)

                        swap_cost = self.schedule.costMatrices.get_choice_cost(family_obj.n_people, family_choice) + \
                            self.schedule.costMatrices.get_choice_cost(swap_family_obj.n_people, swap_family_choice)

                        if swap_cost < no_swap_cost:
                            try:
                                self.schedule.swap_families(family_obj, swap_family_obj)
                                family_choice_queue.pop(idx)

                            except ValueError:
                                pass

            if time.time() - report_time >= 5:
                report_time = time.time()
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} steps taken in {time.time() - ts} seconds")
        print(f"final cost: {self.schedule.schedule_cost}")
        print(f"final accounting cost: {sum(self.schedule.accounting_cost)} final choice cost "
              f"{self.schedule.choice_cost}")

    def sgd_choice(self, steps=1000):
        """
        greedy_sgd_1 is really good at optimizing accounting cost, I need to optimize choice cost without considering
        accounting cost, because the activation energy of accounting cost is too much to traverse the space efficiently

        This algorithm will randomly select families with high choice costs, and move them to days that reduce their
        cost, I might bound the change in accounting cost to be less than some small positive number, maybe as a
        learning rate?
        :param steps:
        :return:
        """

        # local variables to track algorithm -----------------
        misses = 0
        tries = 0

        def choice_heuristic(search, family_id, cur_day, n_day, stp):
            # for the basic heuristic, I will move the family if delta_choice + .1 * delta_accounting > 0
            par_1 = .9  # parameter 1, the variable used to determine the heuristic
            family_obj = search.schedule.family_dict[family_id]
            delta_accounting = search.schedule.delta_accounting(family_obj.n_people, cur_day, n_day)

            delta_choice = search.schedule.costMatrices.get_choice_cost(family_obj.n_people,
                                                                        family_obj.get_choice_from_day(n_day)) - \
                search.schedule.family_choice_cost[family_id]

            return delta_choice + (par_1 * delta_accounting)
        # -----------------------------------------------------

        # start timing
        ts = time.time()

        family_choice_queue = []
        for family in list(self.schedule.family_dict.values()):
            family_choice_queue.append((family.family_id, family.current_choice))

        cost_sort = lambda x: x[1]
        family_choice_queue = sorted(family_choice_queue, key=cost_sort, reverse=True)

        # take the cost of the families currently to select a distribution of indices to select from
        # this should speed up the process of choosing families while retaining the general distribution of costs
        family_cost_p = np.array(list(map(cost_sort, family_choice_queue)))
        family_cost_p = family_cost_p / sum(family_cost_p)
        static_queue_steps = np.random.choice(np.arange(len(family_choice_queue)), size=steps, p=family_cost_p)

        for step in range(steps):
            move_family_id, move_family_choice_cost = family_choice_queue.pop(static_queue_steps[step])
            choice_days = self.schedule.family_dict[move_family_id].choice_array

            current_choice_idx = self.schedule.family_dict[move_family_id].current_choice
            if current_choice_idx == 0:
                continue

            new_day_p = np.square(current_choice_idx - np.arange(current_choice_idx))
            new_day_p = new_day_p / sum(new_day_p)
            new_choice_day = np.random.choice(np.arange(current_choice_idx), p=new_day_p)

            current_day = self.schedule.family_dict[move_family_id].day
            new_day = choice_days[new_choice_day]

            if choice_heuristic(self, move_family_id, current_day, new_day, step) < 0:
                tries += 1
                try:
                    self.schedule.move_family(move_family_id, current_day, new_day)
                except ValueError:
                    misses += 1

            if step % 20 == 0:
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} steps taken in {time.time() - ts} seconds and {misses} misses and {tries} tries")
        print(f"final cost: {self.schedule.schedule_cost}")
        print(f"final accounting cost: {sum(self.schedule.accounting_cost)} final choice cost "
              f"{self.schedule.choice_cost}")

    def family_swapping_2(self, steps=1000):
        """
        define the matrix S, which contains the sum of choice cost to be reduced from day i to day j at S_i,j
        pick a day in S with probability corresponding to the magnitude of S_i,j
        :param steps:
        :return:
        """
        # parameters for the algorithm ------------------------------
        N_CHANNELS = 50

        # -----------------------------------------------------------

        # a dictionary storing families associated with channels, using the flattened index of the channel as a key
        # each value in the dict is a touple containing (delta_cost, family_id)

        steps = int(steps / N_CHANNELS)

        ts = time.time()
        for step in range(steps):
            S_dict = {}
            S = np.zeros((101, 101), dtype=np.float32)
            for family in list(self.schedule.family_dict.values()):
                family_day = family.day
                better_days = family.choice_array[0: family.current_choice]
                delta_cost = self.schedule.family_choice_cost[family.family_id] - \
                             self.schedule.costMatrices.get_choice_array(family.n_people)[0: family.current_choice]

                for idx, d_c in enumerate(delta_cost):
                    S[family_day, better_days[idx]] += d_c

                    if d_c > 0:
                        # S_dict will be used to easily lookup families that are being moved, so no need to store families
                        # on channels that have no improved cost
                        try:
                            S_dict[np.ravel_multi_index((family_day, better_days[idx]), (101, 101))]\
                                .append((d_c, family.family_id))
                        except KeyError:
                            S_dict[np.ravel_multi_index((family_day, better_days[idx]), (101, 101))] = []
                            S_dict[np.ravel_multi_index((family_day, better_days[idx]), (101, 101))].append((d_c, family.family_id))

            P = S / np.sum(S)
            np.nan_to_num(P, False)
            channels = np.random.choice(np.arange(101*101), size=N_CHANNELS, p=P.flatten())

            for channel in channels:
                get_dc = lambda x: x[0]
                sorted(S_dict[channel], key=get_dc, reverse=True)
                dc, max_family_id = S_dict[channel].pop(0)
                channel = np.ravel_multi_index(channel, (101, 101))
                inverse_channel = np.array((channel[1], channel[0]))
                if S[inverse_channel] == 0:
                    # check if move family is a good option
                    delta_sc = self.schedule.delta_move_cost(max_family_id, channel[0], channel[1])
                    move_p = 1 if delta_sc < 5 else (1/(5 * np.log(delta_sc)))

                    # decide whether or not to move
                    if bool(np.random.binomial(1, p=move_p)):
                        self.schedule.move_family(max_family_id, channel[0], channel[1])

                else:
                    # if there is a 2 way channel, make a copy of schedule, swap families, and check if
                    # cost went down, or just up a little
                    sorted(S_dict[inverse_channel], key=get_dc, reverse=True)
                    inverse_dc, inverse_max_family_id = S_dict[inverse_channel].pop(0)
                    channel = np.ravel_multi_index(channel, (101, 101))
                    inverse_channel = np.array((channel[1], channel[0]))

                    schedule_copy = copy.deepcopy(self.schedule)
                    schedule_copy.swap_families(schedule_copy.family_dict[max_family_id],
                                                schedule_copy.family_dict[inverse_max_family_id])

                    delta_sc = schedule_copy.schedule_cost - self.schedule.schedule_cost
                    move_p = 1 if delta_sc < 5 else (1/(5 * np.log(delta_sc)))

                    # decide whether or not to move
                    if bool(np.random.binomial(1, p=move_p)):
                        self.schedule.move_family(self.schedule.family_dict[max_family_id],
                                                  self.schedule.family_dict[inverse_max_family_id])

                    del schedule_copy

            if step % 20 == 0:
                print(f"cost at step {step}: {self.schedule.schedule_cost}")

        print(f"{steps} steps taken in {time.time() - ts} seconds")
        print(f"final cost: {self.schedule.schedule_cost}")
        print(f"final accounting cost: {sum(self.schedule.accounting_cost)} final choice cost "
              f"{self.schedule.choice_cost}")

    def circuit_move(self, time_seconds=60):
        """
        To traverse the space with more complex moves, I will need a method to swap multiple families
        in the same move, I will define channels as moves from day_i to day_j. Channels with choice cost to be gained
        will be considered. Then a circuit will be constructed by going through channels until the original day is found.
        This method should reduce the effect on accounting cost by keeping n_day the same for most moves.

        each move will search a tree structure until a greedy move is found, and maybe continue searching for
        a better move because the overhead of finding one move will be possibly large

        Method:
        - pick a starting day with probability proportional to the cost to be gained from families on that day (S).
        - branch out in a tree to the next channel chosen by P ~ S[i,j] where i was chosen on step 1
        - need to add options in each channel to move no family from i -> j
        - OPTIONS:
            - calculate delta cost for every node on the tree
            - make a copy schedule for every node in the tree to calculate cost ***
            - make a copy schedule for every leaf node that connects back to the starting i chosen in step 1
        - if an improvement on cost is found, continue searching? or start new step.
        - Also, only the S value that were changed will need to be recalculated on each step

        - A heuristic should be added for moving large or small families based on the accounting cost for the effected
            days, this should be a continuous parameter, not boolean.

            A good way to go about estimating accounting cost difference will be to consider the
            change in n_people when 2 channels are linked together. If we were to look at the
            instantaneous accounting cost it would tell us nothing. This will disregard the change
            in accounting cost from the first family moved until the circuit is completed.

        - It might be possible to work in a heuristic for moving a family across a channel while
            considering the effects on accounting cost.

        - I don't want the algorithm to prefer searching for channel connections that move a large
            family to reduce accounting cost from the multiplicative effect of n_people. For the
            accounting cost, only the difference in n,n+1 will be considered to try evenly smoothing
            out the accounting costs.

        :param steps:
        :return:
        """
        def list_insert(l, cit):
            """
            the list has elements (cost, item), the list is in order high to low cost
            this function inserts cit (cost, item) touple, into the list based on cost
            :param l:
            :param cit:
            :return:
            """

            def bisect_index(cl, c):

                """
                :param cl: list of cost
                :param c: cost of element to return index
                :return:
                """

                if len(cl) <= 1:
                    return 0
                mid = int(len(cl) / 2)
                if cl[mid] == c:
                    return mid
                elif cl[mid] < c:
                    return bisect_index(cl[:mid], c)
                elif cl[mid] > c:
                    return 1 + mid + bisect_index(cl[mid+1:], c)

            get_cost = lambda x: x[0]
            ins_idx = bisect_index(list(map(get_cost, l)), cit[0])
            l.insert(ins_idx, cit)

        def accounting_heuristic(delta_people, day):
            n_slice = self.schedule.n_people_pairing[day-1: day+2]
            dm = (n_slice[:2] - 125)/400

            current_a = np.power(n_slice[:2], (.5 + (np.abs(n_slice[:2] - n_slice[1:])/50)))
            dn_slice = n_slice
            dn_slice[1] += delta_people
            next_a = np.power(n_slice[:2], (.5 + (np.abs(n_slice[:2] - n_slice[1:])/50)))

            h_delta = sum(dm * (next_a - current_a))
            return h_delta

        class Node:
            def __init__(self, start_day=None, visited=False, depth=0, h_cost=0, parent=None, children=[],
                         next_days=np.zeros(101, dtype=bool), family_id=-1, channel=np.array([0, 0])):
                self.parent = parent
                self.children = children
                self.depth = depth
                self.h_cost = h_cost
                self.start_day = start_day  # channel[0] of the root node
                self.visited = visited
                self.family_id = family_id
                self.channel = channel

            def node_search(self, schedule, S, S_dict):
                """
                given a node, the function will use S to find a candidate next channel (node), then a list of families
                will be generated until 90% of the cost associated with that channel is accounted for. Then the function
                will calculate the choice cost plus accounting heuristic for each family. then the function returns the
                family with probability based on the delta cost
                :param node:
                :return: node
                """

                N_CHANNELS = 10

                if self.visited:
                    days_visited = list(map(lambda x: x.channel[0], self.children))
                else:
                    days_visited = []
                self.visited = True

                p_candidate_day = S[self.channel[1], :]
                if sum(p_candidate_day) == 0:
                    return False
                np.nan_to_num(p_candidate_day, False)
                p_candidate_day = p_candidate_day / sum(p_candidate_day)
                candidate_days = np.random.choice(np.arange(101), size=N_CHANNELS, p=p_candidate_day)

                best_cost = -1000000
                for candidate_day in candidate_days:
                    if candidate_day in days_visited:
                        continue

                    fam_list = S_dict[self.channel[1], candidate_day]
                    for d_c, fam_id in fam_list:
                        current_n = schedule.family_dict[self.family_id].n_people
                        candidate_n = schedule.family_dict[fam_id].n_people
                        delta_n = candidate_n - current_n
                        h_cost = d_c - accounting_heuristic(delta_n, candidate_day)

                        if self.h_cost + h_cost > best_cost:
                            best_cost = self.h_cost + h_cost

                            return_node = Node(start_day=self.start_day, parent=self, depth=self.depth + 1,
                                               h_cost=self.h_cost + h_cost, family_id=fam_id,
                                               channel=np.array([self.channel[1], candidate_day]))

                        if best_cost > 0:
                            self.children.append(return_node)
                            return return_node

                return return_node

        # Algorithm Parameters --------------------------------------------------------------------------------
        CHANNEL_QUEUE_LEN = 100
        NODES_PER_STEP = 300
        # ----------------------------------------------------------------------------------------------------

        circuits = []
        # define S as the amount of choice cost available on each channel
        S_dict = {}
        S = np.zeros((101, 101), dtype=np.float32)
        for family in list(self.schedule.family_dict.values()):
            family_day = family.day
            family_days = family.choice_array[:family.current_choice]

            # delta cost is the positive amount to be gained in choice cost
            delta_cost = self.schedule.family_choice_cost[family.family_id] - \
                         self.schedule.costMatrices.get_choice_array(family.n_people)[:family.current_choice]

            for idx, d_c in enumerate(delta_cost):
                if d_c > 0:
                    S[family_day, family_days[idx]] += d_c

                    try:
                        S_dict[family_day, family_days[idx]].append((d_c, family.family_id))
                    except KeyError:
                        S_dict[family_day, family_days[idx]] = []
                        S_dict[family_day, family_days[idx]].append((d_c, family.family_id))

            for channel in list(S_dict.keys()):
                key_fn = lambda x: x[0]
                S_dict[channel].sort(key=key_fn, reverse=True)

        # make a probability distribution of S for the magnitude of each entry
        P = S / np.sum(S)
        np.nan_to_num(P, False)
        starting_channels = np.random.choice(np.arange(101*101), size=CHANNEL_QUEUE_LEN, p=P.flatten())

        starting_channels = np.unravel_index(starting_channels, (101, 101))
        starting_channels = np.transpose(starting_channels)

        queue = []
        output = []
        for channel in starting_channels:
            channel_root = Node(start_day=channel[0], h_cost=(sum(S[channel[0], :])), parent=None)
            queue.append((channel_root.h_cost, channel_root))

        queue.sort(key=lambda x: x[0], reverse=True)
        ts = time.time()
        count_steps = 0
        while (time.time() - ts) < time_seconds:
            # the queue is kind of dynamic, it starts with (cost, root node) touples, but once a root node is
            # evaluated, it inserts that search queue back into the queue as (best_cost, search_queue)
            queue_p = np.array(list(map(lambda x: x[0], queue)))
            queue_p = queue_p / sum(queue_p)
            queue_idx = np.random.choice(np.arange(len(queue)), p=queue_p)
            queue_item = queue.pop(queue_idx)
            best_cost = queue_item[0]

            if type(queue_item[1]) is list:
                search_queue = queue_item[1]

            else:
                search_queue = list()
                search_queue.append((best_cost, queue_item[1]))

            for _ in range(NODES_PER_STEP):
                # at this point I have a search queue that I am removing the most expensive node from
                # and looking for child nodes to add to the search queue. in this loop a new node will be found
                # and added to the search queue for the given root node. The function searching for a node to
                # add will have to look for nodes with visited = False, because when a node is evaluated in this
                # loop, it is not removed from the search queue
                count_steps += 1

                search_node = search_queue[0][1]
                next_node = search_node.node_search(self.schedule, S, S_dict)

                if not next_node:
                    continue

                print("FOUND A NODE")

                if next_node.channel[1] == next_node.start_day:
                    moves = []
                    current = next_node
                    while current.parent:
                        moves.append([current.family_id, current.channel[0], current.channel[1]])
                        current = current.parent

                    test_schedule = copy.deepcopy(self.schedule)
                    for move in moves:
                        test_schedule.move_family(move[0], move[1], move[2])

                    if test_schedule.schedule_cost < self.schedule.schedule_cost:
                        output.append(test_schedule)

                else:
                    list_insert(search_queue, (next_node.h_cost, next_node))

            list_insert(queue, (search_queue[0][0], search_queue))

        print(f"steps taken: {count_steps}")
        output.sort(key=lambda x: x.schedule_cost, reverse=True)
        if len(output) > 0:
            self.schedule = output[0]
