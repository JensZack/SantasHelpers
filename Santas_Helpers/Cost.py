# Class for creating cost matrices to keep in RAM for all computations
import numpy as np


class CostMatrices:
    def __init__(self):
        gift_card = np.array([0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500])
        buffet_multiplier = np.array([0, 0, 9, 9, 9, 18, 18, 36, 36, 36, 36])
        helicopter_multiplier = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 199, 398])

        # Axis 0 is the number of people in a family
        # Axis 1 is the choice the family received
        self.ChoiceMatrix = np.zeros((7, 11), dtype=np.int16)

        for i in range(self.ChoiceMatrix.shape[0]):
            for j in range(self.ChoiceMatrix.shape[1]):
                n_people = i + 2
                cost = gift_card[j] + n_people * (buffet_multiplier[j] + helicopter_multiplier[j])
                self.ChoiceMatrix[i, j] = int(cost)

        accounting_dim = 300-125+1
        self.AccountingMatrix = np.zeros((accounting_dim, accounting_dim))

        cost_function = lambda nt, nt1: ((nt - 125) / 400) * np.power(nt, (.5 + (abs(nt - nt1))/50))

        for i in range(accounting_dim):
            for j in range(accounting_dim):
                nt = i + 125
                nt1 = j + 125
                self.AccountingMatrix[i, j] = cost_function(nt, nt1)

    def get_choice_cost(self, n_people, choice):
        n_idx = n_people - 2
        return self.ChoiceMatrix[n_idx, choice]

    def get_choice_array(self, n_people):
        n_idx = n_people - 2
        return self.ChoiceMatrix[n_idx, :]

    def get_accounting_cost(self, nt, nt1):
        nt_idx = nt - 125
        nt1_idx = nt1 - 125
        return self.AccountingMatrix[int(nt_idx), int(nt1_idx)]
