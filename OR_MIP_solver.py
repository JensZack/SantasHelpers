from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time
import pandas as pd
import numpy as np


def example_ortools(desired, n_people, has_accounting=True):
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400
    NUM_THREADS = 4
    NUM_SECONDS = 60
    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])
    num_days = desired.max()
    num_families = desired.shape[0]
    solver = pywraplp.Solver('Santa2019', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)
    solver.SetNumThreads(NUM_THREADS)
    C, B, I = {}, {}, {}
    for fid, choices in enumerate(desired):
        for cid in range(10):
            B[fid, choices[cid]-1] = solver.BoolVar('')
            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]

    for day in range(num_days):
        I[day] = solver.IntVar(125, 300, f'I{day}')
        solver.Add(solver.Sum(n_people[fid] * B[fid, day] for fid in range(num_families) if (fid, day) in B) == I[day])

        for fid in range(num_families):
            solver.Add(solver.Sum(B[fid, day] for day in range(num_days) if (fid, day) in B) == 1)

    objective = solver.Sum(C[fid, day] * B[fid, day] for fid, day in B)
    if has_accounting:
        Y = {}

        for day in range(num_days):
            next_day = min(day + 1, num_days - 1)
            gen = [(u, v) for v in range(176) for u in range(176)]
            for u, v in gen:
                Y[day, u, v] = solver.BoolVar('')
            solver.Add(solver.Sum(Y[day, u, v] * u for u, v in gen) == I[day] - 125)
            solver.Add(solver.Sum(Y[day, u, v] * v for u, v in gen) == I[next_day] - 125)
            solver.Add(solver.Sum(Y[day, u, v] for u, v in gen) == 1)

        accounting_penalties = solver.Sum(accounting_penalty(u + 125, v + 125) * Y[day, u, v] for day, u, v in Y)
        objective += accounting_penalties

        solver.Minimize(objective)
        sol = solver.Solve()
        status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
        print(sol)
        if status[sol] == 'OPTIMAL':
            print("Result: ", objective.solution_value())
            assigned_days = np.zeros(num_families, int)
            for fid, day in B:
                if B[fid, day].solution_value() > 0.5:
                    assigned_days[fid] = day + 1
            return assigned_days


def continuous_optimization(desired, n_people, has_accounting=False):
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400

    NUM_THREADS = 4
    NUM_SECONDS = 60
    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])
    num_days = desired.max()
    num_families = desired.shape[0]

    # solver = pywraplp.Solver('Continuous Approximation', pywraplp.Solver. ??? )
    # solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)
    # solver.SetNumThreads(NUM_THREADS)


def days_to_csv(assigned_days):
    filenameout = 'out/or_tools_' + str(int(time.time()))[-6:] + '.csv'
    dict_out = {'family_id': list(np.arange(5000)), 'assigned_day': list(assigned_days)}
    df_out = pd.DataFrame(dict_out)
    df_out.to_csv(filenameout, index=False)


def main():
    fdata = "/home/george/Documents/learning/kaggle_data/Santa_data/family_data.csv"
    choices = []
    n_people = []
    for itr in pd.read_csv(fdata).iterrows():
        itray = itr[1].values
        choices.append(itray[1:11])
        n_people.append(itray[11])

    choices = np.array(choices)
    n_people = np.array(n_people)
    output = example_ortools(choices, n_people, has_accounting=True)
    days_to_csv(output)


if __name__ == '__main__':
    main()
