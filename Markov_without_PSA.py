import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Costs and utility ratios
costs_with_ASP = np.array([666, 666, 666, 746, 1362, 2088, 2088, 2088, 2338, 4270, 0, 0])
costs_without_ASP = np.array([3236, 3236, 3236, 3625, 6619, 10144, 10144, 10144, 11361, 20747, 0, 0])

# Initial population
initial_population = np.array([800, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0])  # Initial state


transition_matrix_without_ASP = np.array([
    [0.0000, 0.8493, 0.0000, 0.0000, 0.0000, 0.0000, 0.0048, 0.0000, 0.0000, 0.0000, 0.1379, 0.0080],
    [0.0000, 0.0000, 0.8224, 0.0000, 0.0000, 0.0000, 0.0000, 0.0058, 0.0000, 0.0000, 0.1622, 0.0096],
    [0.0000, 0.0000, 0.7834, 0.0019, 0.0029, 0.0000, 0.0000, 0.0071, 0.0000, 0.0000, 0.1929, 0.0118],
    [0.0000, 0.0000, 0.0000, 0.9029, 0.0000, 0.0000, 0.0000, 0.0000, 0.0046, 0.0000, 0.0697, 0.0228],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.9530, 0.0000, 0.0000, 0.0000, 0.0000, 0.0022, 0.0295, 0.0153],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8562, 0.0000, 0.0000, 0.0000, 0.1226, 0.0212],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8321, 0.0000, 0.0000, 0.1430, 0.0249],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7918, 0.0215, 0.0333, 0.1307, 0.0227],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8626, 0.0000, 0.0894, 0.0480],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9540, 0.0257, 0.0203],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])

transition_matrix_with_ASP = np.array([
    [0.0000, 0.8306, 0.0000, 0.0000, 0.0000, 0.0000, 0.0055, 0.0000, 0.0000, 0.0000, 0.1548, 0.0091],
    [0.0000, 0.0000, 0.7700, 0.0000, 0.0000, 0.0000, 0.0000, 0.0100, 0.0000, 0.0000, 0.2100, 0.0100],
    [0.0000, 0.0000, 0.7432, 0.0016, 0.0024, 0.0000, 0.0000, 0.0087, 0.0000, 0.0000, 0.2297, 0.0144],
    [0.0000, 0.0000, 0.0000, 0.9029, 0.0000, 0.0000, 0.0000, 0.0000, 0.0046, 0.0000, 0.0697, 0.0228],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.9530, 0.0000, 0.0000, 0.0000, 0.0000, 0.0022, 0.0295, 0.0153],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8562, 0.0000, 0.0000, 0.0000, 0.1226, 0.0212],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8321, 0.0000, 0.0000, 0.1430, 0.0249],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7933, 0.0151, 0.0234, 0.1432, 0.0250],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8626, 0.0000, 0.0894, 0.0480],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9540, 0.0257, 0.0203],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])

'''
row_sums_without_ASP = transition_matrix_without_ASP.sum(axis=1)
row_sums_with_ASP = transition_matrix_with_ASP.sum(axis=1)

print("Row sums of transition_matrix_without_ASP:")
print(row_sums_without_ASP)

print("\nRow sums of transition_matrix_with_ASP:")
print(row_sums_with_ASP)
'''

def verify_transition_matrix(matrix):
    """
    Verifies that each row of the matrix sums to 1 (within tolerance)
    and contains no negative values.
    """
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1, atol=1e-6):
        raise ValueError("Rows of the transition matrix do not sum to 1.")
    if not np.all(matrix >= 0):
        raise ValueError("Transition matrix contains negative probabilities.")

verify_transition_matrix(transition_matrix_with_ASP)
verify_transition_matrix(transition_matrix_without_ASP)



def run_markov_with_dalys(transition_matrix, costs, population):
    max_cycles=1000
    tolerance=1e-6
    total_cost = 0
    ward_days = 0
    icu_days = 0
    hospital_days = 0
    cycles = 0

    while population[:-2].sum() >= tolerance and cycles < max_cycles:  # Stop when non-terminal states are < 1
        # Calculate costs, utilities, and DALYs for the current cycle
        total_cost += np.dot(population, costs)
        ward_days = ward_days + population[0] + population[1] + population[2] + population[3] + population[4] 
        icu_days = icu_days + population[5] + population[6] + population[7] + population[8] + population[9] 
        hospital_days = ward_days + icu_days
        # Update population
        new_population = np.dot(population, transition_matrix)

        # Check for convergence
        if np.all(np.abs(new_population - population) < tolerance):
            break

        population = new_population
        cycles += 1
    
    #print(total_dalys_lost, cycles)

    #return total_cost, total_utility, total_dalys_lost, cycles
    return total_cost, ward_days, icu_days, hospital_days, cycles, population[11], population[10]

# Run simulations for both scenarios

cost_with_ASP, ward_days_with_ASP, icu_days_with_ASP, hospital_days_with_ASP, cycles_with_ASP, deaths_with_ASP, survived_with_ASP = run_markov_with_dalys(
    transition_matrix_with_ASP, costs_with_ASP, initial_population)

cost_without_ASP, ward_days_without_ASP, icu_days_without_ASP, hospital_days_without_ASP, cycles_without_ASP, deaths_without_ASP, survived_without_ASP = run_markov_with_dalys(
    transition_matrix_without_ASP, costs_without_ASP, initial_population)

incremental_cost = cost_with_ASP - cost_without_ASP
ward_days_saved = -(ward_days_with_ASP - ward_days_without_ASP)
icu_days_saved = -(icu_days_with_ASP - icu_days_without_ASP)
hospital_days_saved = -(hospital_days_with_ASP-hospital_days_without_ASP)
deaths_averted = -(deaths_with_ASP - deaths_without_ASP)


icer_ward_days = incremental_cost / ward_days_saved if ward_days_saved != 0 else float('inf')
icer_icu_days = incremental_cost / icu_days_saved if icu_days_saved != 0 else float('inf')
icer_hospital_days = incremental_cost / hospital_days_saved if hospital_days_saved != 0 else float('inf')
icer_deaths = incremental_cost / deaths_averted if deaths_averted != 0 else float('inf')


# Output results
print("Results with ASP:")
print(f"Total Cost: {cost_with_ASP}")
print(f"Total Ward Days: {ward_days_with_ASP}")
print(f"Total ICU Days: {icu_days_with_ASP}")
print(f"Total Hospital Days: {hospital_days_with_ASP}")
print(f"Total Deaths: {deaths_with_ASP}")
print(f"Cycles until termination: {cycles_with_ASP}")


print("\nResults without ASP:")
print(f"Total Cost: {cost_without_ASP}")
print(f"Total Ward Days: {ward_days_without_ASP}")
print(f"Total ICU Days: {icu_days_without_ASP}")
print(f"Total Hospital Days: {hospital_days_without_ASP}")
print(f"Total Deaths: {deaths_without_ASP}")
print(f"Cycles until termination: {cycles_without_ASP}")

print("\nIncrements and savings:")
print(f"Incremental Cost: {incremental_cost}")
print(f"Ward Days Saved: {ward_days_saved}")
print(f"ICU Days Saved: {icu_days_saved}")
print(f"Hospital Days Saved: {hospital_days_saved}")
print(f"Deaths Averted: {deaths_averted}")

print("\nICERs:")

print(f"ICER deaths: {icer_deaths}")
print(f"ICER ward days: {icer_ward_days}")
print(f"ICER ICU days: {icer_icu_days}")
print(f"ICER hospital days: {icer_hospital_days}")