import numpy as np
import random

methods = {
    'roulette': 'roulette',
    'tournament': 'tournament',
    'sus': 'sus',
}

class GeneticAlg:
    def __init__(self):
        print("------------------------------------------------")
        print("Model initialized.")
        print("------------------------------------------------")
        self.best_fitness = -float('inf')

    def calculate_hamming_distance(self, chain1, chain2):
        return sum(a != b for a, b in zip(chain1, chain2))
    
    def calculate_max_hamming_distance(self, dna_chains, solution):
        max_distance = max(self.calculate_hamming_distance(solution, chain) for chain in dna_chains)
        return max_distance
    
    def build_initial_population(self, dna_chains, population_size):
        n = len(dna_chains[0])  # length of each DNA chain
        population = []

        # Randomly generate DNA strings of length `n`
        for _ in range(population_size):
            random_dna = ''.join(random.choice('ACGT') for _ in range(n))
            population.append(random_dna)

        return population
    
    def calculate_fitness(self, dna_chains, solution):
        return -self.calculate_max_hamming_distance(dna_chains, solution)

    def crossover(self, parent1, parent2):
        # Use uniform crossover for more diversity
        child = ''.join(random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2))
        return child

    def roulette_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        selected = []

        for _ in range(len(population)):
            spin = random.random()
            prob_sum = 0

            for i in range(len(probabilities)):
                prob_sum += probabilities[i]
                if prob_sum >= spin:
                    selected.append(population[i])
                    break

        return selected

    def tournament_selection(self, population, fitness_scores, tournament_size=3): # Increased size
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(population, tournament_size)
            candidate_indices = [population.index(individual) for individual in candidates]
            candidate_scores = [fitness_scores[i] for i in candidate_indices]

            winner_index = candidate_indices[candidate_scores.index(max(candidate_scores))]
            selected.append(population[winner_index])

        return selected

    def sus_selection(self, population, fitness_scores):
        selected = []
        total_fitness = sum(fitness_scores)
        normalized_fitness = [score / total_fitness for score in fitness_scores]

        step_size = 1.0 / len(population)
        points = [random.random() * step_size for _ in range(len(population))]

        current_point = 0
        for i in range(len(population)):
            while points[i] > normalized_fitness[current_point]:
                current_point += 1
                if current_point == len(population):
                    current_point = 0

            selected.append(population[current_point])

        return selected
    
    def mutation(self, solution, mutation_rate):
        n = len(solution)
        solution = list(solution)  # Convert to list for mutability

        for i in range(n):
            if random.random() < mutation_rate:
                solution[i] = random.choice('ACGT')

        return ''.join(solution)  

    def apply_local_search(self, solution, dna_chains):

        best_solution = solution
        best_fitness = self.calculate_fitness(dna_chains, solution)
        
        for i in range(len(solution)):
            for nucleotide in 'ACGT':
                if solution[i] == nucleotide:
                    continue
                new_solution = solution[:i] + nucleotide + solution[i+1:]
                new_fitness = self.calculate_fitness(dna_chains, new_solution)

                if new_fitness > best_fitness: 
                    best_solution = new_solution
                    best_fitness = new_fitness
        
        return best_solution

    def find_solution(self, selection_method, dna_chains, population_size, max_iterations, mutation_rate): 
        population = self.build_initial_population(dna_chains, population_size)
        best_solution = None

        for iteration in range(max_iterations):
            fitness_scores = [self.calculate_fitness(dna_chains, solution) for solution in population]
            selected_parents = self.get_selection(selection_method, population, fitness_scores)
            new_population = []

            for _ in range(int(population_size / 2)):
                parent1, parent2 = random.sample(selected_parents, 2)

                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                child1 = self.mutation(child1, mutation_rate)
                child2 = self.mutation(child2, mutation_rate)

                # child1 = self.apply_local_search(child1, dna_chains)
                # child2 = self.apply_local_search(child2, dna_chains)

                new_population.append(child1)
                new_population.append(child2)

            elite_count = 2
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elites = [population[i] for i in elite_indices]
            population = new_population + elites

        best_solution_index = np.argmax(fitness_scores)
        best_solution = population[best_solution_index]

        return best_solution
    
    def get_selection(self, method, population, fitness_scores):
        if method == methods['roulette']:
            return self.roulette_selection(population, fitness_scores)
        if method == methods['tournament']:
            return self.tournament_selection(population, fitness_scores)
        if method == methods['sus']:
            return self.sus_selection(population, fitness_scores)
