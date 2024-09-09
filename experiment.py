from model import methods, GeneticAlg
import numpy as np
import csv
import os
import datetime

class Experiment():
    def __init__(
            self, 
            method, 
            configuration = None, 
            num_experiments = 5,
            population_size = 75, 
            max_iterations = 50, 
            mutation_rate = 0.1
        ):
        self.method = method
        self.num_experiments = num_experiments

        self.population_size = population_size
        self.max_iterations = max_iterations  
        self.mutation_rate = mutation_rate

        if(configuration is not None):
            self.configurations = configuration
        else:
            self.configurations = [
                (10, 1000),
                (10, 2000),
                (10, 3000),
                (10, 4000),
                (10, 5000),
                (20, 1000),
                (20, 2000),
                (20, 3000),
                (20, 4000),
                (20, 5000),
                (30, 1000),
                (30, 2000),
                (30, 3000),
                (30, 4000),
                (30, 5000)
            ]

        print("---------------------------------------")
        print(f"Experiment with Method: {method}")
        print("---------------------------------------")

    def run_experiments(self, configurations, num_experiments, population_size, max_iterations, mutation_rate):

        model = GeneticAlg()

        # Debug
        report = "n m Min Avg. Max Std. Time Mean\n"
        begin = datetime.datetime.now()
        count = 1

        print("Starting the experiments...")
        print("---------------------------------------")
        for (n, m) in configurations:
            results = []
            times = []

            exp_start = datetime.datetime.now()
            for _ in range(num_experiments):
                dna_chains = model.generate_dna_chain(m,n)

                start_time = datetime.datetime.now()

                best_solution = model.find_solution(
                    selection_method=methods[self.method],
                    dna_chains=dna_chains,
                    max_iterations=max_iterations,
                    population_size=population_size,
                    mutation_rate=mutation_rate
                )
                best_distance = model.calculate_max_hamming_distance(dna_chains, best_solution)

                end_time = datetime.datetime.now()

                results.append(best_distance)
                duration = end_time - start_time

                times.append(duration.total_seconds()/60)

            min_val = min(results)
            avg_val = np.mean(results)
            max_val = max(results)
            std_val = np.std(results)
            avg_time = np.mean(times)

            # Calcula o Gap em relação à melhor solução conhecida (min_val)
            gaps = [(result - min_val) / min_val * 100 for result in results]
            avg_gap = np.mean(gaps)

            exp_end = datetime.datetime.now()
            exp_duration = exp_end - exp_start
            
            newReport = f"{n} {m} {min_val} {avg_val:.1f} {max_val} {std_val:.2f} {avg_time:.2f} {avg_gap:.2f}\n"
            report += newReport

            print(f"Setup: {count} expend {exp_duration.total_seconds()/60} minutes, report: \n{report}")
            count+=1


        end = datetime.datetime.now()

        print(f'The experiments takes: {end - begin}')
        print("---------------------------------------")

        report += f'Time spent: {end}'
        return report
    
    def print_configuration(self):
        print("Genetic Algorithm Params: ")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Population Size: {self.population_size}")
        print(f"Epochs: {self.max_iterations}")
        print("---------------------------------------")
        print('Experiment configurations:')
        print(f"Iterations: {self.num_experiments}")
        for (n, m) in self.configurations:
            print(f'n: {n}, m: {m}')
        print("---------------------------------------")

    def save_report(self, report, dir):

        if not os.path.exists(dir):
            os.mkdir(dir)

        rows = [line.split() for line in report.split('\n')]
        rows.append("\n")

        file_path = os.path.join(dir, 'report.csv')

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f'Data has been written to {file_path}')

    def start(self, report_dir = None):
        report = self.run_experiments(
            configurations=self.configurations,
            max_iterations=self.max_iterations,
            mutation_rate=self.mutation_rate,
            num_experiments=self.num_experiments,
            population_size=self.population_size,
        )

        if(report_dir is not None):
            self.save_report(report, report_dir)

