from model import *
import datetime
import time

def run_experiments(configurations, num_experiments, population_size, max_iterations, mutation_rate):

    model = GeneticAlg()

    # Debug
    report = "n m Min Avg. Max Std. Time Mean\n"
    begin = datetime.datetime.now()
    count = 1

    for (n, m) in configurations:
        results = []
        times = []

        for _ in range(num_experiments):
            dna_chains = [''.join(np.random.choice(['A', 'C', 'G', 'T'], size=m)) for _ in range(n)]

            start_time = time.time()

            best_solution = model.find_solution(
                selection_method=methods["roulette"],
                dna_chains=dna_chains,
                max_iterations=max_iterations,
                population_size=population_size,
                mutation_rate=mutation_rate
            )
            best_distance = model.calculate_max_hamming_distance(dna_chains, best_solution)

            end_time = time.time()

            results.append(best_distance)
            times.append(end_time - start_time)

        min_val = min(results)
        avg_val = np.mean(results)
        max_val = max(results)
        std_val = np.std(results)
        avg_time = np.mean(times)

        # Calcula o Gap em relação à melhor solução conhecida (min_val)
        gaps = [(result - min_val) / min_val * 100 for result in results]
        avg_gap = np.mean(gaps)

        newReport = f"{n} {m} {min_val} {avg_val:.1f} {max_val} {std_val:.2f} {avg_time:.2f} {avg_gap:.2f}\n"
        report += newReport

        print(f"Setup: {count}, report: \n{report}")
        count+=1


    end = datetime.datetime.now()

    print(f'The experiments takes: {end - begin}')
    return report


def main():
    num_experiments = 5
    configurations = [
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

    # GA params
    population_size = 100
    max_iterations = 50  
    mutation_rate = 0.3

    run_experiments(
        configurations=configurations,
        max_iterations=max_iterations,
        mutation_rate=mutation_rate,
        num_experiments=num_experiments,
        population_size=population_size,
    )

if __name__ == "__main__":
    main()
