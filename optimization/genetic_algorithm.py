import random


class GeneticAlgorithm:

    def __init__(self, simulation, policies):
        # save the simulation and the policies
        self.simulation = simulation
        self.policies = policies
        self.population = []
        self.population_evaluated = []
        self.population_parents = []
        # get the individual size
        self.chromosome_size = len(self.simulation.get_actual_combination())
        # get the set of machines
        self.machines_list = list(self.simulation.available_machines.keys())
        # saves the actual generation
        self.generation_idx = 0

    def fitness_function(self):
        # get the values from the simulation
        uv = self.simulation.read_utility_values()
        # function component (l_path_avg, ram_avg, cpu_avg)
        # calculate the fitness function
        fitness_function = sum([self.policies[metric]*value for metric, value in uv.items()])
        # return the value
        return fitness_function

    def create_population(self, population_size):
        # resets the actual generation
        self.generation_idx = 0
        # over the population size
        for i in range(population_size):
            # create each individual
            individual = random.choices(self.machines_list, k=self.chromosome_size)
            # append the individual
            self.population.append(individual)

    def evaluate(self):
        fitness_results = []
        # iterate over the population
        for individual in self.population:
            # sets the simulation with the current individual
            self.simulation.update_combination(individual)
            # runs the simulation
            self.simulation.simulate_graph()
            # calls the fitness function with that values
            fitness_value = self.fitness_function()
            # append the result to the list
            fitness_results.append(fitness_value)
        # joins all in a list
        self.population_evaluated = list(zip(self.population, fitness_results))
        # return the list of tuples
        return self.population_evaluated

    def rank_selection(self, n_selected):
        # sort the population by the fitness
        sorted_population = sorted(self.population_evaluated, key=lambda x: x[1])
        # select the best n individuals
        self.population_parents = sorted_population[:n_selected]
        # update the simulation with the best individual
        self.simulation.update_combination(self.population_parents[0][0])
        # return the n best individuals
        return self.population_parents

    def create_new_population(self, population_size, n_mutations=5):
        # increments the generation idx
        self.generation_idx += 1
        # resets the population
        self.population = []
        # get the parents list
        parents, _ = zip(*self.population_parents)
        parents = list(parents)
        # calculates the number of children
        n_children = population_size - len(parents) - n_mutations
        # stores the parents in the new population
        self.population += parents
        # creates the children population
        for _ in range(n_children):
            # selects 2 random parents
            curr_parents = random.sample(parents[:5], k=2)
            # crossover the parents
            children = self.uniform_crossover(curr_parents)
            # appends the children to the population
            self.population.append(children)
        # make mutations in the new generation
        for idx, curr_individual in enumerate(random.choices(parents, k=n_mutations)):
            # mutates the current individual
            if idx < (n_mutations // 2):
                # normal mutation
                mutated_individual = self.mutation(curr_individual.copy())
            else:
                # swap mutation
                mutated_individual = self.swap_mutation(curr_individual.copy())
            # appends the mutated individual
            self.population.append(mutated_individual)

    def mutation(self, chromosome):
        # selects the idx and flips them
        idx = random.choice(range(self.chromosome_size))
        # replace the gene
        chromosome[idx] = random.choice(self.machines_list)
        # returns the mutated individual
        return chromosome

    def swap_mutation(self, chromosome):
        # selects the positions to swap
        pos = random.sample(range(self.chromosome_size), k=2)
        # swaps the genes
        chromosome[pos[0]], chromosome[pos[1]] = chromosome[pos[1]], chromosome[pos[0]]
        # returns the mutated individual
        return chromosome

    def crossover(self, parents):
        # select the crossover point
        point = random.choice(range(self.chromosome_size))
        # built the children
        children = parents[0][:point] + parents[1][point:]
        # returns the value
        return children

    def uniform_crossover(self, parents):
        # select the crossover point
        point = random.choice(range(self.chromosome_size))
        # crossover the current parents
        father_indexes = random.sample(range(self.chromosome_size), k=point)
        # maps the children according to the parents
        children = []
        for idx in range(self.chromosome_size):
            # checks where is the index
            idx_value = parents[0][idx] if idx in father_indexes else parents[1][idx]
            # appends the value to the children
            children.append(idx_value)
        # return the children chromosome
        return children

    def select_stronger(self):
        # selects the minimum
        individual, fitness = min(self.population_parents, key=lambda x: x[1])
        # returns the individual
        return individual, fitness

    def population_variability(self):
        pass

    def get_optimization_state(self):
        # dict to store the ga information
        data_dict = dict()
        # store the number of generations, n_individuals tested
        data_dict['n_generations'] = self.generation_idx
        data_dict['chromosome_size'] = self.chromosome_size
        data_dict['best_individual'], data_dict['fitness'] = self.select_stronger()
        # returns the dict
        return data_dict
