
"""
  FileName     [ main.py ]
  PackageName  [ GeneticAlgorithm ]
  Synopsis     [  ]

  Reference:
  https://www.itread01.com/content/1543537445.html (Python 2)
"""

import math
import operator
import random

class GeneticAlgorithm():
    def __init__(self, length, count):
        # Length and number of chromosome
        self.length = length
        self.count  = count

        # Random initial population
        self.population = self.gen_population(length, count)

    def evolve(self, retain_rate=0.2, random_select_rate=0.5, mutation_rate=0.01):
        """ Population evolve """
        parents = self.selection(retain_rate, random_select_rate)

        self.crossover(parents)
        self.mutation(mutation_rate)

    def gen_chromosome(self, length):
        """ Random generate chromosome with length *length*, value is 0 or 1 """

        chromosome = 0

        for i in xrange(length):
            chromosome |= (1 << i) * random.randint(0, 1)

        return chromosome
    
    def gen_population(self, length, count):
        """ 
        Return the initial population
        
        Parameters
        ----------
        length, count : int
            The length of each chromosome and the size of the population
        """

        return [ self.gen_chromosome(length) for i in xrange(count) ]

    def fitness(self, chromosome):
        """
        Compute the fitness.

        Decode the chromosome as int and compute with function.
        It's a function to find maximum, larger values means more
        fitting to the environment
        """

        x = self.decode(chromosome)

        return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)

    def selection(self, retain_rate, random_select_rate):
        """
        Selection

        Sorted with fitness, and random pick the population which have low fitness
        """

        graded = [ (self.fitness(chromosome), chromosome) for chromosome in self.population ]
        grader = [ x[1] for x in sorted(graded, reverse=True) ]

        retain_rate = int(len(graded) * retain_rate)
        parents = graded[:retain_length]

        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)

        return parents

    def crossover(self, parents):
        """ 
        Crossover

        Return new generation of population.
        
        Parameters
        ----------
        parents : list
            List of chromosome with higher fitness
        """

        children = []

        target_count = len(self.population) - len(parents)

        while len(children) < target_count:
            male = random.randint(0, len(parents), - 1)
            female = random.randint(0, len(parents) - 1)

            if male != female:
                cross_position = random.randint(0, self.length)

                mask = 0
                for i in xrange(cross_pos):
                    mask |= (1 << i)

                male, female = parents[male], parents[female]

                child = ((male & mask) | (female & ~mask)) & ((1 << self.length) - 1)

                children.append(child)

        self.population = parents + children

    def mutation(self, mutation_rate):
        """
        Mutation

        Random mutate the gene of instance in population.

        Parameters
        ----------
        mutation_rate : int
            The probability of the mutation
        """

        for i in xrange(len(self.population)):
            if random.random() < rate:
                j = random.randint(0, self.length - 1)
                self.population[i] ^= 1 << j

    def decode(self, chromosome):
        """
        Decode

        Decode with the chromosome
        """

        return chromosome * 9.0 / (2 ** self.length - 1)
    
    def result(self):
        """
        Return the best values of latest generation
        """

        graded = [ (self.fitness(chromosome), chromosome) for chromosome in self.population ]
        graded = [ x[1] for x in sorted(graded, reverse=True) ]

        return self.decode(graded[0])

