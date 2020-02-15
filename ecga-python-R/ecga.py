"""
  FileName    [ ecga.py ]
  PackageName [ ga ]
  Synposis    [ ecGA model builder ] 
  Support     [ python3 ]
"""

import argparse
import pprint
import os
import numpy as np
import itertools
import math
import random
from collections import Counter

class ECGA:
    def __init__(self):
        self.population = []

class Prob(Counter):
    def __init__(self):
        pass

record = {}

def loadPopulation(fname):
    """ Read population set from newest population file. """
    population = []

    with open(fname, 'r') as textfile:
        population = np.array([ [ bool(int(bit)) for bit in line.strip() ] for line in textfile.readlines() ])

    return population

def loadModel(fname):
    """ Parse model from model file. """
    model = []

    with open(fname, 'r') as textfile:
        for line in textfile.readlines():
            block = tuple(sorted([ int(num) for num in line.strip().split(' ')[1:]] ))
            if len(block): model.append(block)

    return model

def arrayRepresentation(indice):
    # Count Bit Number
    totalIndices = 0
    for block in indice: totalIndices += len(block)

    nparray = np.zeros(totalIndices)
    for index, block in enumerate(indice, 0): nparray[ np.array(block) ] = index
    return nparray

def indiceRepresentation(nparray):
    """ Using indice to represent a model """
    return sorted([ tuple(np.where(nparray == i)[0]) for i in range(0, np.amax(nparray + 1)) ])

def ModelDescriptionLength(model, resolution):
    """ D_{Model} = \sum_{bb \in BB} (2 ^ {|bb|} - 1) * \log_2{n}"""
    return sum([ ((2 ** len(block)) - 1) * math.log2(resolution) for block in model ])

def BlockDescriptionLength(block, pattern):
    """ D_Block = \sum_{x \in bb} p(x) log_2{p(x)} """
    probability = Counter()
    length = 0
    resolution = pattern.shape[0]

    # Calculate prob for each pattern
    for p in pattern: probability.update({"".join([ str(c) for c in p.tolist() ]): 1})

    # Get Length
    for p, prob in probability.items(): length += (prob / resolution) * math.log2(prob / resolution)

    return length

def DataDescriptionLength(model, population):
    """ D_{Data} = -n \sum_{bb \in BB} \sum_{x \in bb} p(x) log_2{p(x)} """
    global record

    totalLength = 0
    length = 0
    resolution = population.shape[0]

    for block in model:
        # Get Pattern
        if block in record: 
            totalLength += record[block]
            continue

        record[block] = BlockDescriptionLength(block, population[ :, np.array(block)].reshape(-1, len(block)))
        totalLength += record[block]

    totalLength = -1 * resolution * totalLength

    return totalLength

def descriptionLength(model, population):
    """ D = D_{Model} + D_{Data} """
    return ModelDescriptionLength(model, population.shape[0]) + DataDescriptionLength(model, population)

def report(model, population):
    """ Generate the model statistics report """
    for block in model:
        prob = Counter()
        for p in population[ :, np.array(block) ]:
            prob.update({"".join([ str(c) for c in p.tolist() ]): 1})
        print(block, "\t", prob)

    return 

def diff(prev, present):
    """ Compare the difference of the model """
    if len(set(prev) - set(present)):
        print("Difference Blocks in previous generation: ", set(prev) - set(present))
        print("Difference Blocks in present generation: ", set(present) - set(prev))
    else:
        print("No difference between previous and present generation")   
 
    return bool(len(set(prev) - set(present)))

def main(opt):
    global record

    # Initailize
    record = {}

    # Load Population
    newestGeneration = sorted(os.listdir("./population"))[-1] if (opt.num is None) else "popu{}.txt".format(str(opt.num).zfill(3))
    population = loadPopulation(os.path.join("./population", newestGeneration)).astype(np.uint8)
    print("Loaded: {}".format(os.path.join("./population", newestGeneration)))

    # Initialize Model
    model = [ (i, ) for i in range(0, 50) ]
    minimumDescriptionLength = descriptionLength(model, population)

    # Greedy Search
    while True:
        # Initialize
        candidate = None
        minimumDescriptionLength = descriptionLength(model, population)

        # Get New Model
        iterlist = [ c for c in itertools.combinations(model, 2) ]
        random.shuffle(iterlist)

        # Search For New Model
        for index, selectElements in enumerate(iterlist, 1):
            newModel = list(set(model) - set(selectElements))
            newModel.append(tuple(set(selectElements[0]) | set(selectElements[1])))
            newLength = descriptionLength(newModel, population)
        
            # Update if find new candidate
            update = True if (newLength < minimumDescriptionLength) else False
            if (newLength < minimumDescriptionLength):
                candidate = selectElements
                minimumDescriptionLength = newLength

        # End Condition
        if candidate is None: break

        # Update for next round searching
        model = list(set(model) - set(candidate))
        model.append(tuple(set(candidate[0]) | set(candidate[1])))
        print("Merge ({}, {})".format(candidate[0], candidate[1]))

    # Sorted
    model = [ tuple(sorted(block)) for block in sorted(model) ]

    # Output Model and Score
    print("Building Blocks: ")
    print(arrayRepresentation(model))
    print("MDL: {} / {}".format(ModelDescriptionLength(model, population.shape[0]), DataDescriptionLength(model, population)))

    # Print out report
    report(model, population)

    # Find difference
    outputFile = os.path.join("model", newestGeneration.replace("popu", "mpm"))

    with open(outputFile, 'w') as textfile:
        textfile.writelines([str(len(model)), "\n"])
        for block in model: textfile.writelines(str(len(block)) + " " + " ".join([ str(b) for b in block ]) + "\n")

    if os.path.exists(outputFile.replace(newestGeneration[4:7], str(int(newestGeneration[4:7]) - 1).zfill(3))):
        diff(sorted(loadModel(outputFile.replace(newestGeneration[4:7], str(int(newestGeneration[4:7]) - 1).zfill(3)))), model)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int)
    parser.add_argument("-a", "--all", action="store_true", default=False)
    opt = parser.parse_args()

    models = {}

    if opt.all:
        for i in range(0, len(os.listdir('./population'))):
            opt.num = i
            models[i] = main(opt)
    else:
        main(opt)
