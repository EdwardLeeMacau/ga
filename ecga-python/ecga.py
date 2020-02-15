"""
  FileName    [ ecga.py ]
  PackageName [ ga ]
  Synposis    [ ecGA model builder ] 
  Support     [ python3 ]
"""

import argparse
import itertools
import json
import math
import os
import pprint
import random
from collections import Counter

import numpy as np
import pandas as pd
import requests

EPSILON        = 0.001
STUDENT_ID     = "B05901119"
HISTORY_URL    = "http://140.112.175.111:8888/api/{}"
POPULATION_URL = "http://140.112.175.111:8888/population/B05901119/{}"

record = {}

def findnth(target, needle, n):
    parts = target.split(needle, n+1)
    return -1 if len(parts) <= n + 1 else len(target) - len(parts[-1]) - len(needle)

def getHistory(url):
    return json.loads(requests.get(url).text)["history"]

def downloadPopulation(url, dir):
    response = requests.get(url)
    with open(dir, 'w') as file: file.write(response.text)
    return True

def uploadModel(url, fname, generation):
    with open(fname, 'rb') as file:
        response = requests.post(url, files={'file': file.read()}, data={'gen': generation}).text
    response = json.loads(response)
    return response

def parseHistory(history):
    chromosome = [ hist['fitness'].strip()[-50:] for hist in history ]

    fitness = []
    for hist in history:
        tmp = hist['fitness']
        fitness.append(tmp[findnth(tmp, ':', 2) + 1: findnth(tmp, ' ', 2)].split('/'))
    fitness = np.array(fitness).transpose()
    
    df = pd.DataFrame({'MAX': fitness[0], 'MEAN': fitness[1], 'MIN': fitness[2], 'BEST': chromosome})
    return df

def oneRun(population, opt):
    """ Download, generate model and upload """
    downloadPopulation(POPULATION_URL.format(population), os.path.join('./population', population))
    main(opt)
    history = uploadModel(HISTORY_URL.format(STUDENT_ID), os.path.join("model", population.replace("popu", "mpm")), int(population[4:7]))

    return history

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
    global record

    # Query if exists
    if block in record: return record[block]

    probability = Counter()
    length = 0
    resolution = pattern.shape[0]

    # Calculate prob for each pattern
    for p in pattern: probability.update({"".join([ str(c) for c in p.tolist() ]): 1})

    # Get Length
    for p, prob in probability.items(): length += (prob / resolution) * math.log2(prob / resolution)

    # Record Length
    record[block] = length

    return length

def DataDescriptionLength(model, population):
    """ D_{Data} = -n \sum_{bb \in BB} \sum_{x \in bb} p(x) log_2{p(x)} """
    resolution = population.shape[0]

    totalLength = sum([ BlockDescriptionLength(block, population[:, np.array(block)].reshape(-1, len(block))) for block in model ])
    totalLength = -1 * resolution * totalLength

    return totalLength

def descriptionLength(model, population, epsilon = 0.8):
    """ D = D_{Model} + D_{Data} """
    return epsilon * ModelDescriptionLength(model, population.shape[0]) + DataDescriptionLength(model, population)

def report(model, population, indent = 0):
    """ Generate the model statistics report """
    result = {}

    for block in model:
        prob = Counter()
        for p in population[ :, np.array(block) ]:
            prob.update({"".join([ str(c) for c in p.tolist() ]): 1})
        result[block] = prob

    # Print the Prob
    pprint.PrettyPrinter().pprint(result)

    return  

def merge(a, b):
    """ Merge the block """
    return tuple(set(a) | set(b))

def diff(prev, present, population):
    """ Compare the difference of the model """
    if len(set(prev) - set(present)):
        print("Difference Blocks in previous generation: ", set(prev) - set(present))
        print("Difference Blocks in present generation: ", set(present) - set(prev))
        print("Reduced: {}".format(descriptionLength(prev, population) - descriptionLength(present, population)))
    else:
        print("No difference between previous and present generation")   
 
    return bool(len(set(prev) - set(present)))

def isSaturated(model, population):
    staurated = []
    return staurated

def main(opt):
    global record

    # Initailize
    record = {}

    # Load Population
    newestGeneration = sorted(os.listdir("./population"))[-1] if (opt.num is None) else "popu{}.txt".format(str(opt.num).zfill(3))
    population = loadPopulation(os.path.join("./population", newestGeneration)).astype(np.uint8)
    print("Loaded: {}".format(os.path.join("./population", newestGeneration)))

    # Initialize Model
    model = [ (i, ) for i in range(0, population.shape[1]) ]

    # Greedy Search
    while True:
        # Initialize
        candidate = None
        minimumDescriptionLength = descriptionLength(model, population)

        # Get New Model
        iterlist = [ c for c in itertools.combinations(model, 2) ]
        np.random.shuffle(iterlist)

        # Search For New Model
        for selectElements in iterlist:
            newModel = list(set(model) - set(selectElements))
            newModel.append(merge(*selectElements))
            newLength = descriptionLength(newModel, population)

            # Update if find new candidate
            if (newLength < minimumDescriptionLength):
                candidate = selectElements
                minimumDescriptionLength = newLength

        # End Condition
        if candidate is None: break

        # Update for next round searching
        model = sorted(list(set(model) - set(candidate)), key=lambda x: x[0])
        model.append(merge(*candidate))
        print("Merge ({}, {})".format(candidate[0], candidate[1]))

    # Sorted
    model = [ tuple(sorted(block)) for block in sorted(model, key=lambda x: x[0]) ]

    # Output Model and Score
    print("MDL: {} / {}".format(ModelDescriptionLength(model, population.shape[0]), DataDescriptionLength(model, population)))

    # Print report
    if not opt.mute: report(model, population)

    # Output
    outputFile = os.path.join("model", newestGeneration.replace("popu", "mpm"))
    with open(outputFile, 'w') as textfile:
        textfile.writelines([str(len(model)), "\n"])
        for block in model: textfile.writelines(str(len(block)) + " " + " ".join([ str(b) for b in block ]) + "\n")

    # Find difference
    if os.path.exists(outputFile.replace(newestGeneration[4:7], str(int(newestGeneration[4:7]) - 1).zfill(3))):
        prevModel = sorted(loadModel(outputFile.replace(newestGeneration[4:7], str(int(newestGeneration[4:7]) - 1).zfill(3))))
        diff(prevModel, model, population)
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int)
    parser.add_argument("-a", "--all", action="store_true", default=False)
    parser.add_argument("-m", "--mute", action="store_true", default=False)
    parser.add_argument("-s", "--server", action="store_true", default=False)
    parser.add_argument("-e", "--epsilon", type=float, default=0.5)
    parser.add_argument("--history", action="store_true", default=False)
    opt = parser.parse_args()

    models = {}

    if opt.history:
        history = parseHistory(getHistory(HISTORY_URL.format(STUDENT_ID)))
        print(history)

    elif opt.server:
        if not opt.all:
            population = getHistory(HISTORY_URL.format(STUDENT_ID))[-1]['filename']
            history = oneRun(population, opt)
            population, fitness = history['filename'], history['fitness']

            begin, end = findnth(fitness, ':', 2) + 1, findnth(fitness, ' ', 2)
            fitness = [ float(num) for num in fitness[begin:end].split('/') ]

            print("MAX: {}, MEAN: {}, MIN: {}".format(fitness[0], fitness[1], fitness[2]))
        
        else:
            population = "popu000.txt"
            opt.mute = True
    
            while True:
                opt.num = int(population[4:7])            
                history = oneRun(population, opt)
                population, fitness = history['filename'], history['fitness']

                begin, end = findnth(fitness, ':', 2) + 1, findnth(fitness, ' ', 2)
                fitness = [ float(num) for num in fitness[begin:end].split('/') ]

                print("MAX: {}, MEAN: {}, MIN: {}".format(fitness[0], fitness[1], fitness[2]))

                # If find the optimum
                if fitness[0] == 100:
                    chromosome = history['fitness'].strip()[-50:]
                    with open('./optimum.txt', 'w') as file: file.write(chromosome)
                    print("Optimum Solution: {}".format(chromosome))

                    break

                # If loss diversity
                if fitness[0] - fitness[1] < EPSILON:
                    break

    elif opt.all:
        raise NotImplementedError

    else:
        main(opt)
