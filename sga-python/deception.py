import numpy as np
import pprint
import itertools
from collections import Counter

LENGTH   = 4
REPEAT   = 10 ** 6
MAXIMUM  = (2 ** LENGTH) - 1
IDXARRAY = np.arange(0, MAXIMUM)
pprint   = pprint.PrettyPrinter().pprint

LOG_INTERVAL = 10 ** 4

int2BitString = {index: ''.join(string) for index, string in enumerate(itertools.product('01', repeat=LENGTH))}
bitString2Int = {value: key for key, value in int2BitString.items()}

def expandSchema(schema: str):
    stack  = [schema]
    result = []

    while not (len(stack) == 0):
        s = stack.pop()

        if (s.find('*') == -1):
            result.append(s)
        else:
            stack.append(s.replace('*', '0', 1))
            stack.append(s.replace('*', '1', 1))

    return tuple(result)

def average(fitness, schemas: list):
    return np.average([ fitness[bitString2Int[schema]] for schema in schemas ])

def schemaComp(schema):
    results = []

    for order in range(1, len(schema)):
        masks = list(itertools.combinations(range(len(schema)), order))

        for mask in masks:
            superior = schema

            for m in mask:
                superior = superior[:m] + '*' + superior[m + 1:]

            flexible_bits = [ ''.join(string) for string in itertools.product('01', repeat=len(schema) - len(mask))]

            schemas = []
            for c in flexible_bits:
                for m in mask:      c = c[:m] + '*' + c[m:]
                if c == superior:   continue

                schemas.append(c)

            results.append((superior, schemas))

    return results

def deception(fitness):
    """ Check if the fitness function is deception """
    index       = np.argmax(fitness)        # Maximum Value
    complement  = MAXIMUM - index           # Complement index
    c_bitString = int2BitString[complement] # Complement index in bitString
    schemas     = schemaComp(c_bitString)

    for source, targets in schemas:
        f_s = average(fitness, expandSchema(source))

        for target in targets:
            f_t = average(fitness, expandSchema(target))

            if f_s < f_t:   return False

    return True

def main():
    countDeception = 0

    for i in range(1, REPEAT + 1):
        fitness = np.random.rand(2, 2, 2, 2).flatten()
        
        if deception(fitness): 
            countDeception += 1
        
        if i % LOG_INTERVAL == 0:
            print("{} / {}, ({:.2%})".format(countDeception, i, countDeception / i))

if __name__ == '__main__':
    main()