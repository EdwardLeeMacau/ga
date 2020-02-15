import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def UniformXO():
    df = pd.read_csv('./sga-python/100_uniform.csv')
    truncated_index = df.where(df.Generation.diff() < 0).dropna().index
    
    index = 0 
    for i in truncated_index:
        generation   = df.loc[index: i - 1, 'Generation'].to_numpy()[1:]
        fitness_diff = df.loc[index: i - 1, 'Fitness'].diff().to_numpy()[1:]
        variance     = df.loc[index: i - 1, 'Variance'].to_numpy()[:-1]

        selectionIntensity = fitness_diff / np.sqrt(variance)

        plt.plot(generation, selectionIntensity, linewidth=0.25)

        index = i
    
    plt.plot(generation, [0.57629] * generation.shape[0], label='Theoretical Value')
    plt.title('Selection Intensity in OneMax Problem (UniformXO)')
    plt.ylabel('Selection Intensity')
    plt.xlabel('Generation')
    plt.savefig('./p2-2.png')

    return

def OnePointXO():
    df = pd.read_csv('./sga-python/100_onepoint.csv')
    truncated_index = df.where(df.Generation.diff() < 0).dropna().index
    
    index = 0 
    for i in truncated_index:
        generation   = df.loc[index: i - 1, 'Generation'].to_numpy()[1:]
        fitness_diff = df.loc[index: i - 1, 'Fitness'].diff().to_numpy()[1:]
        variance     = df.loc[index: i - 1, 'Variance'].to_numpy()[:-1]

        selectionIntensity = fitness_diff / np.sqrt(variance)

        plt.plot(generation, selectionIntensity, linewidth=0.25)

        index = i
    
    plt.plot(generation, [0.57629] * generation.shape[0], label='Theoretical Value')
    plt.title('Selection Intensity in OneMax Problem (OnePointXO)')
    plt.ylabel('Selection Intensity')
    plt.xlabel('Generation')
    plt.savefig('./p2-3.png')

    return

def populationXO():
    df = pd.read_csv('./sga-python/100_population.csv')
    truncated_index = df.where(df.Generation.diff() < 0).dropna().index
    
    index = 0 
    for i in truncated_index:
        generation   = df.loc[index: i - 1, 'Generation'].to_numpy()[1:]
        fitness_diff = df.loc[index: i - 1, 'Fitness'].diff().to_numpy()[1:]
        variance     = df.loc[index: i - 1, 'Variance'].to_numpy()[:-1]

        selectionIntensity = fitness_diff / np.sqrt(variance)

        plt.plot(generation, selectionIntensity, linewidth=0.25)

        index = i
    
    plt.plot(generation, [0.57629] * generation.shape[0], label='Theoretical Value')
    plt.title('Selection Intensity in OneMax Problem (PopulationXO)')
    plt.ylabel('Selection Intensity')
    plt.xlabel('Generation')
    plt.savefig('./p2-4.png')

    return

def main():
    UniformXO()
    plt.clf()
    OnePointXO()
    plt.clf()
    populationXO()
    plt.clf()
    return

if __name__ == "__main__":
    main()