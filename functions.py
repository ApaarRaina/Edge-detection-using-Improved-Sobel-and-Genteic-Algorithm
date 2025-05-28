import numpy as np
import random


def fitness(img, t):
    t = int(t, 2)
    w_o_mask = img < t
    w_1_mask = img >= t
    w_o = np.shape(img[w_o_mask])[0]
    w_1 = np.shape(img[w_1_mask])[0]
    mu_o = np.mean(img[w_o_mask])
    mu_1 = np.mean(img[w_1_mask])
    size = np.shape(img)[0] * np.shape(img)[1]
    if w_1 == 0:
        mu_1 = 0
    if w_o == 0:
        mu_o = 0

    return (w_o * w_1 * ((mu_o - mu_1) ** 2)) / (size ** 2)


def mating(population, img):
    fit = []
    for i in population:
        fit.append(fitness(img, i))

    fit = np.array(fit)
    mean_fit = np.mean(fit)
    n = (fit / mean_fit)
    mating_pool = []

    cnt = 0
    for r in n:
        for i in range(int(r)):
            mating_pool.append(population[cnt])

        cnt += 1

    return mating_pool, fit


def cross(gene_1, gene_2):
    index = random.randint(1, 7)

    head_1 = gene_1[:index]
    tail_1 = gene_1[index:]
    head_2 = gene_2[:index]
    tail_2 = gene_2[index:]

    gene_1 = head_2 + tail_1
    gene_2 = head_1 + tail_2

    return gene_1, gene_2


def crossover(mating_pool, population, P=0.9):
    gene = np.random.choice(mating_pool, size=2)
    gene_1, gene_2 = gene[0], gene[1]

    p = random.uniform(0, 1)
    if p < P:
        new_gene1, new_gene2 = cross(gene_1, gene_2)
    else:
        new_gene1, new_gene2 = gene_1, gene_2

    population = list(population)
    i1, i2 = population.index(gene_1), population.index(gene_2)
    if i1 > i2:
        population.pop(i1)
        population.pop(i2)
    elif i2 > i1:
        population.pop(i2)
        population.pop(i1)

    else:
        population.pop(i2)

    if i1 != i2:
        population.insert(i1, gene_1)
        population.insert(i2, gene_2)

    else:
        population.insert(i2, gene_2)

    return np.array(population)


def mutation(population, fit):
    population = list(population)
    probability = np.zeros(len(population))

    c = (np.max(fit) - fit) / (np.max(fit) - np.mean(fit) + 1e-6)  # to avoid divide by zero
    mask = c > 1

    probability[mask] = 0.1
    probability[~mask] = 0.002

    for i in range(len(population)):
        p = random.uniform(0, 1)
        if p < probability[i]:
            index = random.randint(0, len(population[i]) - 1)
            gene = list(population[i])  # convert string to list for mutation
            gene[index] = '0' if gene[index] == '1' else '1'
            population[i] = ''.join(gene)  # convert back to string

    return np.array(population)