"""
El problema de las reinas consiste en poner el mismo numero de reinas que filas en un tablero de ajedrez.
La dificultad consiste en conseguir que ninguna reina mate a otra
Para resolver este problema utilizo la programación genética, que emula la selección natural
"""
import numpy as np
import random

#Defino variables globales que no debo modificar
INITIAL_POPULATION = 2000
BOARD_SIZE = 8
MUTATION_CHANCE = 0.1
MAX_GENERATIONS = 1000

global_population = []
fitness = []

"""
DEFINIR AGENT, POBLACIÓN Y DISPLAY
Un agent es un tablero 
Este tablero lo defino como una lista en la que la posición me define la columna y el valor la fila
Por ejemplo la lista [1,3,0,2] sería este tablero, en el que resolveriamos el problema.
Eso se debe a que ninguna reina está en la trayectoria de otra

 0 0 1 0
 1 0 0 0
 0 0 0 1
 0 1 0 0 

"""

#Genero un tablero
def init_agent(board_size):
    agent = [i for i in range(board_size)]
    random.shuffle(agent)
    return agent

#Imprimir el tablero
def display_agent(agent):
    board = np.zeros((len(agent),agent.__len__()))
    for x,a in enumerate(agent):
        board[a,x] = 1
        x += 1
    print (board)
    return board


#Genero al total de los tableros que quiero probar
def init_population(initial_population_size,board_size):
    population = []
    for _ in range(0,int(initial_population_size)):
        population.append(init_agent(board_size))
    return population


"""
Aquí defino el fitness, que és la calidad del resultado obtenido.
Para ello compruebo las colisiones entre las reinas
"""

def fitness_agent(queens):
    colisiones = 0
    n = len(queens)
    for i in range(n):
        for j in range(i + 1,n):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == abs(i-j):
                colisiones += 1
    return colisiones



def fitness_population(population):
    fitness_list = []
    for p in population:
        fitness_list.append(fitness_agent(p))
    return fitness_list


"""
En esta función defino la reproducción de los datos, en la que cojo diferentes muestras de la población.
Primero cojo la primera mitad del padre y luego cojo la parte de la madre que no esté ya en el hijo
"""
def crossover(parent1,parent2):
    half = int(len(parent1)//2)
    child1 = parent1[0:half]
    child2 = parent2[0:half]
    for i in parent2:
        if child1.count(i) == 0:
            child1.append(i)
    for i in parent1:
        if child2.count(i) == 0:
            child2.append(i)
    return child1, child2



#En la mutación lo que hago es generar posiciones aleatorias a un agente para que no se base siempre en el padre
def mutation(agent):
    for _ in range(25):
        position1 = random.randint(0,len(agent) -1)
        position2 = random.randint(0,len(agent) -1)
        value1 = agent[position1]
        value2 = agent[position2]
        agent[position1] = value2
        agent[position2] = value1
    return agent

def mutation_with_chance(agent,mutation_chance):
    if random.random() <= mutation_chance:
        mutation(agent)
    return agent
    

def replicate(population, mutation_chance):
    #Mezclo la población que tengo para ue no se mezclen siempre los mejores con los mejores y luego hago que todos hagan crossover
    random.shuffle(population)
    children = []

    for i in range(0,len(population)-1,2):

        child1, child2 = crossover(population[i],population[i+1])

        children.append(mutation_with_chance(child1,mutation_chance))
        children.append(mutation_with_chance(child2,mutation_chance))

    return children

#Filtro la población según el fitness Test para obtener los mejores
def sort_population(population,fitness_list):
    combined = list(zip(population,fitness_list))
    combined.sort(key=lambda x: x[1])
    return [individual for individual,_ in combined]

def select_population(population,board_size):
    pop_len = int(len(population))
    #Elijo al 60% de la población con mejores resultados
    first_half = population[:int(pop_len*0.6)]

    #Elijo el 20% peor de la población
    last_twenty = population[int(pop_len*0.8):]

    #Genero población nueva que me ayude a darle variancia a mis resultados para que puedan mejorar 
    immigrants = init_population(pop_len*0.2,board_size)

    population[::] = first_half[::]
    population.extend(last_twenty)
    population.extend(immigrants)

    return population
  

def main(population_size,board_size,max_gens,mutation_chance):
    best_solution = [] 
    population = init_population(population_size,board_size)
    #Defino un maximo para que no tienda al infinito
    for generation in range(max_gens):
        new_population = replicate(population,mutation_chance)
        population_with_fitness = [(agent,fitness_agent(agent)) for agent in population]
        population_with_fitness.sort(key=lambda x: x[1])
        best_agent,best_fitness = population_with_fitness[0]
        best_solution.append((best_agent,best_fitness)) 
        if best_fitness == 0:
            print(f"Best fitness found {best_fitness}")
            break
        selected_population = select_population(sort_population(new_population,[fit for _,fit in population_with_fitness]),board_size)
        population = replicate(selected_population,mutation_chance)
        print(f"Generation {generation + 1}: Best fitness(colisiones) = {(best_fitness)}")
        generation += 1
    perfect_solution = min(best_solution,key=lambda x: x[1])
    best_agent_overall,best_fitness_overall = perfect_solution
    print(f"Best Agent found: {best_agent_overall}")
    print(f"Best solution found: {int(best_fitness_overall)}")
    display_agent(best_agent_overall) 
    return best_solution 
agent = main(INITIAL_POPULATION,BOARD_SIZE,MAX_GENERATIONS,MUTATION_CHANCE)

