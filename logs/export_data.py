import networkx as nx
import pandas as pd
import numpy as np
import json
import math
import sys
import os

def number_of_hidden_nodes(G):
    path_list = []
    
    for j in range(6):
        if(G.has_node(j)):
            if (nx.has_path(G,j,6)):
                for path in nx.shortest_simple_paths(G, j, 6):
                    path_list = np.append(path_list, (len(path)-2))
            
    for j in range(6):
        if(G.has_node(j)):
            if (nx.has_path(G,j,7)):
                for path in nx.shortest_simple_paths(G, j, 7):
                    path_list = np.append(path_list, (len(path)-2))
    
    return np.max(int(np.max(path_list)),0)

def calculate_hidden_nodes(df, gen_number, individual_number):
    gen_dataframe = pd.DataFrame(df["generations"])
    individuals = gen_dataframe['individuals']

    generation_number = gen_number
    individual_number = individual_number

    individual = individuals.iloc[generation_number][individual_number]

    connections = individual['conn_genes']
    nodes       = individual['node_genes']

    coordinate_lst = []
    hidden_lst     = []

    for key in connections.keys():
        if connections[key][4] == True: 
            i = connections[key][1]
            j = connections[key][2]

            if (i > 7 or j > 7):
                hidden_lst.append((i,j))
            else:
                coordinate_lst.append((i,j))

    G = nx.DiGraph()
    G.add_edges_from(coordinate_lst)

    if nx.has_path(G,6,7):
        G.remove_edge(6,7)
    if nx.has_path(G,7,6):
        G.remove_edge(7,6)

    for i in range(len(hidden_lst)):
        G.add_edge(hidden_lst[i][0], hidden_lst[i][1])
    
    return G.number_of_edges(), number_of_hidden_nodes(G)

def export_numbers(df): 
    gen_dataframe = pd.DataFrame(df["generations"])
    individuals = gen_dataframe['individuals']

    total_fitness_lst = []
    total_nodes_lst   = []
    total_edges_lst   = []
    
    number_of_indiduals = []
    
    for i in range(len(individuals)):                                          #iterate over generation
        fitness_lst = []
        hiddenn_lst = []
        edges___lst = []

        for j in range(len(individuals.iloc[i])):                              #iterate over individuals
            individual   = (((individuals.iloc[i])[j])['stats'])['fitness']
            edges, hidden_nodes = calculate_hidden_nodes(df, i, j)

            fitness_lst.append(individual)
            hiddenn_lst.append(hidden_nodes)
            edges___lst.append(edges)
        
            number_of_indiduals.append(len(individuals.iloc[i]))

        total_fitness_lst.append(fitness_lst)
        total_nodes_lst.append(hiddenn_lst)
        total_edges_lst.append(edges___lst)

    columns = []
    for i in range(np.max(number_of_indiduals)):
        Ind = "Ind " + str(i)
        columns.append(Ind)

    fitness_scores = pd.DataFrame(total_fitness_lst, columns= columns)
    hidden__scores = pd.DataFrame(total_nodes_lst,   columns= columns)
    edges___scores = pd.DataFrame(total_edges_lst,   columns= columns)

    hidden__scores = hidden__scores.transpose()
    fitness_scores = fitness_scores.transpose()
    edges___scores = edges___scores.transpose()
    
    fitness_file_name = "dir_" + str(filename) + "/" + filename + "_fitness.csv"
    hiddenn_file_name = "dir_" + str(filename) + "/" + filename + "_hidLayr.csv"
    edges___file_name = "dir_" + str(filename) + "/" + filename + "_nredges.csv"
    
    fitness_scores.to_csv(fitness_file_name)
    hidden__scores.to_csv(hiddenn_file_name)    
    edges___scores.to_csv(edges___file_name)    


filename = sys.argv[1]
dirname = "dir_" + str(filename)

try:
    os.mkdir(dirname)
except:
    print "directory already present"

df = json.load(open(filename))
export_numbers(df)

