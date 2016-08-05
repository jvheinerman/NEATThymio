import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
import json
import math

def create_images(df, gen_number, ind_number):
    gen_dataframe = pd.DataFrame(df["generations"])
    individuals = gen_dataframe['individuals']
    
    generation_number = gen_number
    individual_number = ind_number

    individual = individuals.iloc[generation_number][individual_number]

    connections = individual['conn_genes']
    nodes       = individual['node_genes']

    coordinate_lst = []
    hidden_lst     = []
    highest_number = []

    for key in connections.keys():
        if connections[key][4] == True: 
            i = connections[key][1]
            j = connections[key][2]

            if (i > 7 or j > 7):
                hidden_lst.append((i,j))
                temp = max(i,j)
                if temp > highest_number:
                    highest_number = temp
            else:
                coordinate_lst.append((i,j))

    plt.figure(figsize=(10,7))
    G = nx.Graph()
    G.add_edges_from(coordinate_lst)

    if highest_number >7:
        fixed_positions = {0:(0,0) , 1:(0,2) , 2:(0,4) , 3:(0,6) , 4:(0,8) , 5:(0,10) , 6:(8,0), \
            7:(8,10), 8:(6.5,6.5), 9:(7,5), 10:(6.5,3.5), 11:(7,8), 12:(7,2), \
                13:(6,5), 14:(5.5,5), 15:(4.5,5), 16:(5,5)}

    for i in range(len(hidden_lst)):
        G.add_edge(hidden_lst[i][0], hidden_lst[i][1])


    colors = ["blue" if (i,j) in coordinate_lst else "red" for (i,j) in G.edges()]

    fixed_nodes = fixed_positions.keys()

    pos = nx.draw_networkx(G,pos=fixed_positions, fixed = fixed_nodes, edge_color=colors)
    
    gen_text = "dir_" + str(filename) + "/gen_" + str(gen_number) +"_ind_" + str(ind_number) + ".png"
    gen      = "Generation: " + str(gen_number)
    plt.title(gen)
    plt.savefig(gen_text, format="PNG")

    plt.clf()
    plt.close()


filename = sys.argv[1]
df = json.load(open(filename))

dirname = "dir_" + str(filename)

try:
    os.mkdir(dirname)
except:
    print "Directory already present"


gen_dataframe = pd.DataFrame(df["generations"])
individuals = gen_dataframe["individuals"]

for i in range(len(individuals)):                                          #iterate over generation
    for j in range(len(individuals.iloc[i])):                              #iterate over indivuduals
        create_images(df, i, j)