import networkx as nx
import pandas as pd
import itertools
import copy
import numpy
import matplotlib.pyplot as plt
import sys
import csv
from networkx.algorithms import community
from evaluateGraph import *

tsv_file = open("/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/cora_graph.tsv")
# read_tsv = csv.reader(tsv_file, delimiter="\t")
# for row in read_tsv:
#   print(row)
output_file = open("/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/output.txt","a")
edgelist = pd.read_csv(tsv_file, delimiter="\t")
graph = nx.Graph()

for i, elrow in edgelist.iterrows():
    graph.add_edge(elrow[0], elrow[1])

lp = list(community.label_propagation_communities(graph))
lpcoms = [tuple(x) for x in lp]
i = 0
#The following code block has been blocked since it has been executed and outputed to a file
# for list_1 in lpcoms:
  
#     for l in list_1:
#         # print(type(l))
#         #rint(str(l) + "\t"+ str(i))
#        # output_file.write(str(l) + "\t"+ str(i)+"\n")
#     i = i +1 
#Read the file and sort based on the node number
algo_graph_list=[]
lines = open('/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/output.txt', 'r').readlines()
for line in sorted(lines, key=lambda line: line.split()[0]):
    # print(line.split('\t')[1])
    algo_graph_list.append(int(line.split('\t')[1].strip()))
algo_graph_nparray= np.array(algo_graph_list)
# algo_graph_nparray = numpy.array(lpcoms)
# #To run the evaluation script
arr=[]

with open("/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/cora_truePartition.tsv","r") as csv_file:
    true_pred = list(csv.reader(csv_file, delimiter="\t"))
    for each_row in true_pred:

        arr.append((int(each_row[1])-1))
    true_value_nparray = np.array(arr)



evaluate_partition(true_value_nparray,algo_graph_nparray)