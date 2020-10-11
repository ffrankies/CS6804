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

edgelist = pd.read_csv(tsv_file, delimiter="\t")
graph = nx.Graph()

for i, elrow in edgelist.iterrows():
    graph.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())

lp = list(community.label_propagation_communities(graph))
lpcoms = [tuple(x) for x in lp]
algo_graph_nparray = numpy.array(lpcoms)
#To run the evaluation script

true_tsv_df = pd.read_csv("/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/cora_truePartition.tsv",sep="\t")
true_value_nparray = true_tsv_df.values
# print(type(true_tsv_df.values))

evaluate_partition(true_value_nparray,algo_graph_nparray)