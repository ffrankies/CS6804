import networkx as nx
import pandas as pd
import itertools
import copy
import matplotlib.pyplot as plt
import sys
import csv

tsv_file = open("/Users/bipashabanerjee/Documents/CS/sem5/SGML/CS6804/cora/cora_graph.tsv")
# read_tsv = csv.reader(tsv_file, delimiter="\t")
# for row in read_tsv:
#   print(row)

edgelist = pd.read_csv(tsv_file, delimiter="\t")
graph = nx.Graph()

for i, elrow in edgelist.iterrows():
    graph.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())
