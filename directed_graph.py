import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import operator

import collections

from scipy.stats import pearsonr 

os.environ["CUDA_VISIBLE_DEVICES"]= "0" # gpu 0

def draw(G, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    labels = nx.draw_networkx_labels(G, pos, font_size=10)
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray')

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

class DirectedNetworkAnalysis():

    def __init__(self, network):
        self.network = network
        self.n = self.network.number_of_nodes()
        self.m = self.network.number_of_edges()

    def getInDegree(self):
        return self.network.in_degree()

    def getOutDegree(self):
        return self.network.out_degree()
        
    def getDensity(self):
        densitiy = (self.m)/(self.n*(self.n-1))
        print("\n# node ..")
        print(self.n)
        print("\n# edge ..")
        print(self.m)
        return densitiy
        
    def getEigenVectorCentrality(self):
        centrality = nx.eigenvector_centrality(self.network)

        # top-10 and bottom-10 by value
        print("\neigen vector top-10, bottom-10 ..")
        sortedByValue = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
        sortedByValue = dict(sorted(sortedByValue.items(), key=operator.itemgetter(1),reverse=True))
        # print(sortedByValue)
        cnt = 0
        for n, c in sortedByValue.items():
            if cnt < 10 or cnt > len(sortedByValue)-10-1:
                if cnt == 0:
                    print("\ntop 10 ..")
                if cnt == len(sortedByValue)-10:
                    print("\nbottom 10 ..")
                print("%s %0.5f"%(n, c))
            cnt = cnt + 1

        return centrality
    
    def getKatzCentrality(self):
        phi = (1+math.sqrt(5))/2.0 # largest eigenvalue of adj matrix
        centrality = nx.katz_centrality_numpy(self.network, 1/phi)
        
        ## top-10 and bottom-10 by key
        # for n,c in sorted(centrality.items()):
        #     print("%s %0.5f"%(n,c))

        # top-10 and bottom-10 by value
        print("\nkatz top-10, bottom-10 ..")
        sortedByValue = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
        sortedByValue = dict(sorted(sortedByValue.items(), key=operator.itemgetter(1),reverse=True))
        # print(sortedByValue)
        cnt = 0
        for n, c in sortedByValue.items():
            if cnt < 10 or cnt > len(sortedByValue)-10-1:
                if cnt == 0:
                    print("\ntop 10 ..")
                if cnt == len(sortedByValue)-10:
                    print("\nbottom 10 ..")
                print("%s %0.5f"%(n, c))
            cnt = cnt + 1

        return centrality

    def getPageRankCentrality(self):
        centrality = nx.pagerank(self.network, alpha=0.9, max_iter=500)

        # top-10 and bottom-10 by value
        print("\npagerank top-10, bottom-10 ..")
        sortedByValue = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
        sortedByValue = dict(sorted(sortedByValue.items(), key=operator.itemgetter(1),reverse=True))
        # print(sortedByValue)
        cnt = 0
        for n, c in sortedByValue.items():
            if cnt < 10 or cnt > len(sortedByValue)-10-1:
                if cnt == 0:
                    print("\ntop 10 ..")
                if cnt == len(sortedByValue)-10:
                    print("\nbottom 10 ..")
                print("%s %0.5f"%(n, c))
            cnt = cnt + 1
        return centrality

    def getClosenessCentrality(self):
        centrality = nx.closeness_centrality(self.network)

        # top-10 and bottom-10 by value
        print("\ncloseness top-10, bottom-10 ..")
        sortedByValue = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
        sortedByValue = dict(sorted(sortedByValue.items(), key=operator.itemgetter(1),reverse=True))
        # print(sortedByValue)
        cnt = 0
        for n, c in sortedByValue.items():
            if cnt < 10 or cnt > len(sortedByValue)-10-1:
                if cnt == 0:
                    print("\ntop 10 ..")
                if cnt == len(sortedByValue)-10:
                    print("\nbottom 10 ..")
                print("%s %0.5f"%(n, c))
            cnt = cnt + 1

        return centrality

    def getBetweennessCentrality(self):
        centrality = nx.betweenness_centrality(self.network)

        # top-10 and bottom-10 by value
        print("\nbetweenness top-10, bottom-10 ..")
        sortedByValue = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
        sortedByValue = dict(sorted(sortedByValue.items(), key=operator.itemgetter(1),reverse=True))
        # print(sortedByValue)
        cnt = 0
        for n, c in sortedByValue.items():
            if cnt < 10 or cnt > len(sortedByValue)-10-1:
                if cnt == 0:
                    print("\ntop 10 ..")
                if cnt == len(sortedByValue)-10:
                    print("\nbottom 10 ..")
                print("%s %0.5f"%(n, c))
            cnt = cnt + 1

        return centrality

    def getPearsonCorrelation(self):
        pearson = nx.degree_pearson_correlation_coefficient(self.network)
        return pearson

    def showDirectedGraph(self):
        pos = nx.spring_layout(self.network, k=0.15, iterations=20)
        nx.draw(self.network, pos, node_color='yellow', edge_color='gray', font_size=10, font_weight='bold', with_labels=True)

        plt.tight_layout()
        plt.show()
        plt.savefig("DirectedGraph.png", format="PNG")


if __name__ == '__main__':

    ## Text preprocessing
    file1 = open('./data/fish.txt', "r")
    file2 = open('./data/computer.txt', "r")

    content2 = file2.read()
    content2 = " ".join(re.split("[^a-zA-Z]*", content2)) 
    content_list2 = content2.split(" ")

    n1 = 2 # preceding
    n2 = 0 # following

    for i, text in enumerate (content_list2):
        content_list2[i] = text.lower()

    content_list2 = ' '.join(content_list2).split()

    print("\ncheck content ..")
    print(content_list2)

    print("\ncontent length ..")
    print(len(content_list2))

    with open("./data/nodes_computer.txt", "w") as nodes_file, open("./data/edges_computer.txt", "w") as edges_file:
        for i, _ in enumerate (content_list2):
            nodes_file.write(content_list2[i]+"\n")
            if len(content_list2) - i > n1: 
                for j in range (1, n1+1):
                    edge = content_list2[i] + " " + content_list2[i+j]
                    edges_file.write(edge+"\n")

    ############################################################################################################################

    # Make directed graph
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Directed Graph', fontsize=10)

    directed_graph = nx.DiGraph()
    nodes = nx.read_adjlist("./data/nodes_computer.txt")
    # nodes = nx.read_adjlist("./data_test/nodes.txt")
    with open("./data/edges_computer.txt", "r") as edges_file:
    # with open("./data_test/edges.txt", "r") as edges_file:
        lines = edges_file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            edge = line.split(' ')
            directed_graph.add_edges_from([(edge[0], edge[1])])
    directed_graph.add_nodes_from(nodes)

    # print(directed_graph.nodes())
    # print(directed_graph.edges())

    # Declare UndirectedNetworkAnalysis Class
    directedNet = DirectedNetworkAnalysis(directed_graph)

    # Get in-degree, out-degree
    print("\nin-degree ..")
    print(directedNet.getInDegree())
    print("\nout-degree ..")
    print(directedNet.getOutDegree())

    # Get density
    densitiy = directedNet.getDensity()
    print("\ndensitiy ..")
    print(densitiy)

    # Get eigen vector centrality
    eigenVectorCentral = directedNet.getEigenVectorCentrality()
    # print("\neigenVectorCentral ..")
    # print(eigenVectorCentral)

    # Get katz centrality
    katzCentral = directedNet.getKatzCentrality()
    # print("\nkatzCentral ..")
    # print(katzCentral)

    # Get pagerank centrality
    pageRankCentral = directedNet.getPageRankCentrality()
    # print("\npageRankCentral ..")
    # print(pageRankCentral)

    # Get closeness centrality
    closenessCentral = directedNet.getClosenessCentrality()
    # print("\closenessCentral ..")
    # print(closenessCentral)

    # Get betweenness centrality
    betweennessCentral = directedNet.getBetweennessCentrality()
    # print("\nbetweennessCentral ..")
    # print(betweennessCentral)
    
    # Get pearson correlation
    pearson_for_closeness = directedNet.getPearsonCorrelation()
    pearson_for_pagerank = directedNet.getPearsonCorrelation()
    print("\npearson closeness vs pagerank ..")
    print("%3.1f"%pearson_for_closeness)
    print("%3.1f"%pearson_for_pagerank)

    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []

    for k, v in eigenVectorCentral.items():
        list1.append(v)

    for k, v in katzCentral.items():
        list2.append(v)

    for k, v in pageRankCentral.items():
        list3.append(v)

    for k, v in closenessCentral.items():
        list4.append(v)

    for k, v in betweennessCentral.items():
        list5.append(v)

    # Get correlation
    corr, _ = pearsonr(list4, list3) 
    print("\npearson closeness vs pagerank .. ")
    print("%.3f" % corr)

    # Visualization directed graph
    directedNet.showDirectedGraph() ## visualization

    # pos = nx.spring_layout(directed_graph, seed=675)
    # draw(directed_graph, pos, nx.degree_centrality(directed_graph), 'Degree Centrality')

    ############################################################################################################################
    

   