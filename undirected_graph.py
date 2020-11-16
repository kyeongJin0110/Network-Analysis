import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

import collections

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

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

class UndirectedNetworkAnalysis():

    def __init__(self, network):
        self.network = network
        self.n = self.network.number_of_nodes()
        self.m = self.network.number_of_edges()
        
    def getDegree(self):      
        degrees = [val for (node, val) in self.network.degree()]
        print("\ndegrees ..")
        print(degrees)
        print("\ndegrees length ..")
        print(len(degrees))
        mean_degree = np.mean(np.array(degrees))
        return mean_degree
    
    def getDensity(self):
        densitiy = (2*self.m)/(self.n*(self.n-1))
        print("\n# node ..")
        print(self.n)
        print("\n# edge ..")
        print(self.m)
        return densitiy
    
    def showDegreeDistribution(self):
        degree_sequence = sorted([d for n, d in self.network.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color="b")

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        Gcc = self.network.subgraph(sorted(nx.connected_components(self.network), key=len, reverse=True)[0])
        pos = nx.spring_layout(self.network)
        plt.axis("off")
        # nx.draw_networkx_nodes(self.network, pos, node_size=20)
        # nx.draw_networkx_edges(self.network, pos, alpha=0.4)
        plt.show()

    # def setGraphLaplacian(self, nodelist):
    #     self.laplacian = nx.laplacian_matrix(self.network, nodelist, weight=None)

    def getGraphLaplacian(self):

        # The adjacency matrix of the graph in numpy format
        A = nx.to_numpy_array(self.network) 

        # The degree matrix
        D = np.diag(A.sum(axis=1))

        # Unnormalized Laplacian 
        self.L = D - A

        return self.L

    def getEigenValues(self):

        # In general, eigenvalues can be complex. 
        # Only special types of matrices give rise to real values only. 
        # So, we’ll take the real parts only and assume that the dropped complex dimension does not contain significant information. 
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.L)  
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenvectors = np.real(self.eigenvectors)

        # order the eigenvalues from small to large:
        self.order = np.argsort(self.eigenvalues)  
        self.eigenvalues = self.eigenvalues[self.order]

        # The first eigenvalues
        # The first eigenvalue is as good as zero and this is a general fact; 
        # the smallest eigenvalue is always zero. The reason it’s not exactly zero above is because of computational accuracy.
        return self.eigenvalues[0:10]

    def showEigenValue(self):

        # 32-dimensional subspace of the full vector space
        embedding_size = 32
        v_0 = self.eigenvectors[:, self.order[0]]
        v = self.eigenvectors[:, self.order[1:(embedding_size+1)]] 

        plt.plot(self.eigenvalues)
        plt.show()
    
    def getClusteringCoefficient(self):
        triangles = nx.triangles(self.network)
        # nx.triangles(self.network,0)

        clusterCoeff = nx.clustering(self.network)
        return triangles, clusterCoeff
        

    def showUndirectedGraph(self):
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, node_color='yellow', edge_color='gray', font_size=10, font_weight='bold', with_labels=True)

        plt.tight_layout()
        plt.show()
        plt.savefig("UndirectedGraph.png", format="PNG")


if __name__ == '__main__':

    # np.set_printoptions(threshold=sys.maxsize)

    ## Text preprocessing
    file1 = open('./data/fish.txt', "r")
    file2 = open('./data/computer.txt', "r")

    content1 = file1.read()
    content1 = " ".join(re.split("[^a-zA-Z]*", content1)) 
    content_list1 = content1.split(" ")

    n1 = 2
    n2 = 2

    for i, text in enumerate (content_list1):
        content_list1[i] = text.lower()

    content_list1 = ' '.join(content_list1).split()

    print("\ncheck content ..")
    print(content_list1)

    print("\ncontent length ..")
    print(len(content_list1))

    with open("./data/nodes_fish.txt", "w") as nodes_file, open("./data/edges_fish.txt", "w") as edges_file:
        for i, _ in enumerate (content_list1):
            nodes_file.write(content_list1[i]+"\n")
            if len(content_list1) - i > n1: 
                for j in range (1, n1+1):
                    edge = content_list1[i] + " " + content_list1[i+j]
                    edges_file.write(edge+"\n")

    ############################################################################################################################                

    ## Make undirected graph
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Undirected Graph', fontsize=10)

    undirected_graph = nx.Graph()
    edges = nx.read_edgelist('./data/edges_fish.txt')
    nodes = nx.read_adjlist("./data/nodes_fish.txt")
    # edges = nx.read_edgelist('./data_test/edges.txt')
    # nodes = nx.read_adjlist("./data_test/nodes.txt")
    undirected_graph.add_edges_from(edges.edges())
    undirected_graph.add_nodes_from(nodes)

    # print(undirected_graph.nodes())
    # print(undirected_graph.edges())

    # Declare UndirectedNetworkAnalysis Class
    undirectedNet = UndirectedNetworkAnalysis(undirected_graph)

    # Get mean degree
    mean_degree = undirectedNet.getDegree()
    print("\nmean degree ..")
    print(mean_degree)

    # Get densitiy
    densitiy = undirectedNet.getDensity()
    print("\ndensitiy ..")
    print(densitiy)

    # Get degree distribution
    # undirectedNet.showDegreeDistribution() ## visualization

    # Get graph laplacian
    laplacian = undirectedNet.getGraphLaplacian()
    print("\nlaplacian ..")
    print(laplacian)

    # Get eigenvalue
    eigenvalue = undirectedNet.getEigenValues()
    print("\neigenvalue ..")
    print(eigenvalue)
    # Plot eigenvalue
    # undirectedNet.showEigenValue() ## visualization

    # Visualization undirected graph
    undirectedNet.showUndirectedGraph() ## visualization

    # Get clustering coefficient
    triangles, clusterCoeff = undirectedNet.getClusteringCoefficient()
    print("\ntriangles ..")
    print(triangles)
    print("\nclusterCoeff ..")
    print(clusterCoeff)
  
    # pos = nx.spring_layout(undirected_graph, seed=675)
    # draw(undirected_graph, pos, nx.degree_centrality(undirected_graph), 'Degree Centrality')