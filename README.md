# Network-Analysis

1. Network generation
    - Choose two Wikipedia pages containing sufficiently long text (e.g., cat), and construct a network for
    each page (only the main content part, and not references, ‘see also’, external links, etc.).
    - Consider each word as a node.
    - Put an unweighted, directed or undirected, edge between a word and a neighboring word in a
    sentence. In other words, a word has edges with n1 preceding words and n2 following words (e.g.,
    n1=n2=1, n1=n2=2, n1 =2 & n2=0, etc.). As an extreme, you can put edges between all word pairs in a
    sentence.
    - Ignore capital/small cases. You can also simply ignore punctuation marks, math symbols, etc.
2. Network topology analysis (undirected)
    - Examine the degrees of the nodes, including mean degree, density, and degree distribution. Discuss
    the results.
    - Obtain the graph Laplacian, and examine its eigenvalues. Verify that all eigenvalues are non-negative
    and the smallest eigenvalue is zero. Check the second smallest eigenvalue.
    - Examine the clustering coefficient.
    - Draw the network. Do you find any interesting observations?
3. Node analysis (directed)
    - Obtain various centrality measures of the nodes, including eigenvector centrality, Katz centrality,
    PageRank, closeness centrality, betweenness centrality, etc.
    - Which nodes (i.e., words) have large or small centrality values? Why?
    - Examine pair-wise correlation between different centrality measures (e.g., closeness vs. PageRank).
4. Comparison
    - Compare the above analysis results for the two networks. Discuss similarities and dissimilarities.

## Requirements

- Ubuntu 16.04
- Python 3.6
- Network 2.3.0

## Testing

```bash
# for testing undirected_graph.py
pip install undirected_graph.py

# for testing directed_graph.py
pip install directed_graph.py
```
