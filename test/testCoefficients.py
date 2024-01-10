import polytensor
import networkx as nx


def testInterface():
    print("\nTest 1: Generate Coefficients")

    # What's the ideal interface for this package?
    # I want to be able to do the following:
    # 1. Specify a polynomial by connectivity.
    #   a. List of tuples.
    #   b. NetworkX Graph.
    #   c. Specify by a
    # 2.
    # 3. Specify the connectivity.
    #
    g = nx.Graph()

    g.add_node(1, weight=1)
    g.add_node(0, weight=0)
    g.add_node(2, weight=0)
    g.add_edge(1, 0, weight=1)
    g.add_edge(2, 0, weight=1)
    sp = polytensor.generators.from_networkx(g, key="weight")

    print(sp)

    print(polytensor.generators.coeffPUBORandomSampler(10, [2, 3], lambda: 1))
