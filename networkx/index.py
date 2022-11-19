import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import powerlaw

from collections import Counter

def read_network():
    return nx.read_edgelist("./data.txt", create_using=nx.MultiGraph)

def print_network_info(description, network):
    print(description, nx.info(network))

def remove_network_duplicates_and_loops(network):
    network_without_duplicates = nx.Graph(network)
    network_without_duplicates.remove_edges_from(nx.selfloop_edges(network_without_duplicates))

    return network_without_duplicates

def count_degrees(G: nx.Graph):
    degrees = degrees_map(G)

    return Counter(degrees.values())


def degrees_list(G: nx.Graph):
    result = []
    for node in G.nodes:
        result.append(G.degree[node])

    return result


def degrees_map(G: nx.Graph):
    result = {}

    for node, degree in G.degree():
        result[node] = degree

    return result

def get_largest_connected_component(network):
    connected_components = sorted(nx.connected_components(network), key=len, reverse=True)
    
    return network.subgraph(connected_components[0])

def approximate_mean_path_length(network, tries: list):
    result = {}

    for try_number in tries:
        result[try_number] = mean_path_length(network, try_number)
    return result


def mean_path_length(network, try_number):
    sum = 0
    i = 0

    while i < try_number:
        nodes = random.sample(list(network.nodes), 2)

        try:
            path_length = nx.shortest_path_length(
                network, source=nodes[0], target=nodes[1])
            sum += path_length
            i += 1
        except nx.NetworkXNoPath:
            continue

    return sum / try_number

def get_highest_cores(network):
    result = {}

    core_numbers = nx.core_number(network)
    counted_core_numbers = Counter(core_numbers.values())

    sorted_counted_core_numbers = sorted(
        counted_core_numbers, key=counted_core_numbers.get, reverse=True)

    for core in sorted_counted_core_numbers[:3:]:
        result[core] = counted_core_numbers[core]

    return result

def print_nodes_degree_distribution(network):
    degrees_map = {}

    for node, degree in network.degree():
        degrees_map[node] = degree

    counted_degrees = Counter(degrees_map.values())
    degrees_sorted_ascending = sorted(counted_degrees, key=counted_degrees.get)

    y = []
    y_total = sum(counted_degrees.values())

    for degree in degrees_sorted_ascending:
        y.append(counted_degrees[degree] / y_total)

    plt.bar(degrees_sorted_ascending, y)
    plt.title('Rozkład stopni wierzchołków')
    plt.xlabel('stopień')
    plt.ylabel('p')

    plt.show()

def draw_hill_plot(network: nx.Graph):
    degrees = degrees_list(network)

    NBINS = 50
    bins = np.logspace(
        np.log10(min(degrees)), np.log10(max(degrees)), num=NBINS)
    bcnt, bedge = np.histogram(np.array(degrees), bins=bins)
    alpha = np.zeros(len(bedge[:-2]))

    for i in range(0, len(bedge)-2):
        fit = powerlaw.Fit(degrees, xmin=bedge[i], discrete=True)
        alpha[i] = fit.alpha

    plt.semilogx(bedge[:-2], alpha)
    plt.title('wykres Hilla')
    plt.show()

    return alpha





if __name__ == "__main__":
    network = read_network()
    # print_network_info("Init network info", network)

    network_without_duplicates = remove_network_duplicates_and_loops(network)
    # print_network_info("Network without loops and duplicates info", network_without_duplicates)

    # largest_connected_component = get_largest_connected_component(network_without_duplicates)
    # print_network_info("Largest connected component", largest_connected_component)

    # print(approximate_mean_path_length(network_without_duplicates, [100, 1000, 10000]))

    # print(get_highest_cores(network_without_duplicates))
    
    # print(print_nodes_degree_distribution(network_without_duplicates))

    print(draw_hill_plot(network_without_duplicates))


