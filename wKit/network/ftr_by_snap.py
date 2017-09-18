# coding=utf-8
# TODO: directed network
import datetime
from collections import defaultdict
import snap
from wKit.utility.time import costs

ud_node_deg_cntr = 'ud_node_degree_centrality'
ud_node_node_ecc = 'ud_node_node_eccentricity'
ud_node_clo_cntr = 'ud_node_closeness_centrality'
ud_node_far_cntr = 'ud_node_farness_centrality'
ud_node_eig_cntr = 'ud_node_eigenvector_centrality'
ud_node_btw_cntr = 'ud_node_betweenness_centrality'
ud_node_page_rank = 'ud_node_PageRank'
ud_node_hub_score = 'ud_node_hub_score'
ud_node_auth_score = 'ud_node_authorities_score'
ud_node_art_pt = 'ud_node_articulation_points'
ud_node_bridge = 'ud_node_node_of_bridge'

ud_edges_btw_cntr = 'ud_edges_betweenness_centrality'
ud_edges_bridge = 'ud_edges_bridge'


def build_relation_index(edge_node_relation):
    node_names = list(set([item[1] for item in edge_node_relation]) | set([item[2] for item in edge_node_relation]))
    node_name2id = {n: i for i, n in enumerate(node_names)}
    node_id2name = {i: n for n, i in node_name2id.items()}
    intize_relation = [(item[0], node_name2id[item[1]], node_name2id[item[2]]) for item in edge_node_relation]
    edge_index = defaultdict(list)
    for item in intize_relation:
        edge_index[(item[1], item[2])].append(item[0])
    return intize_relation, edge_index, node_name2id, node_id2name


def build_network(edge_node_relation, undirect=False):
    """
    :param edge_node_relation: list of tri-tuples: [(edgeid, start_node_id, end_node_id), ....]
    :param undirect: default False. If true, return undirect network; else direct network
    :return: direct/undirect network in snap
    """
    nids = list(set([item[1] for item in edge_node_relation]) | set([item[2] for item in edge_node_relation]))
    nodes_size = len(nids)
    DG = snap.TNGraph.New()
    for n in nids:
        DG.AddNode(int(n))  # snap requires id type as int
    for _, n_start, n_end in edge_node_relation:
        DG.AddEdge(int(n_start), int(n_end))  # snap don't store edge id
    if undirect:
        UG = snap.ConvertGraph(snap.PUNGraph, DG)
        print('network built with node size = {}, directed edges = {}, undirected edges = {}'.format(
            nodes_size, len([0 for e in DG.Edges()]), len([0 for e in UG.Edges()])))
        return UG
    else:
        print('network built with node size = {}, directed edges = {}'.format(nodes_size, len([0 for e in DG.Edges()])))
        return DG


def get_segidxs_in_ug(s, e, edge_index):
    """
    1. There can be multiple edges with the same start and end node. I.e. a direct line and a curve; two one-way edges
    2. When transforming a directed edge into undirected, the s,e could be inverse.
    """
    segidxs1 = edge_index[(s, e)]
    segidxs2 = edge_index[(e, s)]  # this look up will create empty item in edge_index
    return list(set(segidxs1 + segidxs2))


def ftr_undirected_edge(UG, edge_index=None):
    """
    compute various score for edges in undirected graph via snap
    :param UG: snap.PUNGraph
    :param edge_index: get list of edge id given (start_node, end_node), default None
    :return: nested dictionary of features
        if edge_indices is None: {(s,e): {'ftr1': float, 'ftr2': ...}, (s,e): ...}
        else: {eid1: {'ftr1': float, 'ftr2': ...}, eid2: ...}
    """
    features = defaultdict(lambda: defaultdict(float))
    # in ug for snap, the directed edge is sorted, i.e. start<end

    # Computes (approximate) Node and Edge Betweenness Centrality based on a sample of NodeFrac nodes.
    Nodesbud = snap.TIntFltH()
    Edgesbud = snap.TIntPrFltH()
    snap.GetBetweennessCentr(UG, Nodesbud, Edgesbud, 1.0)
    for edge in Edgesbud:
        s, e = edge.GetVal1(), edge.GetVal2()
        if edge_index is None:
            features[(s, e)][ud_edges_btw_cntr] += Edgesbud[edge]  # TODO: why do I write += here?
        else:
            segidxs = get_segidxs_in_ug(s, e, edge_index)
            for sidx in segidxs:
                features[sidx][ud_edges_btw_cntr] += Edgesbud[edge]

    # Returns the edge bridges in Graph in the vector EdgeV.
    # An edge is a bridge if, when removed, increases the number of connected components.
    EdgeV = snap.TIntPrV()
    snap.GetEdgeBridges(UG, EdgeV)
    for edge in EdgeV:
        s, e = edge.GetVal1(), edge.GetVal2()
        if edge_index is None:
            features[(s, e)][ud_edges_bridge] += 1  # TODO: why do I write += here?
        else:
            segidxs = get_segidxs_in_ug(s, e, edge_index)
            for sidx in segidxs:
                features[sidx][ud_edges_bridge] += 1

    return features


def ftr_undirected_node(UG):
    """
    compute various score for nodes in undirected graph via snap
    :param UG: snap.PUNGraph
    :return: nested dictionary of features: {nid1: {'ftr1': float, 'ftr2': ...}, nid2: ...}
    """
    features = defaultdict(lambda: defaultdict(float))

    start_time = datetime.datetime.now()
    print('begin ftr_undirected', costs(start_time))

    for NI in UG.Nodes():
        nid = NI.GetId()

        # Returns degree centrality of a given node NId in Graph.
        # Degree centrality of a node is defined as its degree/(N-1), where N is the number of nodes in the network.
        features[nid][ud_node_deg_cntr] = snap.GetDegreeCentr(UG, nid)

        # Returns node eccentricity,
        # the largest shortest-path distance from the node NId to any other node in the Graph.
        features[nid][ud_node_node_ecc] = snap.GetNodeEcc(UG, nid, False)

        # Returns closeness centrality of a given node NId in Graph.
        # Closeness centrality is equal to 1/farness centrality.
        features[nid][ud_node_clo_cntr] = snap.GetClosenessCentr(UG, nid, True, False)

        # Returns farness centrality of a given node NId in Graph.
        # Farness centrality of a node is the average shortest path length to all other nodes that
        # reside in the same connected component as the given node.
        features[nid][ud_node_far_cntr] = snap.GetFarnessCentr(UG, nid, True, False)
    print('got degree, node, closeness and farness centrality for nodes', costs(start_time))

    # Computes eigenvector centrality of all nodes in Graph and stores it in NIdEigenH.
    # Eigenvector Centrality of a node N is defined recursively as
    # the average of centrality values of N’s neighbors in the network.
    NIdEigenH = snap.TIntFltH()
    snap.GetEigenVectorCentr(UG, NIdEigenH)
    for nid in NIdEigenH:
        features[nid][ud_node_eig_cntr] = NIdEigenH[nid]
    print('got eigen centrality', costs(start_time))

    # Computes (approximate) Node and Edge Betweenness Centrality based on a sample of NodeFrac nodes.
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(UG, Nodes, Edges, 1.0, False)
    for nid in Nodes:
        features[nid][ud_node_btw_cntr] = Nodes[nid]
    print('got btw centrality', costs(start_time))

    # Computes the PageRank score of every node in Graph. The scores are stored in PRankH.
    PRankH = snap.TIntFltH()
    snap.GetPageRank(UG, PRankH)
    for nid in PRankH:
        features[nid][ud_node_page_rank] = PRankH[nid]
    print('got page rank', costs(start_time))

    # Computes the Hubs and Authorities score of every node in Graph. The scores are stored in NIdHubH and NIdAuthH.
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(UG, NIdHubH, NIdAuthH)
    for nid in NIdHubH:
        features[nid][ud_node_hub_score] = NIdHubH[nid]
    for nid in NIdAuthH:
        features[nid][ud_node_auth_score] = NIdAuthH[nid]
    print('got hit score', costs(start_time))

    # Returns articulation points of an undirected InGraph.
    ArtNIdV = snap.TIntV()
    snap.GetArtPoints(UG, ArtNIdV)
    for nid in ArtNIdV:
        features[nid][ud_node_art_pt] = 1
    print('got articulate point', costs(start_time))

    # Returns the edge bridges in Graph in the vector EdgeV.
    # An edge is a bridge if, when removed, increases the number of connected components.
    EdgeV = snap.TIntPrV()
    snap.GetEdgeBridges(UG, EdgeV)
    for edge in EdgeV:
        features[edge.GetVal1()][ud_node_bridge] += 1
        features[edge.GetVal2()][ud_node_bridge] += 1
    print('got bridge', costs(start_time))

    return features
