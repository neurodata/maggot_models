import numpy as np
import itertools
import networkx as nx
from tqdm import tqdm


def path_to_visits(paths, n_verts, from_order=True, out_inds=[]):
    visit_orders = {i: [] for i in range(n_verts)}
    for path in paths:
        for i, n in enumerate(path):
            if from_order:
                visit_orders[n].append(i + 1)
            else:
                visit_orders[n].append(len(path) - i)
    return visit_orders


def to_path_graph(paths):
    path_graph = nx.MultiDiGraph()

    all_nodes = list(itertools.chain.from_iterable(paths))
    all_nodes = np.unique(all_nodes)
    path_graph.add_nodes_from(all_nodes)

    for path in paths:
        path_graph.add_edges_from(nx.utils.pairwise(path))

    path_graph = collapse_multigraph(path_graph)
    return path_graph


def collapse_multigraph(multigraph):
    """REF : https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-...
        into-simple-graph-with-weighted-edges
    
    Parameters
    ----------
    multigraph : [type]
        [description]
    """
    G = nx.DiGraph()
    for u, v, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


# Maybe this class is unnecessary?
class TraverseDispatcher:
    def __init__(
        self, worker, *args, n_init=10, verbose=False, simultaneous=True, **kwargs
    ):
        self._worker = worker(*args, **kwargs)
        self.n_init = n_init
        self.simultaneous = simultaneous
        self.verbose = verbose

    def start(self, start_node):
        worker = self._worker
        hit_hist = np.zeros((worker.n_verts, worker.max_hops))
        for i in tqdm(range(self.n_init)):
            worker.start(start_node)
            traversal = worker.traversal_
            for level, nodes in enumerate(traversal):
                hit_hist[nodes, level] += 1
        self.hit_hist_ = hit_hist
        return hit_hist

    def multistart(self, start_nodes):
        if self.simultaneous:
            hop_hist = self.start(start_nodes)
        else:
            n_verts = len(self._worker.transition_probs)
            hop_hist = np.zeros((n_verts, self._worker.max_hops))
            for s in start_nodes:
                hop_hist += self.start(s)
        return hop_hist


class BaseTraverse:
    def __init__(
        self,
        transition_probs,
        stop_nodes=[],
        max_hops=10,
        hit_hist=None,
        record_traversal=True,
        allow_loops=True,
    ):
        """
        
        Parameters
        ----------
        transition_probs : np.ndarray or dict or list
            if np.ndarray, then the ints indexing the nodes are assumed to correspond to
            the indices of this array.
            if dict or list, then the nodes should be keys in the dict. the elements in
            the dict should represent what transitions are possible
        stop_nodes : list, optional
            list of nodes that should stop the traversal
        max_hops : int, optional
            [description], by default 10
        hit_hist : reference, optional
            [description], by default None
        record_traversal : bool, optional
            [description], by default True
        """
        self.transition_probs = transition_probs
        self.hit_hist = hit_hist
        self.record_traversal = record_traversal
        self.max_hops = max_hops
        self.stop_nodes = stop_nodes
        self.allow_loops = allow_loops
        self.n_verts = len(transition_probs)
        if record_traversal:
            self.traversal_ = None
        if not allow_loops:
            self._visited = None

    def _check_max_hops(self):
        return not self._hop >= self.max_hops  # do not continue if greater than

    def _check_stop_nodes(self):
        return self._active not in self.stop_nodes

    def _check_visited(self):
        if not self.allow_loops:
            return self._active not in self._visited
        else:
            return True

    def _check_stop_conditions(self):
        check_items = [self._check_max_hops(), self._check_visited()]
        return all(check_items)

    def _reset(self):
        self._hop = 0
        self._active = None
        self._visited = np.array([])
        if self.record_traversal:
            self.traversal_ = []

    def _update_state(self, nxt):
        """Takes the next step in the walk and updates the state of the object
        
        Parameters
        ----------
        nxt : node
        
        Returns
        -------
        bool
            True if nxt was not None, meaning, advance the walk
            False if the walk stopped because an advance could not be made 
        """
        if nxt is not None:
            self._active = nxt
            self._hop += 1
            self.traversal_.append(nxt)
            if not self.allow_loops:
                # May be slow? working with set() seemed overly complicated here,
                # looked like I had to convert set to array for every comparison
                self._visited = np.union1d(self._visited, nxt)
            return True
        else:
            return False

    def _step(self):
        if self._check_stop_conditions():
            self._update_state(self._active)
            if self._check_stop_nodes():
                nxt = self._choose_next()
                self._active = nxt
            else:
                nxt = None
            return nxt

    def start(self, start_node):
        self._reset()
        self._active = start_node
        nxt = start_node
        while nxt is not None:
            nxt = self._step()
