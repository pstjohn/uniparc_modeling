import itertools
import gzip
import re
import os 

import numpy as np
import pandas as pd
import networkx as nx

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_group(group):
    out =  {}
    out['type'] = group[0]
    out['is_a'] = []
    out['relationship'] = []
    
    for line in group[1:]:
        key, val = line.split(': ', 1)
        
        # Strip out GO names
        if '!' in val:
            val = re.sub('\ !\ .*$', '', val)
        
        if key == 'relationship':
            val = val.split(' ')
            
        # Convert to lists of GO names
        if key not in out:
            out[key] = val
        else:
            try:
                out[key] += [val]
            except TypeError:
                out[key] = [out[key], val]

    return out

add_rels = False


class Ontology(object):
    def __init__(self, obo_file=None, with_relationships=False,
                 term_count_file=None, threshold=500):
        """ Class to parse an .obo.gz file containing a gene ontology description,
        and build a networkx graph. Allows for propogating scores and annotations
        to descendent nodes.
        
        obo_file: a gzipped obo file that corresponds to the ontology used in the
        training data. Here, QuickGO annotations used GO releases/2020-06-24
        
        with_relationships (bool): whether to include GO relationships as explicit
        links in the dependency graph
        
        """
        
        if obo_file is None:
            obo_file = os.path.join(dir_path, 'go-basic.obo.gz')
            
        if term_count_file is None:
            term_count_file = os.path.join(dir_path, 'term_counts.csv.gz')            
            
        self.G = self.create_graph(obo_file, with_relationships)
        
        term_counts = pd.read_csv(term_count_file, index_col=0)['0']
        self.to_include = set(term_counts[term_counts >= threshold].index)
        self.term_index = {}
        
        for i, (node, data) in enumerate(filter(lambda x: x[0] in self.to_include,
                                                self.G.nodes.items())):
            data['index'] = i
            self.term_index[i] = node
        
        self.total_nodes = i + 1
    
    def create_graph(self, obo_file, with_relationships):

        G = nx.DiGraph()
        
        with gzip.open(obo_file, mode='rt') as f:


            groups = ([l.strip() for l in g] for k, g in
                      itertools.groupby(f, lambda line: line == '\n'))

            for group in groups:
                data = parse_group(group)

                if ('is_obsolete' in data) or (data['type'] != '[Term]'):
                    continue

                G.add_node(data['id'], name=data.get('name'), namespace=data.get('namespace'))

                for target in data['is_a']:
                    G.add_edge(target, data['id'], type='is_a')

                if with_relationships:
                    for type_, target in data['relationship']:
                        G.add_edge(target, data['id'], type=type_)
        
        # Initialize the 
        nx.set_node_attributes(G, None, 'index')

        return G
    
    
    def terms_to_indices(self, terms):
        """ Return a sorted list of indices for the given terms, omitting
        those less common than the threshold """
        return sorted([self.G.nodes[term]['index'] for term in terms if 
                       self.G.nodes[term]['index'] is not None])

    
    def get_ancestors(self, terms):
        """ Includes the query terms themselves """
        return set.union(set(terms), *(nx.ancestors(self.G, term) for term in terms))
    

    def get_descendants(self, term):
        """ Includes the query term """
        return set.union(set([term]), nx.descendants(self.G, term))
    
    
    def get_canoconical_terms(self, terms):
        subgraph = self.G.subgraph(self.get_ancestor_list(terms))
        return {node for node, degree in subgraph.out_degree if degree == 0}
    
    
    def termlist_to_array(self, terms, dtype=bool):
        """ Propogate labels to ancestor nodes """
        arr = np.zeros(self.total_nodes, dtype=dtype)
        arr[np.asarray(self.terms_to_indices(terms))] = 1
        return arr

    
    def get_descendent_array(self):
        return [self.terms_to_indices(self.get_descendants(node))
                for node, index in self.G.nodes(data='index') if index]

    
    def get_head_node_indices(self):
        return self.terms_to_indices([node for node, degree 
                                      in self.G.in_degree if degree == 0])
    
    
# notes, use nx.shortest_path_length(G, root) to find depth? score accuracy by tree depth?