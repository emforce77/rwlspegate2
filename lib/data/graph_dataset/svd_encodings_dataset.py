
import numpy as np
import torch
from tqdm import trange
import numba as nb

from .graph_dataset import GraphDataset

class SVDEncodingsDatasetBase:
    def __init__(self,
                 svd_encodings_key = 'svd_encodings',
                 calculated_dim    = 8,
                 output_dim        = 8,
                 random_neg_splits = ['training'],
                 **kwargs):
        if output_dim > calculated_dim:
            raise ValueError('SVD: output_dim > calculated_dim')
        super().__init__(**kwargs)   
        self.svd_encodings_key = svd_encodings_key
        self.calculated_dim    = calculated_dim    
        self.output_dim        = output_dim        
        self.random_neg_splits = random_neg_splits
    
    def calculate_encodings(self, item):
        raise NotImplementedError('SVDEncodingsDatasetBase.calculate_encodings()')
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        token  = self.record_tokens[index]
        
        try:
            encodings = self._svd_encodings[token]
        except AttributeError:
            encodings = self.calculate_encodings(item)
            self._svd_encodings = {token:encodings}
        except KeyError:
            encodings = self.calculate_encodings(item)
            self._svd_encodings[token] = encodings
        
        if self.output_dim < self.calculated_dim:
            encodings = encodings[:,:self.output_dim,:]
        
        if self.split in self.random_neg_splits:
            rn_factors = np.random.randint(0, high=2, size=(encodings.shape[1],1))*2-1 #size=(encodings.shape[0],1,1)
            encodings = encodings * rn_factors.astype(encodings.dtype)
        
        item[self.svd_encodings_key] = encodings.reshape(encodings.shape[0],-1)
        return item
    
    def calculate_all_svd_encodings(self,verbose=1):
        self._svd_encodings = {}
        if verbose:
            print(f'Calculating all {self.split} SVD encodings...', flush=True)
            for index in trange(super().__len__()):
                item = super().__getitem__(index)
                token  = self.record_tokens[index]
                self._svd_encodings[token] = self.calculate_encodings(item)
        else:
            for index in range(super().__len__()):
                item = super().__getitem__(index)
                token = self.record_tokens[index]
                self._svd_encodings[token] = self.calculate_encodings(item)
    
    def cache_load_and_save(self, base_path, op, verbose):
        super().cache_load_and_save(base_path, op, verbose)
        svd_encodings_path = base_path/'svd_encodings.pt'
        
        if op == 'load':
            self._svd_encodings = torch.load(str(svd_encodings_path))
        elif op == 'save':
            if verbose: print(f'{self.split} SVD encodings cache does not exist! Cacheing...', flush=True)
            self.calculate_all_svd_encodings(verbose=verbose)
            torch.save(self._svd_encodings, str(svd_encodings_path))
            if verbose: print(f'Saved {self.split} SVD encodings cache to disk.', flush=True)
        else:
            raise ValueError(f'Unknown operation: {op}')

'''
@nb.njit
def calculate_svd_encodings(edges, num_nodes, calculated_dim):
    adj = np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i,0]),nb.int64(edges[i,1])] = 1
    
    for i in range(num_nodes):
        adj[i,i] = 1
    u, s, vh = np.linalg.svd(adj)
    
    if calculated_dim < num_nodes:
        s = s[:calculated_dim]
        u = u[:,:calculated_dim]
        vh = vh[:calculated_dim,:]
        
        encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)
    elif calculated_dim > num_nodes:
        z = np.zeros((num_nodes,calculated_dim-num_nodes,2),dtype=np.float32)
        encodings = np.concatenate((np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1), z), axis=1)
    else:
        encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)
    return encodings
'''

import networkx as nx
import numpy as np
import numba as nb

from networkx.algorithms.community.centrality import girvan_newman
import itertools

import torch

'''

def rwpe_encodings(edges, num_nodes, calculated_dim):   # RWPE 만들어주는 것 == edges
    pos_enc_dim = calculated_dim

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i, 0]), nb.int64(edges[i, 1])] = 1

    for i in range(num_nodes):
        adj[i, i] = 1

    A = torch.from_numpy(adj)

    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    identity = torch.eye(A.size()[0])

    diag = torch.diag(A)

    D = diag * in_degree + diag * out_degree - torch.diagflat(torch.diagonal(A))

    Dinv = torch.linalg.inv(D)

    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [M.diagonal().float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(M_power.diagonal().float())
    encodings = torch.stack(PE, dim=-1)

    return encodings

    ######################################################################

    # hierarchy
def hiera_encoding(edges,num_nodes,calculated_dim,kus=5) :


    stacking = []
    

    G = nx.Graph()
    G.add_edges_from( [(kk1,kk2) for kk1, kk2 in zip(nb.int64(edges[:, 0]), nb.int64(edges[:, 1]))] )
     # k말고 다른걸 쓸수는 없을까.
    comp = girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= kus, comp)

    cc = set()

    for communities in limited:
        cc.update(tuple(tuple(sorted(c)) for c in communities))

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edges.shape[0]):
      adj[nb.int64(edges[i, 0]), nb.int64(edges[i, 1])] = 1

    for i in range(num_nodes):
      adj[i, i] = 1

    kcc = list(cc)

    for ksks in range(len(kcc)) :
        for i in range(len(kcc[ksks])) :
          newadj = np.zeros((len(kcc[ksks]), len(kcc[ksks])), dtype=np.float32)


        for x,j in enumerate(kcc[ksks]) :
            for y,k in enumerate(kcc[ksks]) :
                newadj[x,y] = adj[j,k]


        A = torch.from_numpy(newadj)

        out_degree = torch.sum(A, dim=0)
        in_degree = torch.sum(A, dim=1)

        identity = torch.eye(A.size()[0])

        diag = torch.diag(A)

        D = diag * in_degree + diag * out_degree - torch.diagflat(torch.diagonal(A))

        Dinv = torch.linalg.inv(D)

        RW = A * Dinv
        M = RW

        # Iterate
        nb_pos_enc = calculated_dim
        PE = [M.diagonal().float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(M_power.diagonal().float())

        
        PEs = torch.stack(PE, dim=-1)
        PEsss = torch.mean(PEs,dim=0)

        PEssss = np.zeros((num_nodes,calculated_dim))
        for x,i in enumerate(kcc[ksks]):
          PEssss[i,:] = PEsss

        PEssss = torch.tensor(PEssss)

        stacking.append(PEssss)


    hieras = torch.stack(stacking,dim=0)


    return hieras

def encoding_encoding(edges, num_nodes, calculated_dim) :

  final_rwpe = rwpe_encodings(edges, num_nodes, calculated_dim)

  final_hiera = hiera_encoding(edges,num_nodes,calculated_dim,kus=5).mean(dim=0).repeat(2,1,1).numpy().reshape(num_nodes,calculated_dim,2)
  
  return final_rwpe.repeat(2,1,1).numpy().reshape(num_nodes,calculated_dim,2)+ final_hiera
'''
def rwpe_encodings(edges, num_nodes, calculated_dim):   # RWPE 만들어주는 것 == edges
    pos_enc_dim = (3*calculated_dim//4)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i, 0]), nb.int64(edges[i, 1])] = 1

    for i in range(num_nodes):
        adj[i, i] = 1

    A = torch.from_numpy(adj)

    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    identity = torch.eye(A.size()[0])

    diag = torch.diagonal(A).float()

    D = diag * in_degree + diag * out_degree - torch.diagonal(A)

    Dinv = torch.linalg.inv(torch.diag(D))

    RW = torch.matmul(A, Dinv)
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [M]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = torch.matmul(M_power, M)
        PE.append(M_power)
    encodings = torch.stack(PE, axis=-1)

    hiera =  hiera_encoding(A,num_nodes)

    encodings = torch.cat((encodings,hiera),dim=-1).float()

    return encodings.numpy()

    ######################################################################

    # hierarchy
def hiera_encoding(edges,num_nodes,kus=5) :

    

    G = nx.Graph()

    if edges.shape[0] < 2:
        edges = np.array([[0],[0]]).T

    G.add_edges_from( [(kk1,kk2) for kk1, kk2 in zip(nb.int64(edges[:, 0]), nb.int64(edges[:, 1]))] )
     # k말고 다른걸 쓸수는 없을까.
    comp = girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= kus, comp)

    cc = set()

    for communities in limited:
        cc.update(tuple(tuple(sorted(c)) for c in communities))

    kcc = list(cc)


    if len(kcc) < 8:

      if len(kcc) == 0 :
        kcc.append((0,))

      for i in range(8-len(kcc)):
        kcc.append(kcc[-1])
    kcc = sorted(kcc,key = lambda x : -len(x))
    kcc = kcc[:8]

    encodings = []

    for ksks in range(len(kcc)) : # girvan set

        newadj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for x in kcc[ksks] :
            for y in kcc[ksks] :
                newadj[x,y] = edges[x,y]

        encodings.append(torch.tensor(newadj))


    hieras = torch.stack(encodings,dim=-1)



    return hieras


class SVDEncodingsGraphDataset(SVDEncodingsDatasetBase, GraphDataset):
    def calculate_encodings(self, item):
        num_nodes = int(item[self.num_nodes_key])
        edges = item[self.edges_key]
        encodings = rwpe_encodings(edges, num_nodes, self.calculated_dim)
        return encodings


