




import numpy as np
import h5py
from scipy.sparse import csr_matrix
import argparse


parser = argparse.ArgumentParser(description='from edgelist to matlab file')
parser.add_argument('--input_link_file_name', type=str, default="edgelist.train.txt",
                    help='input link file name')
parser.add_argument('--output_mat_file_name', type=str, default="result.mat",
                    help='output mat file name')
args = parser.parse_args()


input_link_file_name= args.input_link_file_name
output_mat_file_name= args.output_mat_file_name

# sparse matrix from h5py to scipy
def h5py_to_scipy(h5f):
    ir = np.array(h5f['ir'])
    jc = np.array(h5f['jc'])
    data = np.array(h5f['data'])
    #no  need to  shape = np.array(h5f['shape'])
    return csr_matrix((data, ir, jc))
def scipy_to_h5py(A, h5f, name):
    h5f.create_dataset(name + '/data', data=A.data)
    h5f.create_dataset(name + '/ir', data=A.indices)
    h5f.create_dataset(name + '/jc', data=A.indptr)

print("load links------------------")
# load links
def load_links(link_file):
    links = []
    with open(link_file) as f:
        for line in f:
            links.append([int(x) for x in line.strip().split()])
    return links
links = load_links(input_link_file_name)

print("check id------------------")
# assert id start from 0 and continuous
def check_id(links):
    ids = set()
    for link in links:
        ids.add(link[0])
        ids.add(link[1])
    ids = list(ids)
    ids.sort()
    assert ids[0] == 0
    assert ids[-1] == len(ids) - 1
    assert len(ids) == ids[-1] + 1
    return ids
ids = check_id(links)
num_nodes = len(ids)

print("build adjacency matrix------------------")
# build adjacency matrix
def build_adjacency_matrix_sparse(links, num_nodes):
    row = []
    col = []
    for link in links:
        row.append(link[0])
        col.append(link[1])
    data = [1] * len(row)
    return csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
A = build_adjacency_matrix_sparse(links, num_nodes)

print("get degree vectors------------------")
# get out degree
def get_out_degree(A):
    return np.sum(A, axis=1)
# get in degree
def get_in_degree(A):
    return np.sum(A, axis=0)
Dout = get_out_degree(A)
Din = get_in_degree(A)

# get transition matrix using out degree
print("get transition matrix------------------")
def get_transition_matrix(A, Dout):
    # A is sparse matrix
    # Dout is a vector
    num_nodes = A.shape[0]
    Dout_inv =  1 / Dout
    # filter out nan
    Dout_inv = np.nan_to_num(Dout_inv)
    # to 1d array
    Dout_inv = np.squeeze(np.asarray(Dout_inv))
    Dout_inv = csr_matrix((Dout_inv, (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes))
    return Dout_inv * A
P = get_transition_matrix(A, Dout)

# write group A and P, dataset Dout and Din into matlab .mat file
import scipy.io as sio
print("write to matlab file")
sio.savemat(output_mat_file_name, {'A': A, 'P': P, 'Dout': Dout, 'Din': Din})
