from collections import defaultdict
import itertools
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from networkx import read_weighted_edgelist, to_scipy_sparse_matrix
import os, time, random
import tempfile
import multiprocessing as mp
import itertools
import logging
import json

logger = logging.getLogger(__name__)


class ModifiedAdsorption():
    def __init__(self, _mu1=1, _mu2=1e-2, _mu3=1, _mu4=1e-4, _beta=2, **argv):
        self._mu1 = _mu1
        self._mu2 = _mu2
        self._mu3 = _mu3
        self._mu4 = _mu4
        self._tol = argv['_tol']
        self._beta = 2.
        self.max_iter = argv['max_iter']
        self.cores = argv['cores']
        self.slice_n = argv['sliec_n']

    def set_value(self, row, col, value):
        try:
            # print(row,col,value)
            self.seeds_matrix[row, col] = value
        except Exception as e:
            logger.error(e, exc_info=True)

    def get_W_Y_labels(self, graph_file, seed_file, delimiter="\t"):
        # print("="*20)
        ##获得结构图G
        G = read_weighted_edgelist(graph_file, delimiter=delimiter)
        ##获得权重矩阵W
        self.W = to_scipy_sparse_matrix(G, dtype=np.float64)
        self.nodes_G = G.nodes()
        # print(nodes_G[0:100])
        nodes_G_dict = dict(zip(self.nodes_G, range(len(self.nodes_G))))
        ###构建Y矩阵
        ##初始化标签列表
        label_index = defaultdict(list)
        ##构建seeds字典、列表
        file_lines = open(seed_file, "r").readlines()
        seed_nodes, seed_labels, seed_values = zip(*[line.split(delimiter) for line in file_lines])
        seed_values = [float(value) for value in seed_values]

        self.unique_labels = sorted(set(seed_labels))
        self.golden_labels = dict(zip(zip(seed_nodes, seed_labels), seed_values))

        # print("golden_labels:",len(self.golden_labels))
        labels_dict = dict(zip(self.unique_labels, range(len(self.unique_labels))))  # [L1, L2, L3,..., DUMMY]
        global _C
        _C = np.zeros((len(self.unique_labels) + 1, len(self.unique_labels) + 1))
        _C = lil_matrix(_C)
        global C_row_sum
        C_row_sum = _C.tocoo().sum(1)
        # print("C_row_sum::",len(C_row_sum))
        ##声明Y矩阵
        self.seeds_matrix = np.zeros([self.W.shape[0], len(self.unique_labels) + 1])

        ##初始化Y矩阵
        print("seeds_matrix.shape", self.seeds_matrix.shape)
        print("_W.shape", self.W.shape)
        # print(label_index.items())
        for index, item in enumerate(self.golden_labels):
            #    print(nodes_G_dict[item[0]], labels_dict[item[1]], golden_labels[item])
            if index < self.slice_n:
                try:
                    self.seeds_matrix[nodes_G_dict[item[0]], labels_dict[item[1]]] = self.golden_labels[item]
                except Exception as e:
                    # print(e)
                    pass
        # print(labels_dict)
        # [self.set_value(nodes_G_dict[item[0]], labels_dict[item[1]], golden_labels[item]) for idx, item in enumerate(golden_labels) if idx < self.slice_n]
        # print("="*20)
        self._Y = self.seeds_matrix
        # return self.W,self.seeds_matrix,unique_labels, golden_labels, nodes_G

    def get_W(self):
        return self.W

    def get_Y(self):
        return self.seeds_matrix

    def get_unique_labels(self):
        return self.unique_labels

    def get_golden_labels(self):
        return self.golden_labels

    def get_nodes_G(self):
        return self.nodes_G

    def cal_Pr(self, row, col, value):
        self._Pr[row, col] = value

    def cal_H(self, row, value):
        self._H[row, 0] += value

    def cal_C(self, row, value):
        self._C[row, 0] = value

    def cal_D(self, row, value):
        self._D[row, 0] = value

    def cal_Z(self, row, value):
        self._Z[row, 0] = value

    def cal_Pcont(self, row, value):
        self._Pcont[row, 0] = value

    def cal_Pinj(self, row, value):
        self._Pinj[row, 0] = value

    def cal_Pabnd(self, row, value):
        self._Pabnd[row, 0] = value

    def init_all_Matrixs(self):
        nr_nodes = self.W.shape[0]
        time_y = time.time()
        ##计算转移概率矩阵Pr
        # print("----->开始计算转移概率矩阵...")
        time_d = time.time()
        self._Pr = lil_matrix((nr_nodes, nr_nodes))
        _WCoo = self.W.tocoo()
        col_sums = {k: v for k, v in enumerate(_WCoo.sum(0).tolist()[0])}  # 权重矩阵列向量的和(对称阵)
        [self.cal_Pr(i, j, v / (col_sums[j] + 0.000001)) \
         for i, j, v in zip(_WCoo.row, _WCoo.col, _WCoo.data)]
        # print("----->转移概率矩阵done.",time.time()-time_d,"s\n")

        ##计算熵矩阵
        # print("----->开始计算熵矩阵...")
        time_d = time.time()
        self._H = lil_matrix((nr_nodes, 1))
        self._Pr = self._Pr.tocoo()
        [self.cal_H(i, -(v * np.log(v))) for i, _, v in zip(self._Pr.row, self._Pr.col, self._Pr.data)]
        # print("----->熵矩阵done.",time.time()-time_d,"s\n")

        ##计算_C
        # print("----->t计算_C")
        time_d = time.time()
        self._C = lil_matrix((nr_nodes, 1))
        log_beta = np.log(self._beta)
        [self.cal_C(i, (log_beta) / (np.log(self._beta + (1 / (np.exp(-self._H[i, 0]) + 0.00001))))) \
         for i in range(self._H.shape[0])]

        # print("----->_C done.",time.time()-time_d,"s\n")

        ##计算Dv
        # print("----->计算Dv")
        time_d = time.time()
        Y_nnz = self._Y.nonzero()
        self._D = lil_matrix((nr_nodes, 1))
        self._H = self._H.tolil()
        [self.cal_D(i, (1. - self._C[i, 0]) * np.sqrt(self._H[i, 0])) for i in Y_nnz[0]]

        # print("----->Dv done.",time.time()-time_d,"s\n")

        ##计算_Z
        # print("----->计算_z")
        time_d = time.time()
        self._Z = lil_matrix((nr_nodes, 1))
        c_v = self._C + self._D
        c_v_nnz = c_v.nonzero()
        [self.cal_Z(i, np.max([c_v[i, 0], 1.])) for i in c_v_nnz[0]]
        # print("------>Z matrix done.",time.time()-time_d,"s\n")

        ##计算三个概率参数 pcont,pinj,pabnd
        # print("----->计算三个概率参数 pcont,pinj,pabnd")
        time_d = time.time()
        self._Pcont = lil_matrix((nr_nodes, 1))
        self._Pinj = lil_matrix((nr_nodes, 1))
        self._Pabnd = lil_matrix((nr_nodes, 1))
        C_nnz = self._C.nonzero()
        [self.cal_Pcont(i, self._C[i, 0] / self._Z[i, 0]) for i in C_nnz[0]]
        [self.cal_Pinj(i, self._D[i, 0] / self._Z[i, 0]) for i in C_nnz[0]]

        self._Pabnd[:, :] = 1.
        pc_pa = self._Pcont + self._Pinj
        pc_pa_nnz = pc_pa.nonzero()
        [self.cal_Pabnd(i, 1. - pc_pa[i, 0]) for i in pc_pa_nnz[0]]
        self._Pabnd = csr_matrix(self._Pabnd)
        self._Pcont = csr_matrix(self._Pcont)
        self._Pinj = csr_matrix(self._Pinj)
        # print("计算完毕，返回函数.",time.time()-time_y, "s\n")
        # print("="*25)
        # return self._Pabnd, self._Pcont, self._Pinj

    ########################################################
    def _check_convergence(self, A, B):
        if not type(A) is csr_matrix:
            A = csr_matrix(A)
        if not type(B) is csr_matrix:
            B = csr_matrix(B)

        norm_a = (A.data ** 2).sum()
        norm_b = (B.data ** 2).sum()
        # print("norm_a:",norm_a,"norm_b",norm_b)
        diff = np.abs(norm_a - norm_b)
        if diff <= self._tol:
            return True
        else:
            print("\t\tNorm differences between Y_old and Y_hat: ", diff)
            return False

    #############
    def Cal_Mvv2(self, v):
        global C_row_sum
        first_part = self._mu1 * self._Pinj[v, 0]
        Mv_row = lil_matrix((1, len(C_row_sum)))
        second_part = 0.
        for u in self.W[v, :].nonzero()[1]:
            if u != v:
                second_part += (self._Pcont[v, 0] * self.W[v, u] + self._Pcont[u, 0] * self.W[u, v])
        for l in range(len(C_row_sum)):
            Mv_row[0, l] = first_part + (self._mu2 * second_part) + self._mu3 + self._mu4 * C_row_sum[l]
        return Mv_row

    #############
    def Cal_Dvv(self, argv):
        global _Yh
        i, j, v = argv[0], argv[1], argv[2]
        return (v * (self._Pcont[i][0] + self._Pcont[j][0])) * _Yh[j, :]

    ############
    def Cal_D(self, argv):
        global Dv, _Yh
        D_row = lil_matrix((1, _Yh.shape[1]))
        for i in range(argv[0]):
            D_row += Dv[i + argv[1]]
        return D_row

    ############
    def Cal_C_Y_multi(self, v):
        global _Yh, _C
        C_Y_multi = lil_matrix((1, _C.shape[1]))
        Yhv_temper = _Yh.toarray()
        for l in range(_C.shape[0]):
            for lv in _C[l, :].nonzero()[0]:
                C_Y_multi[0, l] += _C[l, lv] * Yhv_temper[v, lv]
        return C_Y_multi

    ################
    def Cal_Yh(self, v):
        global D, _M, _C_Y_multi
        second_part = ((self._mu1 * self._Pinj[v, 0] * self._Y[v, :]) +
                       (self._mu2 * D[v]) +
                       (self._mu3 * self._Pabnd[v, 0] * r) +
                       (self._mu4 * _C_Y_multi[v, :]))
        # print("second_part:",second_part.shape)
        _Yhv = np.multiply((1. / (_M[v, :]).toarray()), second_part)
        return _Yhv
        ########################

    def set_Mv(self, row):
        global _M, Mv
        _M[row, :] = Mv[row]

    def cal_W_coo_list(self, value):
        global W_coo_list, sum_
        W_coo_list.append((value, sum_))
        sum_ += value

    def cal_C_Y_multi(self, row):
        global _C_Y_multi, C_Y_multi
        _C_Y_multi[row, :] = C_Y_multi[row]

    def set_Yh(self, row):
        global _Yh, Yhv
        _Yh[row, :] = Yhv[row]

    #################
    def mad_iteration(self):
        # print("="*25)
        # print("\n...Calculating modified adsorption.")
        global _M, Mv
        nr_nodes = self.W.shape[0]
        nr_labels = self._Y.shape[1]
        _M = lil_matrix((nr_nodes, nr_labels))

        ##并行计算Mvv
        # print("----->计算Mvv...")
        time_d = time.time()
        with mp.Pool(self.cores) as excutor:
            Mv = excutor.map(self.Cal_Mvv2, range(nr_nodes))
        # print("----->Mv done.",time.time()-time_d,"s\n")
        time_d = time.time()
        [self.set_Mv(v) for v in range(nr_nodes)]
        # print(_M[0,:].toarray().shape)
        # print("----->Mvv done.",time.time()-time_d,"s\n")

        # print("----->Begin main loop")
        itero = 0
        global r
        r = lil_matrix((1, self._Y.shape[1]))
        r[-1, -1] = 1.
        Yh_old = lil_matrix((self._Y.shape[0], self._Y.shape[1]))
        # Main loop begins
        Pcont = self._Pcont.toarray()
        # print("--->开始while循环，进行迭代计算...")
        global _Yh, Yhv
        _Yh = lil_matrix(self._Y.copy())
        while not self._check_convergence(Yh_old, _Yh, ) and self.max_iter > itero:
            itero += 1
            print(">>>>>Iteration:%d" % itero)
            global _D
            _D = lil_matrix((nr_nodes, self._Y.shape[1]))
            # 4. Calculate Dv
            # print("\t\tCalculating Dvv...")
            time_d = time.time()
            W_coo = self.W.tocoo()
            global Dv
            with mp.Pool(self.cores) as excutor:
                Dv = excutor.map(self.Cal_Dvv, zip(W_coo.row, W_coo.col, W_coo.data))
            # print("Dv并行计算完毕...",time.time()-time_d,"s\n")

            W_coo_dict = dict(zip(*np.unique(W_coo.row, return_counts=True)))
            # print("预处理Dv，准备计算D")
            time_d = time.time()
            global sum_, W_coo_list
            sum_, W_coo_list = 0, []

            [self.cal_W_coo_list(W_coo_dict[key]) for key in W_coo_dict]
            # print("预处理完毕",time.time()-time_d,"s\n")
            # print("并行计算D")
            time_d = time.time()
            global D
            with mp.Pool(self.cores) as excutor:
                D = excutor.map(self.Cal_D, W_coo_list)
            # print("D并行计算完毕...",time.time()-time_d,"s\n")

            # print("\t\tUpdating Y...\n")
            # 5. Update Yh
            time_y = time.time()
            Yh_old = _Yh.copy()
            global _C_Y_multi, C_Y_multi
            _C_Y_multi = lil_matrix((nr_nodes, nr_labels))
            _C_Y_multi_old = _C_Y_multi
            with mp.Pool(self.cores) as excutor:
                C_Y_multi = excutor.map(self.Cal_C_Y_multi, range(nr_nodes))
            # print("--->计算_C_Y_multi....")
            [self.cal_C_Y_multi(v) for v in range(nr_nodes)]
            # print("-"*25)
            self._check_convergence(_C_Y_multi_old, _C_Y_multi)
            # print('-'*25)
            # print("times cost:",time.time()-time_y,"s")
            time_y = time.time()
            # print(_C_Y_multi.shape)

            with mp.Pool(self.cores) as excutor:
                Yhv = excutor.map(self.Cal_Yh, range(nr_nodes))
            [self.set_Yh(v) for v in range(nr_nodes)]
            # print("_YH done.",time.time()-time_y,"s\n")
            # print("="*15)
        print("收敛")
        return _Yh

    def get_result(self, _Yh, potential_file, output_file):
        result_complete = []
        mad_class_index = np.squeeze(np.asarray(_Yh[:, :_Yh.shape[1] - 1].todense().argmax(axis=1)))  # 是否考虑输出other 直接在这边改 -1
        # print([item for item in mad_class_index])
        label_results = []
        for r in mad_class_index:
            if r == len(self.unique_labels):
                label_results.append('other')
            else:
                label_results.append(self.unique_labels[r])
        # print(mad_class_index)
        for i in range(len(label_results)):
            result_complete.append((list(self.nodes_G)[i], label_results[i]))
        with open(potential_file) as file:
            potential_list = json.load(file)

        f = open(output_file, "w", encoding='utf-8')
        # print(seg_lists)
        for item in result_complete:
            if (item[0] in potential_list):
                # print(item)
                f.write(item[0] + '\t' + item[1] + '\n')
        f.close()

    def start_MAD(self, graph_file, seed_file, potential_file, output_file):
        print('Start MAD...')
        self.get_W_Y_labels(graph_file, seed_file)
        self.init_all_Matrixs()
        Y = self.mad_iteration()
        self.get_result(Y, potential_file, output_file)


# if __name__ == '__main__':
#     MAD = ModifiedAdsorption(_tol=0.1, cores=21, sliec_n=1000, max_iter=10)
#     MAD.start_MAD(graph_file, seed_file, potential_file, output_file)

#         MAD.get_W_Y_labels(graph_file, seed_file)
#         MAD.init_all_Matrixs()
#         Y = MAD.mad_iteration()
#         MAD.get_result(Y, potential_file, output_file)
