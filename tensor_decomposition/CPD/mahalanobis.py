import numpy as np
import numpy.linalg as la
from .common_kernels import solve_sys, compute_lin_sysN, normalise
try:
    import Queue as queue
except ImportError:
    import queue

class CP_AMDM_Optimizer():
    """
    AMDM method for computing CP decomposition. The algorithm uses the general version of AMDM
    updates. The 
    
    Refer to the paper for details on how the optimization is carried out
    """
    def __init__(self,tenpy,T,A,args):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.R = A[0].shape[1]
        self.sp = args.sp
        self.U = []
        self.sing = []
        self.VT = []
        self.thresh = args.thresh
        self.num_vals = args.num_vals
        self.nrm = None 
        self.update_svd()

    def reduce_vals(self):
        if self.num_vals > 0:
            self.num_vals -= 1

    def absorb_norm(self):
        self.A[0] = self.tenpy.einsum('r,ir->ir',self.nrm,self.A[0])

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = "".join([chr(ord('a') + j) for j in range(nd)]) + ci
        str2 = (chr(ord('a') + ii)) + "R"
        str3 = "".join([chr(ord('a') + j) for j in range(nd) if j != ii]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def update_svd(self, i=None):
        if i is None:
            self.A,self.nrm = normalise(self.tenpy, self.A)
            for i in range(len(self.A)):
                U,s,VT = self.tenpy.svd(self.A[i])
                self.U.append(U)
                self.sing.append(s)
                self.VT.append(VT)
        else:
            self.A,self.nrm = normalise(self.tenpy, self.A, i)
            U,s,VT = self.tenpy.svd(self.A[i])
            self.U[i] = U
            self.sing[i] = s
            self.VT[i] = VT

    def compute_rhs(self,i):
        if self.thresh is not None:
            sing = np.where((self.sing[i][0] / self.sing[i]) < self.thresh, 
                1 / self.sing[i], self.sing[i])
        else:
            sing = np.concatenate((1 / self.sing[i][:self.num_vals], self.sing[i][self.num_vals:]))
        return self.tenpy.einsum('ir,r,rj->ij',self.U[i],sing,self.VT[i])

    def compute_rhs_lst(self,i):
        lst = []
        for j in range(len(self.A)):
            if j != i:
                if self.thresh is not None:
                    sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                        1 / self.sing[j], self.sing[j])
                else:
                    sing = np.concatenate((1 / self.sing[j][:self.num_vals],
                     self.sing[j][self.num_vals:]))
                lst.append(self.tenpy.einsum('ir,r,rj->ij',self.U[j],sing,self.VT[j]))
            else:
                lst.append(self.tenpy.zeros(self.A[i].shape))
        return lst

    def step(self,Regu):
        if not self.sp:
            q = queue.Queue()
        if self.sp:
            s = []
            for i in range(len(self.A)):
                lst = self.compute_rhs_lst(i)
                self.tenpy.MTTKRP(self.T,lst,i)
                if self.thresh is None and self.num_vals == self.A[i].shape[1]:
                    self.A[i] = lst[i]
                else:
                    self.A[i] = self._sp_solve(i,Regu,lst[i])
                self.update_svd(i)
        else:
            for i in range(len(self.A)):
                q.put(i)
            s = [(list(range(len(self.A))),self.T)]

            while not q.empty():
                i = q.get()
                while i not in s[-1][0]:
                    s.pop()
                    assert(len(s) >= 1)
                while len(s[-1][0]) != 1:
                    M = s[-1][1]
                    idx = s[-1][0].index(i)
                    ii = len(s[-1][0])-1
                    if idx == len(s[-1][0])-1:
                        ii = len(s[-1][0])-2
                    rh = self.compute_rhs(ii)
                    einstr = self._einstr_builder(M,s,ii)
                    N = self.tenpy.einsum(einstr,M,rh)

                    ss = s[-1][0][:]
                    ss.remove(ii)
                    s.append((ss,N))
                if self.thresh is None and self.num_vals == self.A[i].shape[1]:
                    self.A[i] = s[-1][1].copy()
                else:
                    self.A[i] = self._solve(i,Regu,s)
                self.update_svd(i)
        self.absorb_norm()
        return self.A

    def _sp_solve(self, i, Regu, g):
        lst = []
        for j in range(len(self.VT)):
            if self.thresh is not None:
                sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                1.0, self.sing[j])
            else:
                sing = np.concatenate((np.ones(self.num_vals), 
                    self.sing[j][self.num_vals:]))
            lst.append(self.tenpy.einsum('r,rj->rj',sing,self.VT[j]))
        return solve_sys(self.tenpy,
                        compute_lin_sysN(self.tenpy, lst, i, Regu),
                        g)

    def _solve(self, i, Regu, s):
        lst = []
        for j in range(len(self.VT)):
            if self.thresh is not None:
                sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                1.0, self.sing[j])
            else:
                sing = np.concatenate((np.ones(self.num_vals),
                 self.sing[j][self.num_vals:]))
            lst.append(self.tenpy.einsum('r,rj->rj',sing,self.VT[j]))
        return solve_sys(self.tenpy,
                        compute_lin_sysN(self.tenpy, lst, i, Regu),
                        s[-1][1])

