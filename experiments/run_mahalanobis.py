import numpy as np
import sys
import time
import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import tensor_decomposition.utils.arg_defs as arg_defs
import csv
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.mahalanobis import CP_AMDM_Optimizer

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def CP_Mahalanobis(tenpy, A, T, O, num_iter, csv_file=None, Regu=None, args=None, res_calc_freq=1):

    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    if Regu is None:
        Regu = 0
    
    iters = 0
    count = 0
    
    
    time_all = 0.
    normT = tenpy.vecnorm(T)
    method = 'AMDM'


    optimizer = CP_AMDM_Optimizer(tenpy, T, A, args)

    for k in range(num_iter):
        if k % res_calc_freq == 0 or k==num_iter-1 :
            if args.sp and O is not None:
                res = get_residual_sp(tenpy,O,T,A)
            else:
                res = get_residual(tenpy,T,A)

            fitness = 1-res/normT
            if args.calc_cond and R < 15 and tenpy.name() == 'numpy':
                cond = compute_condition_number(tenpy, A)
            else:
                cond = 0
            if tenpy.is_master_proc():
                print("[",k,"] Residual is", res, "fitness is: ", fitness)
                # write to csv file
                if csv_file is not None:
                    csv_writer.writerow([method,k, time_all, res, fitness,cond])
                    csv_file.flush()
        t0 = time.time()

        A = optimizer.step(Regu)
        
        if args.reduce_val:
            if k > 0 and k % args.reduce_val_freq==0 :
                optimizer.reduce_vals()

        

        t1 = time.time()

        tenpy.printf("[",k,"] Sweep took", t1-t0,"seconds")

        time_all += t1-t0

        if res < args.tol:
            tenpy.printf('Method converged due to residual tolerance in',k,'iterations')
            break

        if fitness > args.fit:
            tenpy.printf('Method converged due to fitness tolerance in',k,'iterations')
            break

    tenpy.printf(method+" method took",time_all,"seconds overall")
    

    if args.save_tensor:
        folderpath = join(results_dir, arg_defs.get_file_prefix(args))
        save_decomposition_results(T,A,tenpy,folderpath)

    return A


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_amdm_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up CSV logging
    csv_path = join(results_dir, 'Mahalanobis-'+args.tensor+'-order-'+str(args.order)+'-s-'+str(args.s)+'-R-'
        +str(args.R)+'-R_app-'+str(args.R_app)+'-thresh-'+str(args.thresh)+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    s = args.s
    order = args.order
    R = args.R
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor = args.tensor
    tlib = args.tlib
    thresh = args.thresh
    if args.R_app is None:
        R_app = args.R
    else:
        R_app = args.R_app
    if args.num_vals is None:
        args.num_vals = args.R
    

    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin();

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'method','iterations', 'time', 'residual', 'fitness','cond_num'
            ])

    tenpy.seed(args.seed)
    

    if args.load_tensor != '':
        T = tenpy.load_tensor_from_file(args.load_tensor+'tensor.npy')
        O = None
    elif tensor == "random":
        tenpy.printf("Testing random tensor")
        [T,O] = synthetic_tensors.rand(tenpy,order,s,R,sp_frac,args.seed)

    elif tensor == "MGH":
        T = tenpy.load_tensor_from_file("MGH-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif tensor == "SLEEP":
        T = tenpy.load_tensor_from_file("SLEEP-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif tensor == "random_col":
        [T,O] = synthetic_tensors.collinearity_tensor(tenpy, s, order, R, args.col, args.seed)
        O = None
    elif tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)
        O = None
    elif tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
        O = None

    tenpy.printf("The shape of the input tensor is: ", T.shape)

    Regu = args.regularization

    A = []
    
    tenpy.seed(args.seed)
    if args.load_tensor != '':
        for i in range(T.ndim):
            A.append(tenpy.load_tensor_from_file(args.load_tensor+'mat'+str(i)+'.npy'))
    elif args.hosvd != 0:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((args.hosvd_core_dim[i],R_app)))
        elif args.decomposition == "Tucker":
            from Tucker.common_kernels import hosvd
            A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
    else:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], R_app)))
        else:
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], args,hosvd_core_dim[i])))
    CP_Mahalanobis(tenpy, A, T, O, num_iter, csv_file, Regu, args, args.res_calc_freq)
