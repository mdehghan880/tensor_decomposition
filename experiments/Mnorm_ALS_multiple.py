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
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.mahalanobis import CP_AMDM_Optimizer
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer
from run_als import CP_ALS
from run_mahalanobis import CP_Mahalanobis

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_amdm_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up CSV logging
    

    s = args.s
    order = args.order
    R = args.R
    if args.R_app is None:
        R_app = args.R
    else:
        R_app = args.R_app

    if args.num_vals is None:
        args.num_vals = R_app
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor = args.tensor
    tlib = args.tlib
    

    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin();

    if args.load_tensor != '':
        T = tenpy.load_tensor_from_file(args.load_tensor+'tensor.npy')
        O = None
    elif tensor == "random":
        tenpy.printf("Testing random tensor")
        [T,O] = synthetic_tensors.rand(tenpy,order,s,R,sp_frac,np.random.randint(100))

    elif tensor == "MGH":
        T = tenpy.load_tensor_from_file("MGH-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif tensor == "SLEEP":
        T = tenpy.load_tensor_from_file("SLEEP-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif tensor == "random_col":
        [T,O] = synthetic_tensors.collinearity_tensor(tenpy, s, order, R, args.col, np.random.randint(100))
    elif tensor == "scf":
        T = np.load('scf_tensor.npy')
        O = None
    elif tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
        O = None

    tenpy.printf("The shape of the input tensor is: ", T.shape)

    num_tensors = 5
    
    for it in range(num_tensors):
        Regu = args.regularization

        A = []
        
        if args.load_tensor != '':
            for i in range(T.ndim):
                A.append(tenpy.load_tensor_from_file(args.load_tensor+'mat'+str(i)+'.npy'))
        elif args.hosvd != 0:
            if args.decomposition == "CP":
                for i in range(T.ndim):
                    A.append(tenpy.random((args.hosvd_core_dim[i],R_app)))
            elif args.decomposition == "Tucker":
                from tensor_decomposition.Tucker.common_kernels import hosvd
                A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
        else:
            if args.decomposition == "CP":
                for i in range(T.ndim):
                    A.append(tenpy.random((T.shape[i], R_app)))
            else:
                for i in range(T.ndim):
                    A.append(tenpy.random((T.shape[i], args,hosvd_core_dim[i])))
        

        B = A[:]
        C = A[:]
        D = A[:]
        E = A[:]

        csv_path = join(results_dir, 'Mahalanobis-vals-'+str(args.num_vals)+args.tensor+'order'+str(args.order)+str(args.s)+'-R-'
            +str(args.R)+'-R_app-'+str(args.R_app)+'iter'+str(it)+'.csv')
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')#, newline='')
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        args.reduce_val = 0
        if tenpy.is_master_proc():
            # print the arguments
            for arg in vars(args) :
                print( arg+':', getattr(args, arg))
            # initialize the csv file
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness','cond_num'
                ])

         
        CP_Mahalanobis(tenpy, D, T, O, num_iter, csv_file, Regu, args)

        csv_path = join(results_dir, 'Mahalanobis-'+args.tensor+str(args.s)+'-R-'
            +str(args.R)+'-R_app-'+str(args.R_app)+'iter'+str(it)+'.csv')
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')#, newline='')
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)


        args.reduce_val = 0
        args.num_vals = R_app
        if tenpy.is_master_proc():
            # print the arguments
            for arg in vars(args) :
                print( arg+':', getattr(args, arg))
            # initialize the csv file
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness','cond_num'
                ])

        
        CP_Mahalanobis(tenpy, A, T, O, num_iter, csv_file, Regu, args)
        
        csv_path = join(results_dir, 'ALS-'+args.tensor+'order'+str(args.order)+str(args.s)+'-R-'
            +str(args.R)+'-R_app-'+str(args.R_app)+'iter'+str(it)+'.csv')
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')#, newline='')
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)


        Regu = 1e-07
        args.method = 'DT'

        if tenpy.is_master_proc():
            # print the arguments
            for arg in vars(args) :
                print( arg+':', getattr(args, arg))
            # initialize the csv file
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness','cond_num'
                ])


        
        CP_ALS(tenpy,
           B,
           T,
           O,
           num_iter,
           csv_file,
           Regu,
           method='DT',
           args=args,
           res_calc_freq=1,
           tol=1e-05)

        csv_path = join(results_dir, 'Hybrid-'+args.tensor+str(args.s)+'-R-'
            +str(args.R)+'-R_app-'+str(args.R_app)+'iter'+str(it)+'.csv')
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')#, newline='')
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)


        args.reduce_val = 1
        args.num_vals = R_app

        if tenpy.is_master_proc():
            # print the arguments
            for arg in vars(args) :
                print( arg+':', getattr(args, arg))
            # initialize the csv file
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness','cond_num'
                ])

        CP_Mahalanobis(tenpy, C, T, O, num_iter, csv_file, Regu, args)

    
