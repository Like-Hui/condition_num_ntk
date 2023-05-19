# condition_num_ntk

This repo is the code for the experiments given in "ReLU soothes the NTK condition number and accelerates optimization for wide neural networks." arXiv preprint arXiv:2305.08813 (2023)."

min_anlge.py gives the minimal embedding anlge (model depth l=0) of all possible pairs of samples and the minimal gradient angle (model depth l>0) of all possible pairs of samples. This code can get the results given in Figure 2(a) of the paper.

condition_num.py calculate the condition number of the NTK with different depth (l), and when l=0, it gives the condition number of the gram matrix. This code can get the results given in Figure 2(b) of the paper.

nn_train.py trains a fully connected network with NTK initialization and gives the results of Figure (3) in the paper.
