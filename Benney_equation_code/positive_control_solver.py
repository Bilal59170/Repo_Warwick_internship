##Construction of a solver for the system.


import numpy as np 
import qpsolvers 
import time 

print(qpsolvers.available_solvers)


def Mat_QP_problem(M_1, M_2, verbose=False):
    '''
    Function to build the QP optimisation problem.
    M_1: A for the positive Control problem
    M_2: delta*B for the positive Control problem
    '''
    (N, k) = M_2.shape


    ###Construction of the QP Opti problem
    Mat_Hurwitz = np.block([M_1]+N*[M_2])
    Mat_Metzler = np.zeros((N**2, N*(1+k))) #representing a_{ij}d_j + b_iz_j >0 with (i,j) in [|1,N|]

    print("shape Mat_Metzler:", Mat_Metzler.shape)
    print("\n shape Mat_Hurwitz:", Mat_Hurwitz.shape)
    def i_part(s): return int(1+np.floor((s-1)/N))
    def j_part(s): return (s-1)%N + 1

    print("Beginning of the construction of Mat_Hurwitz")

    #Cf construction of the problem. I just make Mat_Hurwitz[s-1, r-1] because the indexes here are [0, n-1] and not [1, n].
    t_i = time.time()
    for s in range(1, Mat_Metzler.shape[0]+1): # s = (i-1)N + j
        i, j = i_part(s), j_part(s)
        if i!=j:
            Mat_Metzler[s-1, j-1] = M_1[i-1, j-1] #r = j: corresponds to a_{ij}d_j
            for r in range(N + (j-1)*k + 1, N + j*k+1):#index of z_j, corresponds to b_iz_j
                Mat_Metzler[s-1, r-1] = M_2[i-1, (r-(N+(j-1)*k)) - 1 ]

    t_tot = time.time()-t_i
    print("total computation time:", t_tot)


    G_QP = np.block([[-Mat_Metzler, np.zeros((Mat_Metzler.shape[0], Mat_Hurwitz.shape[1]))],
                        [np.zeros((Mat_Hurwitz.shape[0], Mat_Metzler.shape[1])), Mat_Hurwitz]])
    print("Shape M_tot:", G_QP.shape)

    A_QP = np.block([np.identity(N*(1+k)), -np.identity(N*(1+k))])

    size_v = 2*N*(1+k)
    # P_QP = np.zeros((size_v, size_v))
    # P_QP[N: N*(k+1), N: N*(k+1)] = np.identity(N*k)
    P_QP = np.zeros((size_v, size_v))


    if verbose: #print of the Opti problem matrices. Have appropriate form for small N and k. 
        print("Matrices: ")
        print("Quad cost matrix P:\n", P_QP)
        print("M_tot:\n", G_QP)
        print("eq constraint matrix G:\n", A_QP)


    return P_QP, G_QP, A_QP

def check_QP_constraints(_G_QP, _A_QP, _v_opti, _K):
    print("\n**** Checking the QP opti problem constraints ****")

    print("Shape of gain matrix K with positive Ctrl: ", _K.shape)
    #Checking that v_opti >0
    print("min of v_opti, supposed to be positive: ",np.min(_v_opti))
    print("max of v_opti: ",np.max(_v_opti))
    #Checking of the condition (G_QP)x < h <=> a_{ij}d_j + b_iz_j >0 with (i,j) in [|1,N_x|] & Ad+delta*B\sum_1^n z_i <0
    print("max of Gv, supposed to be negative:", np.max(_G_QP@_v_opti))
    print("min of Gv:", np.min(_G_QP@_v_opti))

    #Checking of (A_QP)v = 0 i.e that we get the 2 times the same vector 
    print("max of diff: ", np.max(np.absolute(_A_QP@_v_opti)) )  #/v_opti to have relative difference.
    print("min of the diff: ", np.min(np.absolute(_A_QP@_v_opti)))
    #v_opti is little so little diff isn't significant


def is_Metzler_and_Hurwitz(M):
    ''' M = A+B_delta@K'''
    print("\n**** CHeck if the input matrix is Metzler and Hurwitz ****")
    N = M.shape[0]
    m = M[0,1]

    Mat = np.copy(M)
    for i in range(N):
        Mat[i, i] = np.inf
    Mat = np.sort(Mat.flatten())
    nb_neg_terms = np.shape(np.where(Mat<0)[0])[0]

    print("- Metzler Check:") 
    print("min should be >0. 3 lowest values: ", Mat[:3])
    print("Proportion of <0 non diagonal terms (supposed to be 0%): ",
           nb_neg_terms/(N**2-N)*100, "% = ", nb_neg_terms , "terms") 

    eigen_values = np.linalg.eigvals(M)
    print("- Hurwitz check:")
    print("max(Re(eigenvalues)) should be negative. 3 biggest eigenvalues:", np.sort(eigen_values.real)[-3:])
    print("min(Re(eigenvalues)):", min(eigen_values.real))

#Tests
if False:
    Test_mat = np.identity(3)
    Test_mat[0,1], Test_mat[2, 1], Test_mat[1, 2] = -1, -5, -3
    is_Metzler_and_Hurwitz(Test_mat)