##Construction of a solver for the system.


import numpy as np 
import qpsolvers 
import time 
from scipy.sparse import csc_matrix
from header import *



######## Positive Linear Control ############
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
    P_QP = np.zeros((size_v, size_v))
    if True:
        P_QP[N: N*(k+1), N: N*(k+1)] = np.identity(N*k)


    if verbose: #print of the Opti problem matrices. Have appropriate form for small N and k. 
        print("Matrices: ")
        print("Quad cost matrix P:\n", P_QP)
        print("M_tot:\n", G_QP)
        print("eq constraint matrix G:\n", A_QP)


    return P_QP, G_QP, A_QP

def check_QP_constraints(_G_QP, _A_QP, _v_opti, N, k):
    print("\n**** Checking the QP opti problem constraints ****")
    
    assert (N**2+N == _G_QP.shape[0]), "fct check_QP_constraints: shape prblm"
    assert (np.size(_v_opti)==2*N*(1+k)),"fct check_QP_constraints: shape prblm"

    size_v = int(np.size(_v_opti)/2)

    v1, v2 = _v_opti[0:size_v], _v_opti[size_v:]
    shape_H, shape_M = (N, size_v), (N**2, size_v)
    M = (-1)*_G_QP[:shape_M[0],:shape_M[1]]
    H = _G_QP[shape_M[0]:shape_M[0]+shape_H[0], shape_M[1]:shape_M[1]+shape_H[1]]
    
    #V1
    print("First v: ")
    #Checking that v_opti >0
    print("(Positivity) min v1= ",np.min(v1) )
    print("(Metzler test)min and max of Mv1(min of v1, supposed to be positive):",np.min(M@v1), np.max(M@v1))
    print("(Hurwitz test) min and max of Hv1(max of v1, supposed to be negative):",np.min(H@v1), np.max(H@v1))
    
    #V2
    print("Second v:")
    print("(Positivity) min v2= ",np.min(v2) )
    print("(Metzler test) min and max of Mv1(min of v2, supposed to be positive):",np.min(M@v2), np.max(M@v2))
    print("(Hurwitz test) min and max of Hv1(max of v2, supposed to be negative):",np.min(H@v2), np.max(H@v2))

    #Checking of (A_QP)v = 0 i.e that we get the 2 times the same vector 
    print("v1 and v2 comparison:")
    print("max of diff: ", np.max(np.absolute(_A_QP@_v_opti)) )  #need to check the relative diff instead
    print("min of the diff: ", np.min(np.absolute(_A_QP@_v_opti)))

def is_Metzler_and_Hurwitz_aux(M):
    ''' Auxiliary function of the fct is_Metzler_and_Hurwitz. M = A+B_delta@K'''
    print("\n**** CHeck if the input matrix is Metzler and Hurwitz ****")
    N = M.shape[0]

    Mat = np.copy(M)
    for i in range(N):
        Mat[i, i] = np.inf
    Mat = np.sort(Mat.flatten())
    M_flat_sorted = np.sort(M.flatten())
    nb_neg_terms = np.sum(Mat<0)

    print("- Metzler Check:") 
    print("min should be >0: 3 lowest and 3 highest values without the diagonal: ",
           Mat[:3], Mat[-3-N:-(N-1)])  #Mat[-3-N:-(N-1)]: so don't take the diagonal
    print("extremal values with the diagonal: ",  M_flat_sorted[:3], M_flat_sorted[-3:])
    print("Proportion of <0 non diagonal terms (supposed to be 0%): ",
           nb_neg_terms/(N**2-N)*100, "% = ", nb_neg_terms , "terms") 

    eigen_values = np.linalg.eigvals(M)
    eig_v_sorted = np.sort(eigen_values.real)
    
    print("- Hurwitz check:")
    print("max(Re(eigval)) should be negative. 3 lowest and 3 highest values without the diagonal:", eig_v_sorted[:3], eig_v_sorted[-3:])
    print("Proportion of eigenvalue with Re(eigv)>0 (supposed to be 0):", np.sum(eig_v_sorted>0)/(eig_v_sorted.shape[0])*100, "%")

def is_Metzler_and_Hurwitz(M):
    
    is_Metzler_and_Hurwitz_aux(M)

    print("\n\nSame by putting + min")
    Mat = np.copy(M)
    for i in range(M.shape[0]):
        Mat[i, i] = np.inf
    min_M = np.min(Mat)

    pos_mat = (Mat < 0 )*np.absolute(min_M)
    print("Check sum:", np.sum(pos_mat)/np.absolute(min_M))
    is_Metzler_and_Hurwitz_aux(M+ pos_mat)
    # is_Metzler_and_Hurwitz_aux(M+ np.absolute(min_M))
#Tests of is_Metzler_and_Hurwitz
if False:
    Test_mat = np.identity(3)
    Test_mat[0,1], Test_mat[2, 1], Test_mat[1, 2] = -1, -5, -3
    is_Metzler_and_Hurwitz(Test_mat)

def solve_opti_QP(N_test, k_test, _P_QP, _G_QP, _A_QP, v_init=None, _solver="piqp", reg_h=0, reg_lb=0):

    size_v = 2*N_test*(1+k_test)

    print("Initial V: ", v_init)
    print("BEGINNING OF THE SOLVING FOR POSITIVE CTRL")
    t_i = time.time()
    v_opti = qpsolvers.solve_qp(P=csc_matrix(_P_QP), q=np.zeros(size_v), 
                    G=csc_matrix(_G_QP), h=reg_h*np.ones(N_test + (N_test)**2), A=csc_matrix(_A_QP), 
                    b=np.zeros(int(size_v/2)), lb=reg_lb*np.ones(size_v), solver=_solver, initvals=v_init)
    print("Solving duration: ", time.time()-t_i, "s")
    if v_opti is None:
        assert False,  "fct solve_opti_QP: No solution found"

    #Construct d and z by take the first half of the opti problem
    d, z = v_opti[:N_test], np.array([v_opti[N_test+i*k_test : N_test+(i+1)*k_test] for i in range(N_test)] )
    K = np.array([z[i,:]/d[i] for i in range(N_test)]).T #K = [z_1/d_1 |...|z_n/d_n]

    return v_opti, d, z, K
#Test of different A and B for the opti algo
if False:
    Test_mat = np.identity(3)
    Test_mat[0,1], Test_mat[2, 1], Test_mat[1, 2] = -1, -5, -3
    is_Metzler_and_Hurwitz(Test_mat)


####Test of different A and Bdelta for the opti algo
if False:
    N_test = 128
    k_prior = 5

    ##Matrix A and deltaB
    A_test = np.random.rand(N_test, N_test)
    # A_test = (-1)*np.identity(N_test)
    # for k in range(k_prior):
    #     A_test[k, k]
    
    #Counting the number of eigenvalues with >0 Real part (how much k do I need to stabilise)
    A_eigval = np.linalg.eigvals(A_test)
    count_pos_eigval = np.sum(A_eigval.real>0)
    print("Number of unstable modes of A_test: ", count_pos_eigval)

    #Defining k taking the stabilisation into account
    k_test = count_pos_eigval
    # k_test = k_prior
    Bdelta_test = np.zeros((N_test, k_test))

    # print("\nA_test:\n", A_test)
    # print("\nBdelta_test:\n", Bdelta_test)
    # for k in range(k_test):
    #     Bdelta_test[k, k] = 1, -2


    if True:
        v_init = np.ones(2*N_test*(1+k_test))
        v_init[N_test: N_test*(1+k_test)] = np.absolute(np.random.rand(N_test*k_test))
        v_init[N_test*(1+k_test):] = v_init[: N_test*(1+k_test)]

    P_QP, G_QP, A_QP = Mat_QP_problem(M_1=A_test, M_2=Bdelta_test)
    v_opti, d, z, K = solve_opti_QP(N_test, k_test, P_QP, G_QP, A_QP)
    print("\n Matrix K:\n")
    # print_neat_matrix(K)
    # Check the QP problem constraints
    check_QP_constraints(_G_QP=G_QP, _A_QP=A_QP, _v_opti=v_opti,N=N_test, k=k_test)

    #Check if A+deltaBK is Metzler & Hurwitz
    A_deltaBK = A_test+Bdelta_test@K
    is_Metzler_and_Hurwitz(A_deltaBK)

    print("\nA+Bdelta@K Matrix:\n", A_deltaBK)
    print(A_deltaBK)




############### Proportionnal Control ###############

##Dispersion relation
def dispersion_benney_air_jets(k, alpha=0):
    k_L = k*nu
    return -2*(1.j)*k_L + k_L**2*(8/15*(Re-Re_0)-alpha/3-k_L**2/(3*Ca))

def dispersion_benney_BS(k, alpha=0):
    k_L = k*nu
    print("nu: ", nu)
    return   -alpha*(1+2*Re*k_L/3*(1.j))-2*(1.j)*k_L + (k_L**2)*8/15*((Re-Re_0)-5*(k_L**2)/(8*Ca))

alpha_B_BS = 16*Ca*(Re-Re_0)**2/75
alpha_B_AJ = 24/15*(Re-Re_0)

domain_k = np.linspace(0, k_0*2, 100)
plt.plot(domain_k, dispersion_benney_BS(domain_k).real, label=r"$\alpha = 0$")
plt.plot(domain_k, dispersion_benney_BS(domain_k, alpha_B_BS).real, label=r"$\alpha = \alpha_B$")
plt.xlabel("k"), plt.ylabel(r"$\lambda_k$")
plt.ylim(bottom=-1)
plt.axhline(y=0 ,color='k'), plt.axvline(x=k_0, color='k')
plt.legend(), plt.title("Dispersion relation of blowing and suction control")
plt.show()

plt.plot(domain_k, dispersion_benney_air_jets(domain_k).real, label=r"$\alpha = 0$")
plt.plot(domain_k, dispersion_benney_air_jets(domain_k, alpha_B_AJ).real, label=r"$\alpha = \alpha_B$")
plt.xlabel("k"), plt.ylabel(r"$\lambda_k")
plt.ylim(bottom=-1)
plt.axhline(y=0 ,color='k'), plt.axvline(x=k_0, color='k')
plt.legend(), plt.title("Dispersion relation of blowing and suction control")
plt.show()
