    ###### Solver of the Benney equation with BDF & Finite Difference ######

def solve_Benney_BDF_FD(L_x, T, CFL_factor):
        ##VARIABLEs: steps & domain
    #Steps: 2 ways
    #Space, then time
    N_x = 512 #To choose little at first and then, increase 
    dx = L_x/N_x #and not  dx = L_x/(N_x-1) as it periodic along x-axis so we don't count the last point so Lx-dx = (Nx-1)dx
    dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4
    CFL_factor = 100 #take it big 
    dt = dx/U_N/CFL_factor #CFL conditions
    N_t = int(T/dt+1) #bcs (N_t-1)dt = T
    print("Nb of (space, time) points: ", (N_x, N_t) )

    #Time, then space
    # N_t = 256 #To choose little at first and then, increase 
    # dt = T/(N_t-1)
    # CFL_factor = 100 #take it big 
    # dx = dt*U_N*CFL_factor #My own CFL conditions 
    # N_x = int(L_x/dx) #as it's periodic. no +1 
    # dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4

    print("Nb of (space, time) points: ", (N_x, N_t) )

    #Time & space domains
    domain_x = np.linspace(0, L_x, N_x, endpoint=False) #Periodic domain: we don't need the last point (endpoint=False)
    domain_t = np.linspace(0, T, N_t, endpoint=True) #Not periodic: we need last point
    #test the boundaries of the domain (0 and L_x -dx if well initialised)
    print("First and last point of the domain:", domain_x[0], domain_x[-1]) 
    print("Lx-dx:", L_x-dx)


    ##Initial conditions 
    #Somes functions to use for Initial Condition
    def sincos(x, _h_mean, _ampl_c, _ampl_s, _freq_c, _freq_s):
        return _h_mean + _ampl_c*np.cos((2*np.pi/L_x)*_freq_c*x) + _ampl_s*np.sin((2*np.pi/L_x)*_freq_s*x) #Initial condition. (sinus, periodic on D)

    #initial condition
    h_mat = np.zeros((N_t, N_x)) #Matrix of the normalised height. Each line is for a given time from 0 to (N_t-1)*dt
    h_mean = 1
    ampl_c, ampl_s, freq_c, freq_s = 0.02, 0.01, 3, 1  #frequencies need to be a relative to have periodicity
    h_mat[0,:] = sincos(domain_x, h_mean, ampl_c, ampl_s, freq_c, freq_s)

    if True:#Plot of initial condition
        plt.plot(domain_x, h_mat[0, :])
        plt.title("Initial conditions for the normalized height h ")
        plt.show()



    ######## SOLVING #######

    ###### Finite Difference & BDF Scheme ######
    ## Some useful functions
    def mat_FD_periodic(length_h, list_coef):
        '''Finite difference matrix with periodic boundary conditions. Cf matdiag notation in the "Finite difference" part 
        in the solving of KS equation in the obsidian file
        input: list_coef = [a, b, c, d, ..] list of coef (int)
        output: returns a mat with diagonal a, subdiag b (with periodic extension), supdiag c (same), 2nd subdiag d etc..'''

        n = len(list_coef)
        assert (n<=length_h), "fct mat_FD_periodic: problem in the dimensions"
        mat_FD = np.zeros((length_h, length_h))
        for i in range(n):
            nb_diag = (-1)**(i+1)* int((i+1)/2) # 0:0; 1:1; 2:-1; 3:2; 4:-2...

            for j in range(length_h):
                mat_FD[(j+nb_diag)%length_h, j%length_h] = list_coef[i]

        return mat_FD
    #Tests
    print(mat_FD_periodic(5, [0, -1, 1, -2, 2]))
    # print(mat_FD_periodic(5, [0, -1, 1, -2, 2, 3])) #Case where the assert raises

    #Finite Difference matrices
    mat_DF_x = mat_FD_periodic(N_x, [0, -1, 1])/(2*dx)
    mat_DF_xx = mat_FD_periodic(N_x, [-2, 1, 1])/dx_2
    mat_DF_xxx = mat_FD_periodic(N_x, [3, -1, -3, 0, 1])/dx_3
    mat_DF_xxxx = mat_FD_periodic(N_x, [6, -4, -4, 1, 1])/dx_4
    print(mat_DF_x@h_mat[0, :])

    def F_space(h_arr):
        '''Input: 
            - h_arr: array of height at time t+dt; (N_x) float array
            - h_arr_before: array of height at time t; (N_x) float array
        '''
        h_x = mat_DF_x@h_arr
        h_xx = mat_DF_xx@h_arr #no definition of h_xxx and h_xxxx bcs they are computed just once in the function

        return ( h_x*(h_arr**2)*(2*np.ones_like(h_arr)-2*h_x/np.tan(theta) + (1/Ca)*mat_DF_xxx@h_arr) 
                + (1/3)*(h_arr**3)*(-(2/np.tan(theta))*h_xx + (1/Ca)*mat_DF_xxxx@h_arr) 
                + (8*Re/15)*(6*(h_arr**5)*(h_x**2) + (h_arr**6)*h_xx))

    def time_scheme_coef(_p):
        '''give the coefficient of the numerical time method'''
        match _p:
            case 1:
                return (np.array([-1, 1]), np.array([1]))
            case 2:
                return (np.array([1/2, -2, 3/2]), np.array([-1, 2]))
            case 3:
                return (np.array([-1/3, 3/2, -3, 11/6]), np.array([1, -3, 3]))
            case 4:
                return (np.array([1/4, -4/3, 3, -4, 25/12]), np.array([-1, 4, -6, 4]))
            case 5:
                return (np.array([-1/5, 5/4, -10/3, 5, -5, 137/60]), np.array([1, -5, 10, -10, 5]))
            case 6:
                return (np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 147/60]), np.array([-1, 6, -15, 20, -15, 6]))
            case _:
                raise Exception("Function alpha_gamma_coef: Error in the calculus, wrong p value.")
            
    def F_time(h_arr, h_arr_before):
        '''Function to step the order of the time step method'''
        return (h_arr-h_arr_before)/dt


    ## Testing the first time step
    f_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[0,:]) + F_space(h_arr)

    if False: #Testing of Newton's method: scipy.optimize.newton
        ##Test of the newton method with 1st order implicit time scheme
        print("\n#### TEST TO SET THE NEWTON METHOD ####")

        print("len(h_mat[0, :]) = N_x:", len(h_mat[0, :]) == N_x)

        '''One loop of Newton's method on the Initial Condition. 
        - Returns (root, converged, derivative). root: root of the function, converged: array of the converged dimensions,
        derivative: say if a derivative is 0 (means that the method struggles to CV)
        - Used previous height as starting point as the next one should be nearby'''
        t_i = time.time()
        root_one_loop, converged_one_loop, newton_derivatives = scipy.optimize.newton(f_objective, x0=h_mat[0, :]
                                                ,maxiter=50, full_output=True) 
        calculus_time = time.time()-t_i

        print("Computation time for one loop with N_x = {N_x}:".format(N_x=N_x), calculus_time)
        print("Number of converged coordinates, ratio/N_x ", np.sum(converged_one_loop), np.sum(converged_one_loop)/N_x)
        print("Number of derivatives = 0 (not good for the method):", np.sum(newton_derivatives))
        print("maximum of the error", np.max(np.absolute(f_newton(root_one_loop) )), "\n") 

        #See the overall physical validity of the computed height
        print("minimum new height (see if it's <0):", np.min(root_one_loop))
        print("maximum new height (see if it's too big):", np.max(root_one_loop))
        plt.plot(domain_x, root_one_loop)
        plt.title("height at t=dt with Newton method")
        plt.show()

    if False: #Testing of the scipy.optimize.root function 
        ##Test of the newton method with 1st order implicit time scheme
        print("\n#### TEST of the scipy.optimize.root function ####")

        t_i = time.time()
        result = scipy.optimize.root(f_objective, x0=h_mat[0, :]) 
        calculus_time = time.time()-t_i
        root_2, converged_2, msg_2 = result["x"], result["success"], result["message"]

        print("Computation time for one loop with N_x = {N_x}:".format(N_x=N_x), calculus_time)
        print("Method root converged: ",converged_2)
        print("maximum of the error", np.max(np.absolute(f_objective(root_2))))

        #See the overall physical validity of the computed height
        print("minimum new height (see if it's <0):", np.min(root_2))
        print("maximum new height (see if it's too big):", np.max(root_2))
        plt.plot(domain_x, root_2)
        plt.title("height at t=dt with root method")
        plt.show()


    ## Solving
    #1 order
    print("\n## SOLVING BENNEY EQ ##")
    #computation times :  ((N_x, N_t), t computation): [(128, 229),14s), ((256, 458), 60s), ((512, 915),T= 710s)]
    t_i = time.time()
    if False:#main loop
        root_method_CV_arr, root_method_errors_arr = np.zeros(N_t, dtype=bool), np.zeros(N_t)

        for n_t in range(N_t-1):
            fct_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[n_t,:]) + F_space(h_arr)
            result = scipy.optimize.root(fun= fct_objective, x0= h_mat[n_t,:]) 
            h_mat[n_t+1, :] = result["x"]
            root_method_CV_arr[n_t] = result["success"]
            root_method_errors_arr[n_t]= np.max(np.absolute(fct_objective(result["x"])))
            
            #Display of the computation progress
            nb_percent = 5 #The step of percent at which we display the progress
            if np.floor((100/nb_percent)*(n_t+1)/(N_t-1)) != np.floor((100/nb_percent)*(n_t)/(N_t-1)): #displays the progress of the computation every nb_percent
                print("Computation progress:", np.floor(100*(n_t+1)/(N_t-1)), "%; time passed until start: ", time.time()-t_i)

        total_computation_time = time.time()-t_i
        print("Total computation time:", total_computation_time)
        print("Number of time the method didn't converge & N_t", (np.sum(~root_method_CV_arr), N_t))
        print("Max error (evaluation on the supposed root) and its index", (np.max(root_method_errors_arr), np.argmax(root_method_errors_arr)))


        ##Saving or loading the solution
        bool_save = False
        bool_load_solution = False

        if bool_load_solution:
            h_mat= np.loadtxt('Benney_equation_code\\Benney_numerical_solution_Nx_128.txt')
            assert ((h_mat.shape[0]==N_t)and(h_mat.shape[1]==N_x)), "Solution loading: Problem of shape "
        elif bool_save: 
            np.savetxt('Benney_equation_code\\Benney_numerical_solution_Nx_{N_x}.txt'.format(N_x=N_x), h_mat)