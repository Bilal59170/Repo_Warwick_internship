# Control of Falling Liquid Film: visit at the Warwick Mathematics Institute

Welcome to this repository which contains the python code and the report that I did in my visit at the Warwick Mathematics Insitute supervised by [Dr. Susana Gomes](https://warwick.ac.uk/fac/sci/maths/people/staff/gomes/) and [Dr. Radu Cimpeanu](https://www.raducimpeanu.com/).

The goal of this 5-month visit was to take interest into Control problem on specific fluids systems. More precisely, the problem was to stabilize a wavy thin film falling down an inclined plane with controled air jet pertubations. 

For the description of the Mathematics, Physics or the code, see the sections below. For details on the folders, read the ``READ_ME.md`` files of the folders you want (``Benney_equation_code`` or ``KS_equation_code``) . For even more details, feel free to take a look at the report ``Bilal_BM_report.pdf``. 

  
# Physics & Mathematics behind the code
## The equations

All this work is about studying Reduced-order models (ROM) of the Navier-Stokes equations. 

The first part of the visit (part II of the report) is an introduction work. It focuses on the KS (Kuramoto-Sivashinski) equation: 

$$u_{,t} + u_{,xxxx} + u_{xx} + uu_{,x} = 0,$$

 on the domain $[0, L)$, supposing u L-periodic. This equation is a PDE (Partial Differential Equation) a quite simple non linearity: we say that it is a weakly non-linear PDE. This is one of the simplest ROM of the Navier-Stokes equation. We solved it numerically using a specific scheme (cf report) and observed the behaviour of the solution which can be quite diverse depending on the parameter 
 $$\nu = \Big( \frac{2\pi}{L} \Big)^2.$$

The main part of the visit (part III to V of the report) focuses on the study the Benney equation, another ROM of the Navier-Stokes equations with more complex non-linearities. We added to this equation a term of normal stress $N_s$ of the air jet component, neglecting the tangential component. The Benney equation is a mass conservation equation with $h$ being the interfacial height of the fluid, and $q(x,t) = \int_0^{h(x,t)} u(x,y,t)dy$ the horizontal flux of the falling liquid:   

$$\begin{equation}
\left\lbrace
\begin{aligned}
    h_{,t}+q_{,x} &= 0,\\
    q(x, t) &= \frac{h^3}{3}(2-p_{l0,x})+Re\frac{8h^6h_{x,}}{15}\\
    p_{l0}&= N_s + 2(h-y)cot(\theta) - \frac{h_{,xx}}{Ca}
\end{aligned}
\right., 
\end{equation}$$

which gives: 

$$\begin{equation}
    h_{,t} + h_{,x}h^2 \Big( 2-N_{s, x}-2h_{,x}cot(\theta) + \frac{h_{,xxx}}{Ca}\Big) - \frac{h^3}{3}\Big(N_{s, xx} + 2h_{,xx}cot(\theta)  - \frac{h_{,xxxx}}{Ca} \Big) + \frac{8Re}{15} \Big( 6h^5 {h_{,x}}^2 + h^6 h_{,xx} \Big) = 0.
\end{equation}$$



## Control Strategies
We designed two main control strategies to stabilize the Benney system. Let  

$$\tilde{h} = \frac{h-1}{\delta}, \tilde{N}_s = \frac{N_s}{\delta}$$ 

the zoomed discrepancy of h around its stable normalized state $h=1$ and zoomed air jet perturbation.

- **Proportional Control**: All the air jets blow/suck the same way proportionaly to the size of the gap $\tilde{h}$. Which gives  $$N_s = \alpha \tilde{h}, \quad \alpha >0.$$ 
As we just wanted blowing only air jets, we did 

$$\tilde{N}_s = \alpha |\tilde{h}|^+.$$ 

- **LQR Control**: We linearized the Benney equation into : 

$$\tilde{h}_{,t} = A\tilde{h} + BN_s = A\tilde{h} + BK\tilde{h}$$ 

and we minimized the quadratic cost 

$$\begin{equation}
    \kappa(u) = \int_0^{+\infty}\int_0^L \left[\beta\tilde{h}(x,t)^2+ (1-\beta)\tilde{N}_s(x,t)^2\right]dxdt.
\end{equation}$$

As we just wanted blowing only air jets, we took the control 

$$u = |K\tilde{N}_s|^+$$

and actualized the system 

$$\tilde{h}_{,t} = A\tilde{h} + Bu.$$ 

## The Code
Here is a brief description of each python file. The output of each file and its structure is detailed at the beginning of each of the code files.

### KS equation
- ``main_KS_equation.ipynb``: Solving of the KS equation with a specific numerical scheme (cf report part II). Visualisation of the behaviour of the solution.

### Benney equation
- ``solver_BDF.py``: Solving of the Benney equations. Contains the implementation of the numerical scheme. Cf part IV of the report for the description.
- ``benney_eq_verif.py``: Verification of the numerical scheme used to solve the Benney equation. Cf Part IV of the report for the description of the verifications.
- ``Control_file.py``: Implementation of the proportional and positive control strategies. Cf part V of the report for the theoretical descriptions.
- ``main_benney_eq_Ctrl.py``: Solving of the Benney equations using ``solver_BDF.py`` (or load an existing solution already computed) and apply a control on it. 
- ``header.py``: Definition of all the global variables and functions used in several .py files, like the physical variables of the system.  
