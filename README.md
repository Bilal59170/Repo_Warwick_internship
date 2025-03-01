# Control of Falling Liquid Film: visit at the Warwick Mathematics Institute

Welcome to this repository which contains the python code and the report that I did in my visit at the Warwick Mathematics Insitute supervised by [Dr. Susana Gomes](https://warwick.ac.uk/fac/sci/maths/people/staff/gomes/) and [Dr. Radu Cimpeanu](https://www.raducimpeanu.com/).

The goal of this 5-month visit was to take interest into Control problem on specific fluids systems. More precisely, the problem was to stabilize a wavy thin film falling down an inclined plane. 

For the description of the Mathematics, Physics or the code, see the sections below. For even more details, feel free to take a look at the report ``report.pdf``. 

  
## Physics & Mathematics behind the code


## The Code
Here is a brief description of each python file.


- ``solver_BDF.py``: Solving of the Benney equations. Contains the implementation of the numerical scheme. Cf part IV of the report for the description.
- ``benney_eq_verif.py``: Verification of the numerical scheme used to solve the Benney equation. Cf Part IV of the report for the description of the verifications.
- ``Control_file.py``: Implementation of the proportional and positive control strategies. Cf part V of the report for the theoretical descriptions.
- ``main_benney_eq_Ctrl.py``: Solving of the Benney equations using ``solver_BDF.py`` (or load an existing solution already computed) and apply a control on it. 
- ``header.py``: Definition of all the global variables and functions used in several .py files, like the physical variables of the system.  