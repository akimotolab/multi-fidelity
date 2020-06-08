# Multi-Fidelity Continuous Minimization under Prior and Posterior Constraints

#### Reference

Y. Akimoto, N. Sakamoto, M. Ohtani. Multi-Fidelity Continuous Minimization under Prior and Posterior Constraints, in Parallel Problem Solving from Nature, PPSN, 2020. Accepted.

#### List of source codes:

* ddcma.py : DD-CMA-ES [Akimoto and Hansen, ECJ2019](https://doi.org/10.1162/evco_a_00260). Code is based on [Gist](https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518)
* constraint_handling.py : QUAK + QRSK constraint handling. ARCH [Sakamoto and Akimoto, GECCO2019](https://doi.org/10.1145/3321707.3321717) for QUAK, MCR [de Paula Garcia et al, 2017](https://doi.org/10.1016/j.compstruc.2017.03.023) (or its modification) for QRSK
* adaptive_simulator_switcher.py : Adaptive Simulator Selection for Multi-Fidelity Optimization [Akimoto et al, GECCO2019](https://doi.org/10.1145/3321707.3321709). Code is based on [Gist](https://gist.github.com/youheiakimoto/0630db3d461f855fb09d799d5dc48dd8)
* compliance_minimization.py : Compliance Minimization Problem Definition [Andreassen et al, 2011](https://doi.org/10.1007/s00158-010-0594-7). Code is based on [topopt_cholmod.py](http://www.topopt.mek.dtu.dk/-/media/Subsites/topopt/apps/dokumenter-og-filer-til-apps/topopt_cholmod.ashx?la=da&hash=4F2AC8256B93F55884C753207053A05B04145F96)

#### Scripts for evaluation:

* experiments_fixedfidelity.py
* experiments_multifidelity.py
* experiments_evaluation.py
* experiments_plot.py

#### Sample Script (notebook):

* example.ipynb
