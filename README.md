# SSPW k-means: 
----------

Authors: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) and Takumi Fukunaga

Last page update: June 08, 2021

Latest version: 1.0.0 (see Release notes for more info) 

<br />

Introduction
----------
This repository contains the code of simplex projection based Wasserstein k-means, called SSPW k-means, that is a faster Wasserstein k-means algorithm for histogram 
data by reducing Wasserstein distance computations and exploiting sparse simplex projection. We shrink data samples, centroids, and the ground cost matrix, which 
leads to considerable reduction of the computations used to solve optimal transport problems without loss of clustering quality. Furthermore, SSPW k-means dynamically 
reduced the computational complexity by removing lower-valued data samples and harnessing sparse simplex projection while keeping the degradation of clustering quality lower. 
<br />


<br />

Paper
----------

T. Fukunaga and H. Kasai, "Wasserstein k-means with sparse simplex projection," ICPR2020. [Publisher's site](https://ieeexplore.ieee.org/document/9412131), [arXiv](https://arxiv.org/abs/2011.12542).



<br />


Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - A demonstration script. 
|algorithms             - Contains the implementation file of the proposed SSPW k-means
|tools                  - Contains some files for execution.
|datasets               - Contains some datasets.
</pre>

<br />  

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

<br />  

Demonstration
----------------------------
Run `demo` for a demonstration.
```Matlab
%% Execute a demonstration script.
demo; 
```


<br />

Notes
-------
* Some parts are borrowed from below: 

    - Staib, Matthew and Jegelka, Stefanie, "Wasserstein k-means++ for Cloud Regime Histogram Clustering," Proceedings of the Seventh International Workshop on Climate
Informatics: CI 2017, 2017, [Code](https://github.com/mstaib/cloud-regime-clustering-code).

<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) (email: hiroyuki **dot** kasai **at** waseda **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.0 (June 08, 2021)
    - Initial version.
