# cal-free-m-worm
calcium dynamics in TRNs  &amp; behavioral analysis in freely moving worms *C. elegans*

## Description
0. **STEP0**:
   + Prerequisite to run the codes is to have trackmate neuron trajectory file prepared in *.xml format.
The file should not include any other trajectories that do not represent neurons. Moreover, you need to label the neurons in NeuronLabels.csv file 
and give the path to tracks files.
2. **STEP1_calcium_subtract_trackmate.py**
   + This step uses the neuron positions in the consecutive frames and integrate around in two channels. At the end it should save '*.xlsx' file.    
4. **STEP2_skeletonise.py**
   + This step segments the videos including single worms, and find the skeletons. Does not perform well with self-crossed frames. At the end should save '*_skel.csv' file.
5. **STEP3_correlate_behaviour_activity.py**
   + This step orients the skeleons using the neurons position data (for now works with TRNs only), find the tangential and normal components of the velocity and should save '*_analysed.pkl' file at the end.
7. **STEP_cross_cov_analysis.py**
   + This step generates calculates the windowed cross-covariance between neural activity and velocity.


## How to cite 
part of the manuscript: Active sensing of substrate mechanics optimizes locomotion of *C. elegans*, A. Pidde, C. Agazzi, M. Porta-de-la-Riva, C. Martínez-Fernández, A. Lorrach, A. Bijalwan, E. Torralba, R. Das, J. Munoz, M. Krieg


 
