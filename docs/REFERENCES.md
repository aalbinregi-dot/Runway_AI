# References

Each entry notes what it is actually used for, rather than being a general
reading list.

## Pose estimation

1. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., &
   Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose Tracking.*
   CVPR Workshop on Computer Vision for AR/VR.
   — the model behind MediaPipe Pose; source of the 33-landmark topology.
2. Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception
   Pipelines.* arXiv:1906.08172.
3. Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2021). *OpenPose:
   Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.* IEEE
   TPAMI 43(1). — the main alternative backend.
4. Martinez, J., Hossain, R., Romero, J., & Little, J. (2017). *A Simple Yet
   Effective Baseline for 3D Human Pose Estimation.* ICCV.
   — the route for the 3D-lifting extension that would fix the sagittal criteria.
5. Pavllo, D., Feichtenhofer, C., Grangier, D., & Auli, M. (2019). *3D Human
   Pose Estimation in Video with Temporal Convolutions and Semi-Supervised
   Training.* CVPR.

## Gait analysis and biomechanics

6. Drillis, R., & Contini, R. (1966). *Body Segment Parameters.* Technical
   Report 1166-03, New York University. — the anthropometric segment ratios used
   by the synthetic generator.
7. Saunders, J. B., Inman, V. T., & Eberhart, H. D. (1953). *The Major
   Determinants in Normal and Pathological Gait.* JBJS 35-A(3). — the compass-gait
   model, and why a rigid version overpredicts hip displacement.
8. Perry, J., & Burnfield, J. (2010). *Gait Analysis: Normal and Pathological
   Function* (2nd ed.). SLACK. — stance/swing structure and event definitions.
9. Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*
   (4th ed.). Wiley. — filtering conventions and joint-angle computation.
10. Hausdorff, J. M. (2007). *Gait Dynamics, Fractals and Falls.* Human Movement
    Science 26(4). — stride-time CV as the standard regularity measure.
11. Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). *Use of Force Platform
    Variables to Quantify the Effects of Chiropractic Manipulation on Gait
    Symmetry.* J. Manipulative Physiol. Ther. 10(4). — the symmetry index.
12. Patterson, K. K. et al. (2010). *Evaluation of Gait Symmetry after Stroke.*
    Gait & Posture 31(2). — step-time asymmetry.

## Signal processing

13. Savitzky, A., & Golay, M. J. E. (1964). *Smoothing and Differentiation of
    Data by Simplified Least Squares Procedures.* Analytical Chemistry 36(8).
    — chosen over a moving average because it preserves gait peak timing.
14. Boashash, B. (1992). *Estimating and Interpreting the Instantaneous
    Frequency of a Signal.* Proc. IEEE 80(4). — Hilbert instantaneous phase and
    its boundary artefacts, which caused a real bug here.
15. Welch, P. (1967). *The Use of FFT for the Estimation of Power Spectra.* IEEE
    Trans. Audio Electroacoust. 15(2).

## Machine learning

16. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree.* NeurIPS. — the histogram-based boosting scikit-learn's
    `HistGradientBoostingRegressor` implements.
17. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting
    Machine.* Annals of Statistics 29(5).
18. Breiman, L. (2001). *Random Forests.* Machine Learning 45(1). — source of
    permutation importance.
19. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based
    models still outperform deep learning on typical tabular data?* NeurIPS
    Datasets and Benchmarks. — the evidence behind choosing boosting over a
    neural network at this data scale.
20. Kraskov, A., Stogbauer, H., & Grassberger, P. (2004). *Estimating Mutual
    Information.* Physical Review E 69(6). — the estimator behind
    `mutual_info_regression`, used for the non-monotone feature sweep.

## Automated action quality assessment

21. Pirsiavash, H., Vondrick, C., & Torralba, A. (2014). *Assessing the Quality
    of Actions.* ECCV. — the foundational AQA formulation this project sits in.
22. Parmar, P., & Morris, B. T. (2019). *What and How Well You Performed? A
    Multitask Learning Approach to Action Quality Assessment.* CVPR.
    — multi-criterion scoring, the closest published analogue.
23. Tang, Y. et al. (2020). *Uncertainty-aware Score Distribution Learning for
    Action Quality Assessment.* CVPR. — relevant to modelling judge disagreement
    once expert labels exist.
24. Lei, Q., Du, J.-X., Zhang, H.-B., Ye, S., & Chen, D.-S. (2019). *A Survey of
    Vision-Based Human Action Evaluation Methods.* Sensors 19(19).

## Synthetic data and sim-to-real

25. Varol, G. et al. (2017). *Learning from Synthetic Humans.* CVPR.
26. Tobin, J. et al. (2017). *Domain Randomization for Transferring Deep Neural
    Networks from Simulation to the Real World.* IROS. — the justification for
    randomising camera pose, noise and frame rate in the generator.
27. Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak
    Supervision.* VLDB 11(3). — programmatic labelling, the family this
    generator belongs to.

## Measurement and agreement

28. Shrout, P. E., & Fleiss, J. L. (1979). *Intraclass Correlations: Uses in
    Assessing Rater Reliability.* Psychological Bulletin 86(2).
29. Krippendorff, K. (2018). *Content Analysis: An Introduction to Its
    Methodology* (4th ed.). SAGE. — alpha for inter-rater agreement on the
    planned expert-labelled tier.
