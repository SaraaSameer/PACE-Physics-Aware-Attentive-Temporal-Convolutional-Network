# Physics-Aware Attentive Temporal Convolutional Network for Battery Health Estimation

* Sara Sameer, Wei Zhang, Lou Xin, Gao Yulin

A lightweight deep learning framework for accurate and efficient battery State-of-Health (SoH) monitoring. PACE combines temporal convolutional networks with physics-informed features from equivalent circuit models and chunked attention mechanisms to achieve superior performance while maintaining computational efficiency.

## Overview of the Architecture
![Image](https://github.com/user-attachments/assets/7eb5a6ca-5fbd-4ca4-ac06-3466183951c0)

## Results
\begin{table*}[]
\caption{Performance comparison across models on battery SoH prediction. Metrics include the number of model parameters, FLOPs, RSME, MAE, and efficiency $\eta$, which measures predictive performance per model size. $\uparrow$ ($\downarrow$) means the higher (lower) the better. Best results are highlighted in bold and \textsc{Pace} achieves the best efficiency.}
\label{tab:comp}
\centering
\renewcommand{\arraystretch}{1.3}
\footnotesize
\begin{tabular}{c|rr|rrr|rrr|rrr}
\hline\hline
\multirow{2}{*}{Model}
& \multicolumn{1}{c}{\#Params $\downarrow$} & \multicolumn{1}{c|}{FLOPs $\downarrow$} & \multicolumn{3}{c|}{1-Cycle} & \multicolumn{3}{c|}{30-Cycle} & \multicolumn{3}{c}{50-Cycle} \\ \cline{4-12}
& \multicolumn{1}{c}{($\times 10^3$)} & \multicolumn{1}{c|}{($\times 10^6$)} & \multicolumn{1}{c}{RMSE $\downarrow$} & \multicolumn{1}{c}{MAE $\downarrow$} & \multicolumn{1}{c|}{$\eta$ $\uparrow$} & \multicolumn{1}{c}{RMSE $\downarrow$} & \multicolumn{1}{c}{MAE $\downarrow$} & \multicolumn{1}{c|}{$\eta$ $\uparrow$} & \multicolumn{1}{c}{RMSE $\downarrow$} & \multicolumn{1}{c}{MAE $\downarrow$} & \multicolumn{1}{c}{$\eta$ $\uparrow$} \\
\hline\hline
Transformer & 2,559.5 & 133.2 & 0.014 & 0.009  & 27.9 & 0.016  & 0.010 & 24.4 & 0.017  & 0.009  & 23.0 \\ \hline
TCN & 47.0 & 1.7 & 0.036  & 0.022  & 580.2 & 0.053  & 0.039 & 401.8 & 0.057  & 0.042  & 373.6 \\ \hline
BCA & 173.5 & 5.0 & 0.034 & 0.018 & 169.5 & 0.037 & 0.020  & 155.8 & 0.038  & 0.020 & 150.5 \\ \hline
\textsc{Pace} (ours) & 70.9 & 5.1 & 0.023 & 0.010  & \textbf{{613.4}} & 0.033 & 0.014 & \textbf{{427.5}} & 0.035  & 0.015 & \textbf{{403.1}} \\
\hline\hline
\end{tabular}
\end{table*}
