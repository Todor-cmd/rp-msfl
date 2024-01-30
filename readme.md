<h1>Robustness Against Untargeted Attacks of Multi-Server Federated Learning for
Image Classification</h1>

This repository provides extensions of state-of-art defenses Median, Krum, Multi-Krum, Bulyan, Trimmed-Mean and DnC to Multi-Server Federated Learning with FedMes. As-well as the Min-Max attack.

Along with the data collected in results folder, the code should allow for transparent evaluation and reproductions of the findings in our paper *Robustness Against Untargeted Attacks of Multi-Server Federated Learning for Image Classification: Are Defenses Based on Existing Methods Enough?* 

This work was done as part of the 2023-2024 Q2 Reseach Project at TU Delft. To see the full works done as part of this project click [here](https://cse3000-research-project.github.io/).


Repository inspired by:  
https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning  
https://github.com/GillHuang-Xtler/msfl

In collaboration with:  
https://github.com/Riliano/rp-msfl


To set-up repository run:
```
pip install -r pip_requirements.txt
```
For first time, to download data-sets, run:
```
python data.py
```

Then arguments.py can be set up for the required setting and main.py can be run. 