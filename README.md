# Detecting Model Inconsistency Attacks Against Federated Learning Systems


Architecture du VAE 
![The variational auto-encoder architecture: encoder,decoder, and the re-parameterization trick.](/home/mbouharoun/my_vae.png)

**Requirements**:

* python3 / jupyter
  * TensorFlow2 
  * numpy
  * matplotlib




The notebook 'CanaryGradientInteractive_POC.ipynb' allows to test the canary gradient attack in an interactive fashion. A non-interactive version of the code is available in 'canary_attack_main.py'. This can be used by providing a configuration file as input. For instance:

```
python3 main.py -s nf-unsw1_nf-unsw2.py
```

The main hyper-parameters for 'canary_attack_main.py' are in 'settings/\_\_init\_\_.py'

The result will be saved in the 'results' folder and it can be read using the notebook 'plot_data.ipynb'. The script 'run_all.sh' can be used to run all the tests.




