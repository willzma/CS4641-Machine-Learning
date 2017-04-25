I did not write a substantial amount of source code for this paper, only making minor changes to Juan San Emeterio's library in order to do my own experiments.
For example, I changed around analysis hyperparameters, the reward functions, as well as the actual GridWorlds in use. All other code besides my minor changes
are credited to Juan Jose San Emeterio's extension to the Brown-UMBC Reinforcement Learning and Planning library, both of which are available on GitHub
at the following links, respectively:

https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4
https://github.com/jmacglashan/burlap

Installation/Running:
1. Install Java 8 JDK
2. To run experiments, simply run EasyGridWorldLauncher.java and HardGridWorldLauncher.java, both of which can be found in the included code repository

Where to find files:
- data from experiments can be found in GridWorldData
- code can be found in omscs-cs7641
- parameters for running the experiments can be found in the actual paper itself, wma61-analysis.pdf