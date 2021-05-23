# Armos: ARtificial MOtor Speech center toolkit 
# Trains acoustic-to-articulatory inversion NN by reinforcement learning
# Shinozaki Lab at Tokyo Institute of Technology 2021

# How to use:

# VocalTractLab is using python2, and we accordingly use python2.
# requirements.txt : library dependency list

# Initial setup for a python2 environment:
# We are using Anaconda3.
# Since some libraries are not available in the conda repository
# for python2, we use pip in a dedicated virtual environment.
$ conda create -n armos python=2.7
$ conda activate armos
$ pip install -r requirements.txt


# Misc initial setups for experiments:
# To download VTL and speech database, and do some checks and initializations,
# first run setup.sh.
# The script gives instruction for some operations. See its output.
$ bash setup.sh

# Running experiments:
# Move to one of the directories of the learning algorithms:
# policy_gradient, actor_critic, dpg_pn.

# Prepare datalist/training_list.txt, which is the list of data path.

# Make directories for trained models and log files
$ ./makedir.sh

# Edit config file
# e.g. specify the data path

# model training
$ python pretrain.py
$ python train.py 1>/dev/null

# VTL outputs the standard output every time, so it is recommended to discard the standard output.
# The models and optimizers are saved in exp/ directory every 1000 iterations.

# pretrain.py just makes the output of policy close to 0.5.
# It may be skipped.

# If you resume training from the middle, 
# you should rewrite train.py directly: 
# *Get the lasted saved model's iteration number;
# *Delete log after that iteration number;
# *Change model load and sigma_init and iteration start in train.py.
# *The Log file is added to existing one.

# Evaluation
# To make a reward curve in training step:
$ ./make_reward_graph.sh
# The output is log/train_reward_graph.csv.

# To calculate MSE for validation set:
$ python validation.py iteration validation batch_size num_cpu
# The log is saved in log/valid.log.

# To calculate MSE for test set:
$ python test.py iteration testing batch_size num_cpu
# The log is saved in log/test.log.

# To calculate correlation coefficient for test data:
$ ./eval_corr.sh
# If you write "seq 1 10" in eval_corr.sh, evaluate the model from 1000 to 10000 iteration.

# To synthesize a sound, an example is:
$ python generate_sound.py path/to/hoge.wav exp/p1000.model
$ sox synthesized.wav output.wav trim 0.05 1.6
# A synthesized sound is padded for 5 frames.
# Therefore, you should cut the sound manually.
