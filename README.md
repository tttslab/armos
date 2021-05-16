# physical-speech-synth
acoustic-to-articulatory inversion by reinforcement learning

# Requirement
requirements.txt is in each directory.
Note that python2 is necessary for VocalTractLab.

# Data preparation
Numpy 2d-array whose first dimension is time and second one is acoustic feature's dimension is available.
Data preparation scripts for Google Speech Commands and LibriSpeech are prepared in scripts/gsc_preprocess.py and scripts/librispeech_preprocess.py, so copy this to experimental directory and you can run it.

# Usage
Prepare datalist/training_list.txt, which is the list of data path.
Execute a below command to make directories:
```
./makedir.sh
```
Then you can execute below commands.
```
./pretrain.py
./train.py 1>/dev/null
```
VTL outputs the standard output every time, so it is recommended to discard the standard output.
The models and optimizers are saved in exp/ directory every 1000 iterations.

pretrain.py just makes the output of policy close to 0.5.
So it is not necessarily to run this.

If you resume training from the middle, 
you should rewrite train.py directly: 

Get the lasted saved model's iteration number;

Delete log after that iteration number;

Change model load and sigma_init and iteration start in train.py.

The Log file is added to existing one.

# Evaluation
To make a reward curve in training step:
```
./make_reward_graph.sh
```
The output is log/train_reward_graph.csv.

To calculate MSE for validation set:
```
./validation.py iteration validation batch_size num_cpu
```
The log is saved in log/valid.log.

To calculate MSE for test set:
```
./test.py iteration testing batch_size num_cpu
```
The log is saved in log/test.log.

To calculate correlation coefficient for test data:
```
./eval_corr.sh
```
If you write "seq 1 10" in eval_corr.sh, evaluate the model from 1000 to 10000 iteration.

To synthesize a sound, an example is:
```
./generate_sound.py path/to/hoge.wav exp/p1000.model
sox synthesized.wav output.wav trim 0.05 1.6
```
A synthesized sound is padded for 5 frames.
Therefore, you should cut the sound manually.
