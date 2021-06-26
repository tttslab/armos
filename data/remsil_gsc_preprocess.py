import librosa
import numpy as np
import os
import re
import hashlib
import logging
import configparser

#------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
conf = configparser.SafeConfigParser()
conf.read("config.ini")
wavedir       = conf.get('main', 'wave_dir') + '/'
outdir        = conf.get('main', 'feat_dir') + '/'
datalist      = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', \
                 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', \
                 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
SAMPLING_RATE = int(conf.get('main', 'sampling_rate'))
FEAT_DIM      = int(conf.get('main', 'in_size'))
FEAT_TYPE     = conf.get('main', 'feat_type')
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# This function is the same as the code in README.md of speech_commands
# https://github.com/tensorflow/tensorflow/blob/40f9a0744af6e89f5e84980c02116ba670759b45/tensorflow/examples/speech_commands/input_data.py#L70
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
logging.info('Start making features')
if not os.path.exists(outdir):
  os.makedirs(outdir)
training_list   = open(outdir+'training_list.txt',   'a')
testing_list    = open(outdir+'testing_list.txt',    'a')
validation_list = open(outdir+'validation_list.txt', 'a')
for command in datalist:
    if os.path.exists(outdir+command+'/.complete'):     # check if data exist. if true, skip feature generation, 
        logging.info(command+' data is already prepared')
        continue
    
    logging.info('Processing `'+command+'`... ')
    if not os.path.exists(outdir+command):
      os.makedirs(outdir+command)

    for wavfile in os.listdir(wavedir+command):
        audio, sr = librosa.load(wavedir+command+'/'+wavfile, sr=SAMPLING_RATE)  # load wav file

        if   FEAT_TYPE == 'mfcc':
          feats = librosa.feature.mfcc(audio, sr=SAMPLING_RATE, n_mfcc=FEAT_DIM+1, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
          feats = np.asarray(feats, dtype=np.float32)
          feats = feats[1:, :]
        elif FEAT_TYPE == 'fbank':
          feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=FEAT_DIM, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
          feats = np.asarray(feats, dtype=np.float32)
        elif FEAT_TYPE == 'logfbank':
          feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=FEAT_DIM, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
          feats = np.log(np.asarray(feats+1e-30, dtype=np.float32))

        np.save(outdir+command+'/'+wavfile[:-4]+'.npy', feats.T)  # save features
        partition = which_set(wavfile, 10, 10)                   # divide to "training", "validation", "testing" 3 parts
        if partition == 'training':
            training_list.write(command+'/'+wavfile[:-4]+'.npy\n')
        if partition == 'testing':
            testing_list.write(command+'/'+wavfile[:-4]+'.npy\n')
        if partition == 'validation':
            validation_list.write(command+'/'+wavfile[:-4]+'.npy\n')

    with open(outdir+command+'/.complete', 'w') as f:
      f.write('complete preprocess')

training_list.close()
testing_list.close()
validation_list.close()
logging.info('Done')
#------------------------------------------------------------------------------
