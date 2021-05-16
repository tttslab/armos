import numpy as np
import librosa
import glob
import os
import re
import configparser
import logging

#------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
conf = configparser.SafeConfigParser()
conf.read("config.ini")
wavedir       = conf.get('main', 'wave_dir') + '/'
outdir        = conf.get('main', 'feat_dir') + '/'
SAMPLING_RATE = int(conf.get('main', 'sampling_rate'))
FEAT_DIM      = int(conf.get('main', 'in_size'))
FEAT_TYPE     = conf.get('main', 'feat_type')
#------------------------------------------------------------------------------

os.makedirs(outdir, exist_ok=True)
training_datalist = open(outdir + 'training_datalist.txt', 'w')
processed = 0
for data in glob.glob(wavedir + '/*/*/*.flac'):
        dir_name  = data.rsplit('/', 3)
        utt_name  = dir_name[-3] + '/' + dir_name[-2] + '/' + dir_name[-1][:-5]
        os.makedirs(outdir + dir_name[-3] + '/' + dir_name[-2], exist_ok=True)
        audio, sr = librosa.load(data, sr=SAMPLING_RATE)
        if FEAT_TYPE == 'mfcc':
                feats = librosa.feature.mfcc(audio, sr=SAMPLING_RATE, n_mfcc=FEAT_DIM+1, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
                feats = np.asarray(feats, dtype=np.float32)
                feats = feats[1:, :]
        elif FEAT_TYPE == 'fbank':
                feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=FEAT_DIM, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
                feats = np.asarray(feats, dtype=np.float32)
        elif FEAT_TYPE == 'logfbank':
                feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=FEAT_DIM, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(SAMPLING_RATE/100.0))
                feats = np.log(np.asarray(feats+1e-30, dtype=np.float32))
        np.save(outdir + utt_name, feats.T)
        training_datalist.write(utt_name + ' ' + outdir + utt_name + '.npy\n')

        processed += 1
        if processed%100 == 0:
                logging.info(str(processed) + ' utterances have been processed.')
training_datalist.close()
