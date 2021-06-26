# -*- coding: utf-8 -*-
import ctypes			# for accessing C libraries 
import os			# for retrieving path names
import numpy  # for array operations
import scipy.io.wavfile # to write wav file
import configparser

conf = configparser.SafeConfigParser()
# TODO: apiのパスをどうするか
conf.read("config/dpg_pn/config.ini")
frameRate_Hz = int(conf.get('main', 'frameRate_Hz'))

def vtlSynthesize(tractParams, glottisParams, duration_s):
    dllFile = os.path.abspath('VocalTractLabApi.so')
    lib = ctypes.cdll.LoadLibrary(dllFile)

    #speakerFileName = os.path.abspath('child-1y.speaker')
    speakerFileName = os.path.abspath('JD2.speaker')
    wavFileName = os.path.abspath('reconst.wav')

    lib.vtlInitialize(speakerFileName)

    # get vtl constants
    c_int_ptr = ctypes.c_int * 1 # type for int*
    audioSamplingRate_ptr = c_int_ptr(0);
    numTubeSections_ptr = c_int_ptr(0);
    numVocalTractParams_ptr = c_int_ptr(0);
    numGlottisParams_ptr = c_int_ptr(0);
    lib.vtlGetConstants(audioSamplingRate_ptr, numTubeSections_ptr, numVocalTractParams_ptr, numGlottisParams_ptr);
    audioSamplingRate = audioSamplingRate_ptr[0]
    numTubeSections = numTubeSections_ptr[0]
    numVocalTractParams = numVocalTractParams_ptr[0]
    numGlottisParams = numGlottisParams_ptr[0]

    # get tract info
    c_numTractParam_ptr = ctypes.c_double * numVocalTractParams;
    tractParamNames = ctypes.create_string_buffer(numVocalTractParams * 32);
    tractParamMin = c_numTractParam_ptr(0);
    tractParamMax = c_numTractParam_ptr(0);
    tractParamNeutral = c_numTractParam_ptr(0);
    lib.vtlGetTractParamInfo(tractParamNames, tractParamMin, tractParamMax, tractParamNeutral);

    # get glottis info
    c_numGlottisParam_ptr = ctypes.c_double * numGlottisParams;
    glottisParamNames = ctypes.create_string_buffer(numGlottisParams * 32);
    glottisParamMin = c_numGlottisParam_ptr(0);
    glottisParamMax = c_numGlottisParam_ptr(0);
    glottisParamNeutral = c_numGlottisParam_ptr(0);
    lib.vtlGetGlottisParamInfo(glottisParamNames, glottisParamMin, glottisParamMax, glottisParamNeutral);

    numFrames = round(duration_s * frameRate_Hz);
    # 2000 samples more in the audio signal for safety.
    c_audio_ptr = ctypes.c_double * int(duration_s * audioSamplingRate + 2000)
    audio = c_audio_ptr(0);
    numAudioSamples = c_int_ptr(0);

    # Init the arrays.
    tractParamFrame = [0] * numVocalTractParams;
    glottisParamFrame = [0] * numGlottisParams;
    c_tubeAreas_ptr = ctypes.c_double * int(numFrames * numTubeSections);
    tubeAreas = c_tubeAreas_ptr(0);

    c_tractSequence_ptr = ctypes.c_double * int(numFrames * numVocalTractParams)
    c_glottisSequence_ptr = ctypes.c_double * int(numFrames * numGlottisParams)

    tractParams_ptr = c_tractSequence_ptr(*tractParams)
    glottisParams_ptr = c_glottisSequence_ptr(*glottisParams)

    lib.vtlSynthBlock(tractParams_ptr, glottisParams_ptr, tubeAreas, ctypes.c_int(int(numFrames)), ctypes.c_double(frameRate_Hz), audio, numAudioSamples);
    copiedAudio = numpy.zeros(shape=(len(audio),1), dtype=numpy.float)
    for i in range(0, len(audio)):
        copiedAudio[i] = audio[i]

    # normalize audio and scale to int16 range
    scaledAudio = numpy.int16(copiedAudio/(numpy.max(numpy.abs(copiedAudio))+1e-10) * 32767)
    # write wave file
    #scipy.io.wavfile.write(wavFileName, 22050, scaledAudio)
    lib.vtlClose()
    return scaledAudio/32767.0
