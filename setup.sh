#! /bin/bash -e

# initial setup

echo "For the python environment setup, please refer to README.md"

# check sox command (used to manipulate speech waveform)
if [ "`which sox 2>/dev/null`" != "" ];
then
    echo sox ok;
else
    echo Please first install sox on your system;
    exit
fi

# TODO: ちゃんと動くかどうかの確認
for $dir in exp log; do
    if [ ! -d $dir ]; then
        mkdir $dir
        for $subdir in policy_gradient actor_critic dpg_pn; do
            mkdir exp/$subdir
        done
    fi
done

# download VTL
VTLPKG=https://www.vocaltractlab.de/download-vocaltractlabapi/VTL2.1_Linux.zip
if [ ! -e VTL2.1_Linux ]; then
    echo download VTL2.1_Linux.zip
    wget $VTLPKG
    unzip VTL2.1_Linux.zip -d VTL2.1_Linux
    if [ ! -e VTL2.1_Linux ]; then
	echo failed to download VTL2.1 and extract the package
    fi
fi

# TODO: 将来的には ./　のみにする
for dir in policy_gradient actor_critic dpg_pn ./; do
    echo $dir
    if [ ! -e $dir/JD2.speaker ]; then
	cp -p VTL2.1_Linux/JD2.speaker $dir
    fi
    if [ ! -e $dir/VocalTractLabApi.so ]; then
	cp -p VTL2.1_Linux/VocalTractLabApi.so $dir
    fi
done

# speech databases
if [ ! -e data/speech_commands_v0.01.tar.gz -a ! -e data/speech_commands_v0.01 ]; then
    echo "Please get Google Speech Commands Dataset (v0.01)."
    echo "The file size is 1.4GByte."
    echo "You can donload the data using the following command in data directory:"
    echo "wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
fi

if [ ! -e data/train-clean-100.tar.gz -a ! -e data/LibriSpeech ]; then
    echo "Please get Librispeech dataset (train-clean-100)."
    echo "The file size is 6.0GByte."
    echo "You can donload the data using the following command in data directory:"
    echo "wget https://www.openslr.org/resources/12/train-clean-100.tar.gz"
fi

if [ -e data/speech_commands_v0.01.tar.gz -a ! -e data/speech_commands_v0.01 ]; then
    echo "extracting speech_commands_v0.01.tar.gz"
    mkdir -p data/speech_commands_v0.01
    tar zxf data/speech_commands_v0.01.tar.gz -C data/speech_commands_v0.01
fi

if [ -e data/train-clean-100.tar.gz -a ! -e data/LibriSpeech ]; then
    echo "extracting train-clean-100.tar.gz"
    mkdir -p data/LibriSpeech
    tar zxf data/train-clean-100.tar.gz -C data/LibriSpeech
fi

if [ -e data/speech_commands_v0.01 -a ! -e data/remsil_speech_commands_v0.01 ]; then
    echo "removing silence with gsc"
    (cd data; ./remove_silence_gsc.sh)
fi

if [ -e data/LibriSpeech -a ! -e data/remsil_LibriSpeech ]; then
    echo "removing silence with libri"
    (cd data; ./remove_silence_libri.sh)
fi

# extract acoustic features
# (example)
if [ -e data/remsil_speech_commands_v0.01 -a ! -e gsc_remsil_mfcc ]; then
    echo "Run the following command to extract mfcc with remsil_speech_commands."
    echo "cd data; python remsil_gsc_preprocess.py"
    echo "Do the same thing for LibriSpeech and for other feature type."
fi

echo done
