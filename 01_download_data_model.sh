#!/bin/bash

set -e

source env.sh

# librispeech_corpus=PATH_TO_Librispeech
# iemocap_corpus=PATH_TO_IEMOCAP

# Download and unpack libri_dev and libri_test (dev-clean and test-clean)
for data_set in dev-clean test-clean; do
    dir="data/libri_${data_set}"
    if [ ! -f $dir/wav.scp ]; then
        [ -d $dir ] && rm -r $dir
        if [ ! -f "corpora/${data_set}.tar.gz" ]; then
            mkdir -p corpora
            cd corpora
            echo "Downloading ${data_set}.tar.gz..."
            
            # Use OpenSLR URLs for dev-clean and test-clean
            wget --no-check-certificate "https://www.openslr.org/resources/12/${data_set}.tar.gz" || { 
                echo "Failed to download ${data_set}.tar.gz. Please download manually if there are issues."; 
                cd -; 
                continue; 
            }
            cd -
        fi

        echo "  Unpacking ${data_set} data set..."
        tar -xf "corpora/${data_set}.tar.gz" -C data || exit 1
        mv "data/LibriSpeech/${data_set}" "$dir"
        [ ! -f $dir/text ] && echo "File $dir/text does not exist" && exit 1
        cut -d' ' -f1 $dir/text > $dir/text1
        cut -d' ' -f2- $dir/text | sed -r 's/,|!|\?|\./ /g' | sed -r 's/ +/ /g' | awk '{print toupper($0)}' > $dir/text2
        paste -d' ' $dir/text1 $dir/text2 > $dir/text
        rm $dir/text1 $dir/text2
    fi
done

# Check for train-clean-360 and download if missing
check=corpora/LibriSpeech/train-clean-360
if [ ! -d $check ]; then
    if [ ! -z $librispeech_corpus ]; then
        if [ -d $librispeech_corpus/train-clean-360 ]; then
            [ -d corpora/LibriSpeech ] && rm corpora/LibriSpeech
            echo "Linking '$librispeech_corpus' to 'corpora'"
            mkdir -p corpora
            ln -s $librispeech_corpus corpora
        else
            echo "librispeech_corpus is defined to '$librispeech_corpus', but '$librispeech_corpus/train-clean-360' does not exist."
            echo "Either remove the librispeech_corpus variable from the $0 script to download the dataset or modify it to the correct target."
            exit 1
        fi
    fi
fi

# Download train-clean-360 if needed
if [ ! -d $check ]; then
    echo "Downloading train-clean-360..."
    mkdir -p corpora
    cd corpora
    if [ ! -f train-clean-360.tar.gz ]; then
        wget --no-check-certificate https://www.openslr.org/resources/12/train-clean-360.tar.gz
    fi
    echo "Unpacking train-clean-360"
    tar -xzf train-clean-360.tar.gz
    cd ../
fi

# Remaining data setup and model downloads are unchanged
# ...

