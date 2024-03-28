# Training of STTTS models
If you don't want to use the provided pretrained models in the STTTS pipeline but train them yourself, e.g., 
on different data or with different configurations, you can use the train scripts given in the respective module folder.
In the following, we will explain how to use these scripts but will not go into much detail about the architecture of 
the models. For more information about them, please check the following papers:
- [Meyer, Lux, Denisov, Koch, Tilli, Vu: Speaker Anonymization with Phonetic Intermediate Representations. Interspeech, 2022](https://www.isca-archive.org/interspeech_2022/meyer22b_interspeech.pdf)
- [Meyer, Tilli, Lux, Denisov, Koch, Vu: Cascade of Phonetic Speech Recognition, Speaker Embeddings GAN and
Multispeaker Speech Synthesis for the VoicePrivacy 2022 Challenge. VPC 2022](https://www.voiceprivacychallenge.org/vp2022/docs/3___T04.pdf)
- [Meyer, Tilli, Denisov, Lux, Koch, Vu: Anonymizing Speech with Generative Adversarial Networks to Preserve Speaker Privacy. SLT, 2023](https://ieeexplore.ieee.org/abstract/document/10022601)
- [Meyer, Lux, Koch, Denisov, Tilli, Vu: Prosody Is Not Identity: A Speaker Anonymization Approach Using Prosody Cloning. ICASSP, 2023](https://ieeexplore.ieee.org/abstract/document/10096607)


## Speech Recognition Model
The ASR model in this pipeline is implemented in [ESPnet2](https://github.com/espnet/espnet). 
The training recipe can be found in [anonymization/modules/sttts/text/recognition/asr_training](../../modules/sttts/text/recognition/asr_training).
It is in the standard ESPnet recipe structure with symlinks to files and folders that are shared across recipes. 
Therefore, you need to install the ESPnet repository before you can use this recipe. To install, please follow the 
instruction on [the official website](https://espnet.github.io/espnet/installation.html).
If you use a previous installation of ESPnet on your machine, please make sure that your version is recent enough (i.e., at least version 202308).

The recipe is almost identical to the [LibriTTS recipe](https://github.com/espnet/espnet/tree/master/egs2/libritts/asr1) in the official repository.
During data preparation, LibriTTS is converted into phonetic transcriptions, and the ASR is trained to recognize equal phonetic sequences from the speech.
This is done in order to skip the text-to-phone conversion in the speech synthesis of the pipeline.

In order to run the recipe, copy the [asr_training](../../modules/sttts/text/recognition/asr_training) into the [egs2](https://github.com/espnet/espnet/tree/master/egs2) directory of your local ESPnet installation. 
The relative symlinks should automatically map to the correct files. Do not change the internal structure of the recipe, 
especially do not move the contents of the [asr](../../modules/sttts/text/recognition/asr_training/asr) folder to its parent directory.

To start the training, simply run
```angular2html
./run.sh
```
or 
```angular2html
export CUDA_VISIBLE_DEVICES=0,1; ./run.sh
```
to specify a (or several) specific GPU(s) the training should run for.

Per default, the training will be performed for LibriTTS which will be downloaded automatically. 
It will be stored to a *downloads* inside the recipe that needs to be created before execution.
If you want to change this to a different location, or already have LibriTTS installed and skip to download it again, you need to change the *LibriTTS* value in [db.sh](../../modules/sttts/text/recognition/asr_training/asr/db.sh).

# Speech Synthesis Models
The speech synthesis module consists of two models: a TTS based on FastSpeech2 and a HiFiGAN vocoder. 
These models need to be trained separately.
Both are implemented in the [IMS Toucan toolkit](https://github.com/DigitalPhonetics/IMS-Toucan) which is included in this repository because the STTTS pipeline heavily relies on it.

To start training each model, you need to run [train_tts_model.py](../../modules/sttts/tts/train_tts_model.py) and [train_vocoder_model.py](../../modules/sttts/tts/train_vocoder_model.py) in [anonymization/modules/sttts/tts](../../modules/sttts/tts).
There are a couple of hyperparameters you can specify which should be clear from the respective files.
If you want to run the training with the default parameters, you only need to specify the path to the training data and the GPU id (it will otherwise run on CPU):
```angular2html
python train_tts_model.py --train_data_path <path_to_train_data> --gpu_id <gpu_id>
```
and
```angular2html
python train_vocoder_model.py --train_data_path <path_to_train_data> --gpu_id <gpu_id>
```
`<path_to_train_data` could be for example `corpora/LibriTTS/train-clean-100` and `gpu_id` for example `0`.
In the script, the original structure of LibriTTS or LibriSpeech (the separation into speaker and session) is expected.
If you use a different dataset or structure, you need to change the `build_path_to_transcript_dict_libritts_clean()` in `train_tts_model.py` and `get_file_list_libritts` in `train_vocoder_model.py` to match your data.
Inspirations for that can be found in the [official IMS Toucan repository](https://github.com/DigitalPhonetics/IMS-Toucan/blob/ToucanTTS/Utility/path_to_transcript_dicts.py). 

# Phone Alignment for Prosody Extraction
Prosodic information in form of pitch and energy is extracted for each phone, together with the phone duration.
For this, we need the alignment between the transcribed phones and the input waveform. 
Thus, we train an aligner that provides us these alignments.
The aligner is an ASR model, but much simpler and less accurate than the one we use for the phonetic transcriptions.
Because it is more lightweight, we can finetune the model online for each utterance before generating the alignment which improves the quality of the phone alignment.
More information about this method can be found [in this paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10022433).

As for the speech synthesis model, the aligner is implemented in the IMS Toucan toolkit. 
Pretraining of the aligner model is therefore similar to the training steps for both synthesis models.
The [pretrain_aligner.py](../../modules/sttts/prosody/pretrain_aligner.py) script can be found in [anonymization/modules/sttts/prosody](../../modules/sttts/prosody) and run by:
```angular2html
python pretrain_aligner.py --train_data_path <path_to_train_data> --gpu_id <gpu_id>
```

# Speaker Embeddings GAN
The model to generate artificial speaker embeddings for anonymization is a Wasserstein Generative Adversarial Network. 
It is trained to transform noise vectors into speaker embeddings that follow the same distribution as speaker embeddings from original speakers.
In the default model, these original speaker embeddings are extracted with the GST-based style embeddings function that is trained jointly with the TTS.
However, you can also use other embeddings extractors, e.g. [the ECAPA-TDNN model of SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) by using the parameter `vec_type='ecapa'` during extraction.

The scripts for training the GAN model are located in [anonymization/modules/sttts/speaker_embeddings/anonymization/utils](../../modules/sttts/speaker_embeddings/anonymization/utils).
For the actual training, you only need one tensor file that contains all original speaker embeddings of your training data.
This file is for example generated when you call the [speaker_extraction](../../../evaluation/privacy/asv/speaker_extraction.py) during the STTTS anonymization or during the evaluation pipelines.

If you do not have this file already, please run [extract_train_embeddings.py](../../modules/sttts/speaker_embeddings/anonymization/utils/extract_train_embeddings.py) first.
You need to specify the path to the data in kaldi format (i.e., with `wav.scp`,`utt2spk`, etc.) and the specifics for the embedding extractor.
Example: 
```angular2html
python extract_train_embeddings.py --data_path ../../../../../../data/train-clean-100 --save_dir dataset --gpu_id 0 --emb_model_path ../../../../../../exp/sttts_models/tts/Embedding/embedding_function.pt --vec_type style-embed --emb_level utt
```

Once you have extracted the original speaker embeddings, you can start the GAN training in [train_gan_model.py](../../modules/sttts/speaker_embeddings/anonymization/utils/train_gan_model.py):
```angular2html
python train_gan_model.py --data_path dataset/reference_embeddings/utt-level/speaker_vectors.pt --gpu_id 0
```
