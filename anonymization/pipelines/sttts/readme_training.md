# Training of STTTS models

## ASR
- make sure you have a recent ESPNet installation
- the script will download LibriTTS automatically. If you have it already installed, change the value in db.sh to your local installation to avoid downloading it again. Otherwise, make sure that you have a downloads folder in your recipe, the corpus will be downloaded there. You can also change the path in LibriTTS to specify a different location where it should be downloaded to. 


# Speaker Embeddings GAN
- If you already have a file with the extracted speaker embeddings of your training data, you can just run train_gan_model.py
- Otherwise, you have to run generate_train_embeddings.py first