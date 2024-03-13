from anonymizer import Anonymizer
import soundfile as sf
import os

# checkpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')
checkpoint_dir = None
device = 'cuda:1'

anonymizer = Anonymizer(checkpoint_dir, voice_dirs=['suno_voices/v2']).to(device)
anon_wav = anonymizer('84-121123-0006_24k.wav', target_voice_id='it_speaker_0')
print(f'Output audio of shape {anon_wav.shape}')

sf.write('test_out.wav', anon_wav, 24000)