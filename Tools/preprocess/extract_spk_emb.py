from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
from glob import glob
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
warnings.filterwarnings("ignore")

def extract_res_emb(wav_path, save_path, encoder):
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath) 
    embed = encoder.embed_utterance(wav)
    os.makedirs(save_path.replace(os.path.basename(save_path),''),exist_ok=True)
    np.save(save_path, embed)
    
    
def extract_lrs3_dataset():
    wav_paths = glob(os.path.join("/data0/yfliu/lrs3/audio/test", "*/*.wav"))
    save_paths = [i.replace("pwg_vqmivc/test/rese+emb") for i in wav_paths]
    # Looks like they can support GPU here.
    encoder = VoiceEncoder()
    Parallel(n_jobs=6)(delayed(extract_res_emb)(wav_paths[i], save_paths[i], encoder) for i in tqdm(range(len(wav_paths))))
    

if __name__ == "__main__":
    extract_lrs3_dataset()