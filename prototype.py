# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:02:56 2019

@author: Joaquin
"""
#import glob
import os.path
#import random
#from multiprocessing import Process

#import Utils

import numpy as np
#from lxml import etree
#import librosa
import soundfile
import os
#import tensorflow as tf
import musdb


##########

import musdb
import os.path
database_path="C:/Users/Joaquin/Documents/GitHub/Wave-U-Net/Datasets/MUSDB18/"
mus = musdb.DB(root_dir=database_path, is_wav=False)
subsets = list()
for subset in ["train", "test"]:
    tracks = mus.load_mus_tracks(subset)
    samples = list()
    for track in tracks:
        #print(os.path.isfile(track.path))
        print(track.rate)
        
##########
import musdb
import os.path
path="C:/Users/Joaquin/Documents/GitHub/Wave-U-Net/Datasets/MUSDB18/test\Zeno - Signs.stem.mp4"



        rate = track.rate
        # Skip track if mixture is already written, assuming this track is done already
        track_path = track.path[:-4]
        mix_path = track_path + "_mix.wav"
        acc_path = track_path + "_accompaniment.wav"
        if os.path.exists(mix_path):
            print("WARNING: Skipping track " + mix_path + " since it exists already")
            # Add paths and then skip
            paths = {"mix" : mix_path, "accompaniment" : acc_path}
            paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})
    
            samples.append(paths)
            continue
    
        rate = track.rate










###########

def getMUSDB(database_path):
    mus = musdb.DB(root_dir=database_path, is_wav=False)

    subsets = list()

    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            track_path = track.path[:-4]
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix" : mix_path, "accompaniment" : acc_path}
                paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + "_" + stem + ".wav"
                audio = track.targets[stem].audio
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in stem_audio.keys() if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    return subsets


getMUSDB("Users/Joaquin/Documents/GitHub/Wave-U-Net/Datasets/MUSDB18")


