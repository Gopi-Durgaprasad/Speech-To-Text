import math
import os
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import sox
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader


LABELS = [
  "_",
  "'",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  " "
]


windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
}

def load_audio(path):
    sound, sample_rate = sf.read(path, dtype='int16')
    # TODO this should be 32768.0 to get twos-complement range.
    # TODO the difference is negligible but should be fixed for new models.
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

class SpeechDataset:
    def __init__(self, args, df):
        
        self.args = args
        self.audio_path = df.audio_path.values.tolist()
        self.transcript_path = df.txt_path.values.tolist()
        self.labels_map = dict([(LABELS[i], i) for i in range(len(LABELS))])

    def __getitem__(self, item):

        audio_path = self.audio_path[item]
        transcript_path = self.transcript_path[item]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        
        return spect, transcript

    def __len__(self):
        return len(self.audio_path)
    
    def parse_audio(self, audio_path):
        
        y = load_audio(audio_path)
        
        n_fft = int(self.args.sample_rate * self.args.window_size)
        win_length = n_fft
        hop_length = int(self.args.sample_rate * self.args.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.args.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.args.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
            
        return spect
    
    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript
    


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)
    
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[0]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
