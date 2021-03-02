import json
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from dataset import SpeechDataset, AudioDataLoader, LABELS
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from config import args

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(args, train_loader, model, optimizer, criterion):
    model.train()
    losses = AverageMeter()

    t = tqdm(train_loader)
    for i, data in enumerate(t):

        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        
        inputs = inputs.to(args.device)

        out, output_sizes = model(inputs, input_sizes)
        out = out.transpose(0, 1) # TxNxH

        float_out = out.float() # ensure float32 for loss
        loss = criterion(float_out, targets, output_sizes, target_sizes).to(args.device)
        loss = loss / inputs.size(0) # average the loss by minibatch
        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss_value, inputs.size(0))

    return losses.avg


def test_fn(args, test_loader, model, decoder, target_decoder):
    
    model.eval()

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []

    with torch.no_grad():
        t = tqdm(test_loader)
        for i, data in enumerate(t):

            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(args.device)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out, output_sizes = model(inputs, input_sizes)

            decoded_output, _ = decoder.decode(out, output_sizes)
            target_strings = target_decoder.convert_to_strings(split_targets)

            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))

            for x in range(len(target_strings)):
                transcipt, reference = decoded_output[x][0], target_strings[x][0]
                wer_inst = decoder.wer(transcipt, reference)
                cer_inst = decoder.cer(transcipt, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference.replace(' ',''))

        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars
        return wer * 100, cer * 100, output_data


def main(args):


    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = LABELS
    
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window)
    
    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.rnn_hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)
    
    
    # Data setup
    evaluation_decoder = GreedyDecoder(model.labels) # Decoder used for validation

    train_df = pd.read_csv(args.train_path)
    train_dataset = SpeechDataset(args=args, df=train_df)

    test_df = pd.read_csv(args.test_path)
    test_dataset = SpeechDataset(args=args, df=test_df)

    train_loader = AudioDataLoader(dataset=train_dataset,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size)
    
    test_loader = AudioDataLoader(dataset=test_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    model = model.to(args.device)
    parameters = model.parameters()

    optimizer = torch.optim.AdomW(parameters,
                                  lr=args.learning_rate,
                                  betas=args.betas,
                                  eps=args.eps,
                                  weight_decay=args.weight_decay)
    
    criterion = CTCLoss()

    best_score = 99999

    for epoch in range(args.epochs):
        train_loss = train_fn(args, train_loader, model, optimizer, criterion)
        wer, cer, output_data = test_fn(args=args,
                                        model=model,
                                        decoder=evaluation_decoder,
                                        target_decoder=evaluation_decoder)
        
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
        
        if (wer+cer)/2 < best_score:
            print("**** Model Improved !!!! Saving Model")
            torch.save(model.state_dict(), f"best_model.bin")
            best_score = (wer+cer)/2



    
