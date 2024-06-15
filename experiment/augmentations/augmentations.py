import numpy as np


def dropout_aug_eeg(data, p=0.3):
    for eeg in data:
        eeg = eeg.T
        for i in range(64):
            if i % 2 == 0 and random.random() < p:
                eeg[i] = torch.zeros(640)
        eeg = eeg.T
    return data

def switch_aug_eeg(data, p=0.3):
    for eeg in data:
        eeg = eeg.T
        for i in range(64):
            if i % 2 == 0 and random.random() < p:
                eeg[i], eeg[i+1] = eeg[i+1], eeg[i]
        eeg = eeg.T
    return data

def zero_aug_eeg(data, labels, p=0.1):
    for eeg, out in zip(data, labels):
        for i in range(0, 640, 20):
            if random.random() < p:
                for j in range(i, i+20):
                    eeg[j] = torch.zeros(64)
                    out[j] = 0
    return data, labels

def switch_signal_aug_eeg(data, labels, p=0.1):
    for eeg, out in zip(data, labels):
        if random.random() < p:
            begin = eeg[:320].clone().detach()
            end = eeg[320:].clone().detach()
            eeg[320:] = begin
            eeg[:320] = end
            
            begin = out[:320].clone().detach()
            end = out[320:].clone().detach()
            out[320:] = begin
            out[:320] = end
    return data, labels

def add_noise_aug_eeg(data):
    for eeg in data:
        eeg = eeg.T
        for i in range(64):
            eeg[i] += np.random.normal(0,0.5,640)
        eeg = eeg.T
    return data