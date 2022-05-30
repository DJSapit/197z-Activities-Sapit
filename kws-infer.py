import torch
import requests
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators
from torchvision.transforms import ToTensor
from einops import rearrange
import PySimpleGUI as sg

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    #parser.add_argument("--checkpoint", type=str, default="kws_model.pt")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--button", default=False, action='store_true')
    parser.add_argument("--playback", default=False, action='store_true')
    args = parser.parse_args()
    return args


# main routine
if __name__ == "__main__":
    checkpoint = "kws_model.pt"
    if not os.path.exists(checkpoint):
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        fname = os.path.basename(checkpoint)
        url = f'https://github.com/DJSapit/197z-Activities-Sapit/releases/download/v0.1.1-alpha/kws_model.pt'
        print(f'downloading pretrained model from {url}')
        r = requests.get(url, allow_redirects=True)
        with open(checkpoint, 'wb') as file:
            file.write(r.content)
        if os.path.exists(checkpoint):
            print("pretrained model downloaded successfully")
        else:
            raise Exception("pretrained model download failed")

    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    args = get_args()

    print("Loading model checkpoint: ", checkpoint)
    scripted_module = torch.jit.load(checkpoint)

    sample_rate = 16000
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    sg.theme('DarkAmber')

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=2.0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),]
    ]

    if args.button:
        window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True, return_keyboard_events=True, use_default_focus=False).Finalize()
    else:
        window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    while True:
        if args.button:
            event, values = window.read()
        else:
            event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        if args.button:
            print("event:",event)
            if event in ("Escape:27","q"):
                break
            if event is not None:
                raw_waveform = sd.rec(sample_rate).squeeze()
        else:
            raw_waveform = sd.rec(sample_rate).squeeze()
        sd.wait()
        if raw_waveform.max() > 1.0:
            print("continuing")
            continue

        waveform = torch.from_numpy(raw_waveform).unsqueeze(0)
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = rearrange(mel, 'c (p1 h) (p2 w) -> (p1 p2) (c h w)', p1=1, p2=16)
        mel = mel.unsqueeze(0)
        pred = scripted_module(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        print(f"predmax: {max_prob}")
        pred = torch.argmax(pred, dim=1)
        human_label = f"{idx_to_class[pred.item()]}"
        if max_prob > args.threshold:
            window['-OUTPUT-'].update(human_label)
            print(human_label)
            if args.playback:
                sd.play(raw_waveform,samplerate=sample_rate)
                sd.wait()
            time.sleep(1)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
        else:
            window['-OUTPUT-'].update("...")

    window.close()

            
