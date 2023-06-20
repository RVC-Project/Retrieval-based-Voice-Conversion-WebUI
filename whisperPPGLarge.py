import torch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import pad_or_trim, log_mel_spectrogram


class WhisperPPGLarge(object):
    def __init__(self,vec_path = "large-v2.pt",device=None):
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def encoder(self, wav):
        audio = wav[0]
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln,]).to(self.dev)
            return ppg
