import whisper
from jiwer import wer
import torch
import nemo.collections.asr as nemo_asr

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

@torch.no_grad()

def generate_wer(file, reference):
    model = whisper.load_model("tiny")
    result = model.transcribe(file)
    hypothesis = result["text"]

    error = wer(reference, hypothesis)

    return error

def check_similarity(path2audio_file1, path2audio_file2):

    embs1 = speaker_model.get_embedding(path2audio_file1).squeeze()
    embs2 = speaker_model.get_embedding(path2audio_file2).squeeze()

    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)

    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2

    return similarity_score.tolist()

