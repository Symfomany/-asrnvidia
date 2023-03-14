# import json
# import requests

# API_TOKEN = "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"
# headers = {"Authorization": f"Bearer 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"}
# API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"


# def query(payload):
#     data = json.dumps(payload)
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))


# data = query({"inputs": "The answer to the universe is [MASK]."})
# print(data)

import json
import requests

# headers = {"Authorization": f"Bearer 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"}
# API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-xls-r-1b-french"


# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))

import json
import requests

# headers = {"Authorization": f"Bearer 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"}
# API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"


# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))


# data = query("./files/voirie.wav")

from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import torch
import torchaudio
from transformers import AutoModelForCTC, Wav2Vec2Processor

from pathlib import Path

from fastapi.responses import FileResponse
app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "Device")

try:
    # 2200 hours of French
    print("Loading Model from Hugging face...")
    model = AutoModelForCTC.from_pretrained(
        "bofenghuang/asr-wav2vec2-ctc-french").to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "bofenghuang/asr-wav2vec2-ctc-french")
    model_sample_rate = processor.feature_extractor.sampling_rate
    print("Loaded Model ! ")
except BaseException as e:
    print('Failed to do something: ' + str(e))

# pipe = pipeline("automatic-speech-recognition",
#                 "facebook/wav2vec2-large-xlsr-53-french")


@app.get("/test")
async def index():
    return {"message": "ok"}


@app.post("/")
async def main(file: bytes = File(...)):
    # print(file)
    # try:
    #     contents = file.file.read()
    #     with open(file.filename, 'wb') as f:
    #         f.write(contents)
    # except Exception:
    #     return {"message": "There was an error uploading the file"}
    # finally:
    #     file.file.close()

    # return {"message": f"Successfully uploaded {file.filename}"}
    wav_path = file
    Path("./files/voirie.wav").write_bytes(wav_path)
    # waveform, sample_rate = torchaudio.load(chemin)
    # waveform = waveform.squeeze(axis=0)  #

    # with open("./files/voirie.wav", "rb") as f:
    #     data = f.read()

    waveform, sample_rate = torchaudio.load("./files/voirie.wav")
    waveform = waveform.squeeze(axis=0)  #

    # normalize
    input_dict = processor(
        waveform, sampling_rate=model_sample_rate, return_tensors="pt")

    with torch.inference_mode():
        logits = model(input_dict.input_values.to(device)).logits

    # predicted_sentence = processor.batch_decode(
    #     logits.cpu().numpy()).text[0]

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)[0]

    # res = pipe("./files/voirie.wav")

    return {"message": predicted_sentence}


# Inference API loads models on demand,
# so if it’s your first time using it in a while it will load the model first,
# then you can try to send the request again in couple of seconds.


print('End ...')
# res = query("files/voirie.wav")
# print(res)
# print('... World!')
