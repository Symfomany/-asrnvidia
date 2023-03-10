from fastapi import FastAPI
from transformers import pipeline
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import torchaudio


from fastapi.responses import FileResponse
app = FastAPI()

# pipe_flan = pipeline("text2text-generation", model="google/flan-t5-small")


# @app.get("/infer_t5")
# def t5(input):
#     output = pipe_flan(input)
#     return {"output": output[0]["generated_text"]}
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


# app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
def index():
    wav_path = "./files/voirie.wav"  # path to your audio file
    waveform, sample_rate = torchaudio.load(wav_path)
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

    sentence = ""
    predicted_sentence = predicted_sentence.lower()
    print(predicted_sentence, "prediction")
    return {"message": predicted_sentence}
