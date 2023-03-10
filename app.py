# #!/usr/bin/env python

# import torch
# # import torchaudio
# from transformers import AutoModelForCTC, Wav2Vec2Processor
# from flask import Flask,  request
# from flask_cors import CORS
# # from word2numberi18n import w2n
# # import pyaudio
# # import wave
# import os
# import os.path
# # import numpy as np
# # import matplotlib.pyplot as plt


# # My Libraries
# # from img import Waveform
# # from mfcc import saveMFCC
# # from preprocessing import removeSilence

# # import sounddevice as sd
# # from scipy.io.wavfile import write

# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 16000
# # CHUNK = 1024
# # RECORD_SECONDS = 30

# # fs = 16000  # Sample rate
# # seconds = 3  # Duration of recording

# # WAVE_OUTPUT_FILENAME = "./files/dataset/file.wav"

# # audio = pyaudio.PyAudio()
# # print(audio, "audio !!!")
# # print(FORMAT, CHANNELS, RATE, CHUNK, "CHUNK")

# # os.environ['w2n.lang'] = 'fr'
# UPLOAD_FOLDER = './files'

# app = Flask(__name__)
# CORS(app)
# # app.config['CORS_HEADERS'] = 'Content-Type'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device, "Device")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # torchaudio.set_audio_backend("soundfile")

# """
#     Loading Model
# """
# try:
#     # 2200 hours of French
#     print("Loading Model from Hugging face...")
#     model = AutoModelForCTC.from_pretrained(
#         "bofenghuang/asr-wav2vec2-ctc-french").to(device)
#     processor = Wav2Vec2Processor.from_pretrained(
#         "bofenghuang/asr-wav2vec2-ctc-french")
#     model_sample_rate = processor.feature_extractor.sampling_rate
#     print("Loaded Model ! ")
# except BaseException as e:
#     print('Failed to do something: ' + str(e))


# @app.route('/test',  methods=['GET'])
# def test():
#     return "Test"


# dict = {
#     "ab": "A B",
#     "ka": "K",
#     "rs": "R S",
#     "vw": "V W",
#     "cu": "Q",
#     "un": "1"
# }


# # @app.route('/nb-word',  methods=['POST'])
# # def nbword():
# #     data = request.get_json()
# #     word = str(data["word"])

# #     print(word, "word")

# #     path = "./files/train/{}".format(word.strip())
# #     isExist = os.path.exists(path)
# #     if not isExist:
# #         return 0
# #     nb = (len([entry for entry in os.listdir(path)
# #                if os.path.isfile(os.path.join(path, entry))]))

# #     print(nb, "nb")
# #     return {'nb': str(nb)}


# # @app.route('/translation',  methods=['POST'])
# # def translation():
# #     data = request.get_json()
# #     translation = str(data["translation"])

# #     print(translation, "translation recorder")

# #     path = "./files/train/translation"

# #     # Check whether the specified path exists or not
# #     isExist = os.path.exists(path)
# #     if not isExist:
# #         # Create a new directory because it does not exist
# #         os.makedirs(path)

# #     # nb = ([name for name in os.listdir(path)
# #     #       if os.path.isfile(os.path.join(path, name))])

# #     nb = (len([entry for entry in os.listdir(path)
# #                if os.path.isfile(os.path.join(path, entry))]))

# #     print(nb, "nb")

# #     WAVE_OUTPUT_FILENAME = "./files/train/translation/{}.wav".format(nb)

# #     # start Recording
# #     try:
# #         print("recording...")

# #         # start Recording
# #         stream = audio.open(format=FORMAT, channels=CHANNELS,
# #                             rate=RATE, input=True,
# #                             frames_per_buffer=CHUNK)
# #         print("recording...")
# #         frames = []

# #         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
# #             data = stream.read(CHUNK)
# #             frames.append(data)
# #         print("finished recording")

# #         # stop Recording
# #         stream.stop_stream()
# #         stream.close()
# #         audio.terminate()

# #         waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# #         waveFile.setnchannels(CHANNELS)
# #         waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# #         waveFile.setframerate(RATE)
# #         waveFile.writeframes(b''.join(frames))
# #         waveFile.close()

# #         # myrecording = sd.rec(int(seconds * fs), samplerate=fs,
# #         #                      channels=1,  dtype='int16')
# #         # sd.wait()
# #         # write(WAVE_OUTPUT_FILENAME, fs, myrecording)

# #         waveform = Waveform(WAVE_OUTPUT_FILENAME)
# #         waveform.save()
# #         # removeSilence(WAVE_OUTPUT_FILENAME)

# #         waveform, sample_rate = torchaudio.load(WAVE_OUTPUT_FILENAME)
# #         print("Sample Rate", sample_rate)
# #         waveform = waveform.squeeze(axis=0)  # mono
# #         # resample for comparate models
# #         # if sample_rate != model_sample_rate:
# #         #     resampler = torchaudio.transforms.Resample(
# #         #         sample_rate, model_sample_rate)
# #         #     waveform = resampler(waveform)

# #         # normalize
# #         input_dict = processor(
# #             waveform, sampling_rate=model_sample_rate, return_tensors="pt")

# #         # input_dict = processor(
# #         #     waveform, sampling_rate=model_sample_rate, return_tensors="pt")

# #         with torch.inference_mode():
# #             logits = model(input_dict.input_values.to(device)).logits

# #         # predicted_sentence = processor.batch_decode(
# #         #     logits.cpu().numpy()).text[0]

# #         predicted_ids = torch.argmax(logits, dim=-1)
# #         predicted_sentence = processor.batch_decode(predicted_ids)[0]

# #         # with processor.as_target_processor():
# #         #     labels = processor(
# #         #         predicted_sentence, return_tensors="tf").input_ids
# #         # print(labels, labels)

# #         predicted_sentence = predicted_sentence.lower()
# #         print(predicted_sentence, "prediction")

# #         WAVE_OUTPUT_IMG = "{}.png".format(nb)
# #         WAVE_OUTPUT_FILE = "{}.wav".format(nb)

# #         return {"sentence": predicted_sentence, "file": WAVE_OUTPUT_FILE, "img": WAVE_OUTPUT_IMG}
# #     except Exception as e:
# #         print("*****************")
# #         print(str(e))
# #         print("*****************")
# #         return "No"


# # @app.route('/record-word',  methods=['POST'])
# # def recordword():
# #     data = request.get_json()
# #     word = str(data["word"])

# #     print(word, "word recorder")

# #     path = "./files/train/{}".format(word.strip())

# #     # Check whether the specified path exists or not
# #     isExist = os.path.exists(path)
# #     if not isExist:
# #         # Create a new directory because it does not exist
# #         os.makedirs(path)

# #     # nb = ([name for name in os.listdir(path)
# #     #       if os.path.isfile(os.path.join(path, name))])

# #     nb = (len([entry for entry in os.listdir(path)
# #                if os.path.isfile(os.path.join(path, entry))]))

# #     print(nb, "nb")

# #     WAVE_OUTPUT_FILENAME = "./files/train/{}/{}.wav".format(word, nb)

# #     # start Recording
# #     try:
# #         print("recording...")

# #         # start Recording
# #         stream = audio.open(format=FORMAT, channels=CHANNELS,
# #                             rate=RATE, input=True,
# #                             frames_per_buffer=CHUNK)
# #         print("recording...")
# #         frames = []

# #         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
# #             data = stream.read(CHUNK)
# #             frames.append(data)
# #         print("finished recording")

# #         # stop Recording
# #         stream.stop_stream()
# #         stream.close()
# #         audio.terminate()

# #         waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# #         waveFile.setnchannels(CHANNELS)
# #         waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# #         waveFile.setframerate(RATE)
# #         waveFile.writeframes(b''.join(frames))
# #         waveFile.close()

# #         # myrecording = sd.rec(int(seconds * fs), samplerate=fs,
# #         #                      channels=1,  dtype='int16')
# #         # sd.wait()
# #         # write(WAVE_OUTPUT_FILENAME, fs, myrecording)

# #         waveform = Waveform(WAVE_OUTPUT_FILENAME)
# #         waveform.save()
# #         # removeSilence(WAVE_OUTPUT_FILENAME)

# #         waveform, sample_rate = torchaudio.load(WAVE_OUTPUT_FILENAME)
# #         print("Sample Rate", sample_rate)
# #         waveform = waveform.squeeze(axis=0)  # mono
# #         # resample for comparate models
# #         # if sample_rate != model_sample_rate:
# #         #     resampler = torchaudio.transforms.Resample(
# #         #         sample_rate, model_sample_rate)
# #         #     waveform = resampler(waveform)

# #         # normalize
# #         input_dict = processor(
# #             waveform, sampling_rate=model_sample_rate, return_tensors="pt")

# #         # input_dict = processor(
# #         #     waveform, sampling_rate=model_sample_rate, return_tensors="pt")

# #         with torch.inference_mode():
# #             logits = model(input_dict.input_values.to(device)).logits

# #         # predicted_sentence = processor.batch_decode(
# #         #     logits.cpu().numpy()).text[0]

# #         predicted_ids = torch.argmax(logits, dim=-1)
# #         predicted_sentence = processor.batch_decode(predicted_ids)[0]

# #         # with processor.as_target_processor():
# #         #     labels = processor(
# #         #         predicted_sentence, return_tensors="tf").input_ids
# #         # print(labels, labels)

# #         predicted_sentence = predicted_sentence.lower()
# #         print(predicted_sentence, "prediction")

# #         ph = predicted_sentence.replace("no du lot", "")
# #         ph = predicted_sentence.replace("et", "")
# #         ph = predicted_sentence.replace("le", "")
# #         ph = predicted_sentence.replace("du", "")

# #         WAVE_OUTPUT_IMG = "{}/{}.png".format(word, nb)
# #         WAVE_OUTPUT_FILE = "{}/{}.wav".format(word, nb)
# #         WAVE_OUTPUT_IMG_MFCC = "{}/{}-mfcc.png".format(word, nb)

# #         saveMFCC(WAVE_OUTPUT_FILE, WAVE_OUTPUT_IMG_MFCC)

# #         return {"sentence": ph, "file": WAVE_OUTPUT_FILE, "img": WAVE_OUTPUT_IMG, "mfcc": WAVE_OUTPUT_IMG_MFCC}
# #     except Exception as e:
# #         print("*****************")
# #         print(str(e))
# #         print("*****************")
# #         return "No"


# # @app.route('/record',  methods=['POST'])
# # def record():

# #     audio = pyaudio.PyAudio()
# #     # start Recording
# #     stream = audio.open(format=FORMAT, channels=CHANNELS,
# #                         rate=RATE, input=True,
# #                         frames_per_buffer=CHUNK)
# #     print("recording...")
# #     frames = []

# #     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
# #         data = stream.read(CHUNK)
# #         frames.append(data)
# #     print("finished recording")

# #     # stop Recording
# #     stream.stop_stream()
# #     stream.close()
# #     audio.terminate()

# #     waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# #     waveFile.setnchannels(CHANNELS)
# #     waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# #     waveFile.setframerate(RATE)
# #     waveFile.writeframes(b''.join(frames))
# #     waveFile.close()

# #     return "Ok"


# # @app.route('/',  methods=['POST'])
# # def hello_world():
# #     # file = request.files['file']
# #     # print("file", file)
# #     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'file.wav'))
# #     wav_path = "./files/s.wav"  # path to your audio file
# #     waveform, sample_rate = torchaudio.load(wav_path)
# #     waveform = waveform.squeeze(axis=0)  # mono
# #     # resample for comparate models
# #     # if sample_rate != model_sample_rate:
# #     #     resampler = torchaudio.transforms.Resample(
# #     #         sample_rate, model_sample_rate)
# #     #     waveform = resampler(waveform)

# #     # normalize
# #     input_dict = processor(
# #         waveform, sampling_rate=model_sample_rate, return_tensors="pt")

# #     with torch.inference_mode():
# #         logits = model(input_dict.input_values.to(device)).logits

# #     # predicted_sentence = processor.batch_decode(
# #     #     logits.cpu().numpy()).text[0]

# #     predicted_ids = torch.argmax(logits, dim=-1)
# #     predicted_sentence = processor.batch_decode(predicted_ids)[0]

# #     sentence = ""
# #     predicted_sentence = predicted_sentence.lower()
# #     print(predicted_sentence, "prediction")

# #     # ph = predicted_sentence.replace("no du lot", "")
# #     # ph = predicted_sentence.replace("et", "")
# #     # ph = predicted_sentence.replace("le", "")
# #     # ph = predicted_sentence.replace("du", "")

# #     # for word in ph.split():
# #     #     if len(word) > 2:
# #     #         try:
# #     #             alpha = w2n.word_to_num(word)
# #     #             sentence += " " + str(alpha)
# #     #         except Exception as e:
# #     #             print("Nan")
# #     #             # print(str(e))
# #     #     else:
# #     #         print("word", word)
# #     #         if word in dict.keys():
# #     #             word = dict[word]
# #     #         elif len(word) == 2:
# #     #             word = word[0] + " " + word[1]
# #     #         sentence += " " + word.upper()
# #     return predicted_sentence


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
