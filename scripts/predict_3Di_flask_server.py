#!/usr/bin/env python3
from predict_3Di_encoderOnly import toCPU, load_predictor, get_T5_model
from flask import Flask, jsonify
import numpy as np
import torch
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

torch_device = os.getenv("TORCH_DEVICE", "cuda:0")
model_path = os.getenv("MODEL_PATH", "./weights/")
half_precision = os.getenv("HALF_PRECISION", "1").lower() in ["true", "1", "yes", "y", "t"]
max_prediction_length = int(os.getenv("MAX_PREDICTION_LENGTH", "1200"))

device = torch.device(
    torch_device if torch.cuda.is_available() and torch_device != "cpu" else "cpu"
)
print("Using device: {}".format(device))
predictor = load_predictor(device)
model, vocab = get_T5_model(device, model_path)
prefix = "<AA2fold>"

if half_precision:
    model = model.half()
    predictor = predictor.half()
    print("Using models in half-precision")
else:
    model = model.full()
    predictor = predictor.full()
    print("Using models in full-precision")

if device.type == "cuda":
    torch.cuda.empty_cache()

app = Flask(__name__)


def prediction_to_string(prediction):
    ss_mapping = {
        0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "K", 9: "L",
        10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S", 16: "T", 17: "V", 18: "W", 19: "Y",
    }
    return "".join([ss_mapping[int(yhat)] for yhat in prediction])


@app.route("/predict/<string:seq>", methods=["GET"])
def predict(seq):
    seq_len = len(seq)
    if seq_len > max_prediction_length:
        return jsonify({"error": "Sequence length exceeds maximum length"}), 400

    seq = prefix + " " + " ".join(["X" if aa not in "ACDEFGHIKLMNPQRSTVWY" else aa for aa in seq])
    token_encoding = vocab.encode_plus(
        seq, add_special_tokens=True, padding="longest", return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embedding_repr = model(
            token_encoding.input_ids, attention_mask=token_encoding.attention_mask
        )
    # mask out last token
    token_encoding.attention_mask[0, seq_len + 1] = 0
    # extract last hidden states (=embeddings)
    residue_embedding = embedding_repr.last_hidden_state.detach()
    # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
    residue_embedding = residue_embedding * token_encoding.attention_mask.unsqueeze(dim=-1)
    # slice off embedding of special token prepended before to each sequence
    residue_embedding = residue_embedding[:, 1:]
    prediction = predictor(residue_embedding)
    prediction = toCPU(torch.max(prediction, dim=1, keepdim=True)[1]).astype(np.byte)
    prediction = prediction[0, :, 0:seq_len].squeeze()

    json = jsonify(prediction_to_string(prediction))

    if device.type == "cuda":
        del token_encoding
        del embedding_repr
        del residue_embedding
        del prediction
        torch.cuda.empty_cache()

    return json


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)
