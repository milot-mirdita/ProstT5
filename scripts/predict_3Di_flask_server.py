#!/usr/bin/env python3
from predict_3Di_encoderOnly import toCPU, load_predictor, get_T5_model
from flask import Flask, jsonify
import numpy as np
import torch

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
predictor = load_predictor(device)
model, vocab = get_T5_model(device, "../weights/")
prefix = "<AA2fold>"
half_precision = True

if half_precision:
    model = model.half()
    predictor = predictor.half()
    print("Using models in half-precision.")
else:
    model = model.full()
    predictor = predictor.full()
    print("Using models in full-precision.")

def prediction_to_string(prediction):
    ss_mapping = {
        0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S", 16: "T", 17: "V", 18: "W", 19: "Y"
    }
    return "".join([ss_mapping[int(yhat)] for yhat in prediction])

@app.route('/predict/<string:seq>', methods=['GET'])
def predict(seq):
    seq = seq.replace('U','X').replace('Z','X').replace('O','X')
    seq_len = len(seq)
    seq = prefix + ' ' + ' '.join(list(seq))
    token_encoding = vocab.encode_plus(
        seq,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        embedding_repr = model(
            token_encoding.input_ids,
            attention_mask=token_encoding.attention_mask
        )
    # mask out last token
    token_encoding.attention_mask[0,seq_len+1] = 0
    # extract last hidden states (=embeddings)
    residue_embedding = embedding_repr.last_hidden_state.detach()
    # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
    residue_embedding = residue_embedding*token_encoding.attention_mask.unsqueeze(dim=-1)
    # slice off embedding of special token prepended before to each sequence
    residue_embedding = residue_embedding[:,1:]
    prediction = predictor(residue_embedding)
    prediction = toCPU(torch.max( prediction, dim=1, keepdim=True )[1] ).astype(np.byte)
    prediction = prediction[0,:,0:seq_len].squeeze()

    return jsonify(prediction_to_string(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
