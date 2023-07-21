from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model=load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl','rb'))

def predict(model, tokenizer, text):
  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence)) #return indices of maximum value
  predicted_word = ""

  for key, value in tokenizer.word_index.items(): #iterate over each item in the dictionary of tokenizer file
    if value == preds :
      predicted_word = key
      break

  print(predicted_word)
  return predicted_word

def getres(e):
    return predict(model, tokenizer, e["data"])

@app.route('/predictword', methods=['POST'])
def predictword():
    # Return a JSON response
    req = request.get_json()
    res = {"data": getres(req)}
    return jsonify({"data": res})


if __name__ == '__main__':
    app.run(debug=True)
