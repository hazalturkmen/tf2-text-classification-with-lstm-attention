from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
import numpy as np
import re
import string
import tensorflow as tf
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from attention_layer import Attention
import gensim
from gensim.models import Word2Vec


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def read_data(path):
    df = pd.read_excel(path)
    df["sonuc"] = df['sonuc']
    df["sonuc"] = df["sonuc"].str.replace('VAR', '1')
    df["sonuc"] = df["sonuc"].str.replace('YOK', '0')
    df["sonuc"] = df["sonuc"].str.replace('SERI DISI', '2')
    df['sonuc'] = df['sonuc'].astype(int)
    return df

df_train = read_data('/home/hazal/nlp_dataset/brain_CT/train_labeled_data300_v4.xlsx')
df_valid = read_data('/home/hazal/nlp_dataset/brain_CT/valid_labeled_data300_v4.xlsx')
df_test = read_data('/home/hazal/nlp_dataset/brain_CT/test_labeled_data300_v4.xlsx')

def tokenize (text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    return word_tokenize(nopunct)

counts = Counter()
for index, row in df_train.iterrows():
    counts.update(tokenize(row['rapor']))

print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))

vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)


def encode_sentence(text, vocab2index, N=256):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]

    return encoded
X_train = list(df_train['rapor'].apply(lambda x: encode_sentence(x,vocab2index )))
X_valid = list(df_valid['rapor'].apply(lambda x: encode_sentence(x,vocab2index )))
X_test = list(df_test['rapor'].apply(lambda x: encode_sentence(x,vocab2index )))

y_train = list(df_train['sonuc'])
y_valid = list(df_valid['sonuc'])
y_test = list(df_test['sonuc'])


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_valid = np.asarray(X_valid)
y_valid = np.asarray(y_valid)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

wvec_model = Word2Vec.load("/home/hazal/nlp_dataset/brain_CT/word2vec/word2vec.wordvectors")
vocab_size = len(words)
embedding_dim = 200

def get_emb_matrix(word_vecs, word_counts, emb_size = 200):
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32')
    W[1] = np.random.uniform(-0.25, 0.25, emb_size)
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs.wv:
            W[i] = word_vecs.wv[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx

pretrained_weights, vocab, vocab2index = get_emb_matrix(wvec_model, counts)

def rnn_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, output_dim=200, input_length=256, weights=[pretrained_weights], trainable=False))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True)))
    model.add(Attention(return_sequences=False))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.summary()
    return model

if __name__ == '__main__':
    model = rnn_model()
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=3)
    accr = model.evaluate(X_test, y_test)

