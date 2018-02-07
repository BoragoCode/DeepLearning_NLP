# -*- coding: UTF-8 -*-
import pandas as pd
# from dnlp.crf.crf import ENTITY_CATEGORY

def read_lines_in_file(filename, delimiter='\n', encoding='utf-8', filter_chars=('', '\n', '\r')):
  with open(filename, encoding=encoding) as f:
    return [l for l in f.read().split(delimiter) if l not in filter_chars]


def load_data_in_conll(src_file, column_count=2, delimiter=' ', return_pd=False):
  sentences = []
  labels = []
  sentence_entries = read_lines_in_file(src_file, delimiter='\n\n')

  for sentence in sentence_entries:
    word_entries = pd.DataFrame([l.split(delimiter) for l in sentence.split('\n')])
    words = word_entries[0].tolist()
    ept = [i for i, w in enumerate(words) if not w]
    if ept:
      print(ept)
    columns = word_entries.loc[:, 1:column_count].values.T.tolist()
    if not labels:
      labels = [[c] for c in columns]
    else:
      labels = [l + [c] for l, c in zip(labels, columns)]
    sentences.append(words)
  if return_pd:
    sentences = pd.Series(sentences)
    labels = [pd.Series(l) for l in labels]
  return [sentences] + labels


def labels2entity_texts(labels, sentence,types=None):
  # print(labels)
  if len(sentence) != len(labels):
    raise Exception('length error')

  spans = {}
  if types:
    span_type = {}
  start = 0

  for i, label in enumerate(labels):
    if label[0] ==  'B':
      start = i
      spans[start] = start
      if types:
        span_type[start] = types[label[2:]]
    elif label[0] == 'I':
      spans[start] = i

  entities = []
  for span in spans:
    entities.append((sentence[span:spans[span] + 1],span_type[span]))

  return entities
