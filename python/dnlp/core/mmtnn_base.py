# -*- coding: UTF-8 -*-
import pickle
from dnlp.config.sequence_labeling_config import MMTNNConfig


class MMTNNBase(object):
  def __init__(self, *, config: MMTNNConfig, data_path: str = '', model_path:str='',mode: str = 'train'):
    self.data_path = data_path
    self.mode = mode
    self.config_suffix = '.config.pickle'
    if mode == 'train':
      self.dictionary,self.tags,self.sentences,self.labels = self.__load_data()
    else:
      self.config_path = model_path+self.config_suffix
      self.dictionary,self.tags = self.__load_config()
    self.dict_size = len(self.dictionary)
    self.tags_count = len(self.tags)
    # 初始化超参数
    self.skip_left = config.skip_left
    self.skip_right = config.skip_right
    self.character_embed_size = config.character_embed_size
    self.tag_embed_size = config.tag_embed_size
    self.hidden_unit = config.hidden_unit
    self.learning_rate = config.learning_rate
    self.lam = config.lam
    self.dropout_rate = config.dropout_rate
    self.batch_length = config.batch_length
    self.batch_size = config.batch_size
    self.concat_character_embed_size =(self.skip_right + self.skip_left + 1) * self.character_embed_size
    self.concat_embed_size = self.concat_character_embed_size + self.tag_embed_size

  def __load_data(self):
    with open(self.data_path, 'rb') as f:
      data = pickle.load(f)
      return data['dictionary'], data['tags'], data['characters'], data['labels']
  def __load_config(self):
    with open(self.config_path, 'rb') as cf:
      config = pickle.load(cf)
      return config['dictionary'], config['tags']
  def generate_batch(self):
    pass
