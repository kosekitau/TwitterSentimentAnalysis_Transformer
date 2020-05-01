# -*- coding: utf-8 -*-
import torch
#gpuの確認
print(torch.cuda.is_available())

!pip install janome

!pip install jaconv

import jaconv
from janome.tokenizer import Tokenizer
import re

j_t = Tokenizer()

def clean_text(text):
  text = jaconv.h2z(text)
  result = text.lower()
  result = re.sub(r'[【】]', '', result)                  # 【】の除去
  result = re.sub(r'[（）()]', '', result)                # （）の除去
  result = re.sub(r'[［］\[\]]', '', result)              # ［］の除去
  result = re.sub(r'[@＠]\w+', '', result)               # メンションの除去
  #result = re.sub('( #[\u3041-\u309F]+)+$', '', result)
  #result = re.sub(' #([\u3041-\u309F]+) ', r'\1', result)
  result = re.sub(r'[#]\w+ ', '',result)
  result = re.sub(r'[\r]', '', result)
  result = re.sub(r'　', ' ', result)                    #全角空白の除去
  return result

def tokenizer_janome(text):
  return [tok for tok in j_t.tokenize(text, wakati=True)]

def tokenize_preprocessing(text):
  text = clean_text(text)
  text = tokenizer_janome(text)
  return text

#データの読みこみ
import torchtext

max_length = 140
TEXT = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenize_preprocessing,
                            lower=True, include_lengths=True, batch_first=True,
                            fix_length=max_length, init_token='<cls>',
                            eos_token='<eos>')

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/', train='4/train.csv', validation='4/validation.csv',
    test='4/test.csv', format='csv', fields=[('text', TEXT), ('Label', LABEL)]
)

print(len(train_ds))
print(len(val_ds))
print(len(test_ds))

from torchtext.vocab import Vectors

japanese_word2vec_vectors = Vectors(
    name='drive/My Drive/tweets133_.vec')

print(japanese_word2vec_vectors.dim)
print(len(japanese_word2vec_vectors.itos))

#ボキャブラリを作成
TEXT.build_vocab(train_ds, vectors=japanese_word2vec_vectors)
print(TEXT.vocab.vectors.shape)
print(TEXT.vocab.stoi)

train_dl = torchtext.data.Iterator(train_ds, batch_size=64, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=64, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=64, train=False, sort=False)

batch = next(iter(val_dl))
print(batch.text)
print(batch.Label)

# パッケージのimport
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(12)
np.random.seed(12)

#埋め込み層
class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        #更新はしない
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

#PositonalEncoding
class PositionalEncoder(nn.Module):

    def __init__(self, d_model=300, max_seq_len=140):
        super().__init__()

        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, head_num, dropout_rate):
        super().__init__()
        """
        d_model：出力層の次元(head_bumの倍数)
        head_num：ヘッドの数
        dropout_rate
        """
        self.d_model = d_model
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        #特徴量変換
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        #出力の全結合層
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask):
        #key, query, valueを生成
        q = self.q_linear(q) # [batch_size, max_seq_len, d_model]
        k = self.q_linear(k)
        v = self.q_linear(v)

        #head_numに分割
        q = self._split_head(q) # [batch_size, head_num, max_seq_len, d_model/head_num]
        k = self._split_head(k)
        v = self._split_head(v)

        #queryとkeyの関連度の計算と、Scaled Dot-production
        weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_model)

        #maskをかける
        mask = mask.unsqueeze(1).unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)# [batch_size, head_num, max_seq_len, max_seq_len]

        #AttentionWeightを計算
        attention_weight = F.softmax(weights, dim=-1)# [batch_size, head_num, q_length, k_length]

        #AttentionWeightよりvalueから情報を引き出す
        attention_output = torch.matmul(attention_weight, v)# [batch_size, head_num, q_length, d_model/head_num]
        attention_output = self._combine_head(attention_output)
        output = self.out(attention_output)


        return output, attention_weight

    def _split_head(self, x):
        """
        x.size:[batch_size, length, d_model]
        """
        batch_size, length, d_model = x.size()
        x = x.view(batch_size, length, self.head_num, self.d_model//self.head_num) #reshape
        return x.permute(0, 2, 1, 3)

    #outputする前に分割したheadを戻す。
    def _combine_head(self, x):
        """
        x.size:[batch_size, head_num, length, d_model//head_num]
        """
        batch_size, _, length, _  = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, length, self.d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x size=[batch_size, length, d_model]
        return size=[batch_size, length, d_model]
        """
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout=0.1):
        super().__init__()

        # LayerNormalization
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # Attention
        self.attn = MultiheadAttention(d_model, head_num, dropout)
        # FFN
        self.ff = FeedForward(d_model)
        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # SelfAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)
        x2 = x + self.dropout_1(output)
        # FFN
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights

class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=5):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # <cls>の結果を用いる
        out = self.linear(x0)

        return out

# 最終的なTransformerモデルのクラス


class TransformerEncoderClassification(nn.Module):

    def __init__(self, text_embedding_vectors, head_num, dropout=0.1, d_model=300, max_seq_len=140, output_dim=5):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3 = nn.Dropout(dropout)
        self.net4_1 = TransformerBlock(d_model=d_model, head_num=head_num, dropout=dropout)
        self.net4_2 = TransformerBlock(d_model=d_model, head_num=head_num, dropout=dropout)
        self.net5 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  #Embedding
        x2 = self.net2(x1) #PositinalEncoding
        x3 = self.net3(x2) #Dropout
        x4_1, normlized_weights_1 = self.net4_1(x3, mask) #self-Attention+FFN
        x4_2, normlized_weights_2 = self.net4_2(x4_1, mask)  #self-Attention+FFN
        x5 = self.net5(x4_2)  #linear
        return x5, normlized_weights_1, normlized_weights_2

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}
# モデル構築
net = TransformerEncoderClassification(
    text_embedding_vectors=TEXT.vocab.vectors, head_num=5, dropout=0.1, d_model=300, max_seq_len=140, output_dim=4)

# ネットワークの初期化を定義


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# 訓練モードに設定
net.train()

# TransformerBlockモジュールを初期化実行
net.net4_1.apply(weights_init)
net.net4_2.apply(weights_init)


print('ネットワーク設定完了')

#損失関数を定義
criterion = nn.CrossEntropyLoss()

#最適化手法
learning_rate = 2e-5
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

#モデルを訓練して、訓練したモデルをreturnする
#モデル、辞書型で定義したdataloder(イテレータ)、損失関数、オプティマイザ、エポック数を渡す
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')
    # モデルをGPUへ渡す
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 各epoch
    for epoch in range(num_epochs):
        # 学習と検証
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 各バッチ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書オブジェクト

                # GPUが使えるならGPUにデータを送る
                inputs = batch.text[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # mask作成
                    input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                    input_mask = (inputs != input_pad) #mask部分がFalseに

                    # モデルに入力
                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        #勾配を計算
                        loss.backward()
                        #パラメータの更新
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

    return net

import torch.nn.functional as F
num_epochs = 10
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net_trained.eval()
net_trained.to(device)

y_true = np.array([])
y_pred = np.array([])

epoch_corrects = 0

for batch in (test_dl):
  inputs = batch.text[0].to(device)
  labels = batch.Label.to(device)

  with torch.set_grad_enabled(False):
    input_pad = 1
    input_mask = (inputs != input_pad)

    outputs, _, _ = net_trained(inputs, input_mask)
    _, preds = torch.max(outputs, 1)

    y_true = np.concatenate([y_true, labels.to("cpu", torch.double).numpy()])
    y_pred = np.concatenate([y_pred, preds.to("cpu", torch.double).numpy()])

    epoch_corrects += torch.sum(preds == labels.data)

# 正解率
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

print('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset),epoch_acc))

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

model_path = 'drive/My Drive/net0430.pth'
torch.save(net.to('cpu').state_dict(), model_path)

import pickle
with open('drive/My Drive/stoi0430.pkl', 'wb') as f:
    pickle.dump(dict(TEXT.vocab.stoi), f)

with open('drive/My Drive/itos0430.pkl', 'wb') as f:
    pickle.dump(list(TEXT.vocab.itos), f)

#torch.tensorをnumpy配列へ変換
x = TEXT.vocab.vectors.to('cpu').detach().numpy().copy()
np.save('drive/My Drive/omomi0430', x)
