# 加载数据
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

data_path = "./data/poetry_5.txt"
final_model_path = "./model/Poetry5_LSTM_model.pt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"设备: {device}")
max_len = 30
poetry = []
char2id = {}
id2char = ['<pad>', '<unk>', '<sos>', '<eos>']


# 打开文件并读取数据
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        poetry.append(line.strip())  # 去掉换行符并存储每一行
        # 制作词汇表
        for char in line.strip():  # 去掉每行的换行符后逐字符处理
            if char not in id2char:  # 避免重复添加字符
                id2char.append(char)

# 构建字符到 ID 的映射
char2id = {char: idx for idx, char in enumerate(id2char)}
print(f"词典大小: {len(char2id)}")
print(f"数据集大小: {len(poetry)}")
# 打印结果
print("char2id 示例：", list(char2id.items())[:10])  # 打印前 10 个字符及其 ID
print("id2char 示例：", id2char[:10])  # 打印前 10 个字符

def text_to_id(text, vocab):
    """将字符序列转换为词汇表中的ID"""
    return [vocab.get(char, vocab['<unk>']) for char in text]


class CustomDataset(Dataset):
    def __init__(self, dataset, vocab, max_length=30):
        self.dataset = dataset
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # 将输入文本和标签转换为ID形式
        ids = [self.vocab['<sos>']] + text_to_id(data, self.vocab) + [self.vocab['<eos>']]
        return ids, data

    def collate_fn(self, batch):
        batch_ids, batch_texts = zip(*batch)
        batch_data = pad_sequence([torch.LongTensor(ids) for ids in batch_ids], True, self.vocab['<pad>'])
        batch_inputs = batch_data[:, :-1]
        batch_labels = batch_data[:, 1:]
        return batch_inputs, batch_labels, batch_texts

train_dataset = CustomDataset(poetry, char2id)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)

import torch.nn as nn
import torch.optim as optim
import numpy as np

# LSTM 语言模型
class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=128):
        super(LSTMTextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2  # 2
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)
        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h = torch.tensor(np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)).to(device)
            c = torch.tensor(np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)).to(device)
        else:
            h, c = hidden
        h = h.to(device)
        c = c.to(device)
        x = self.embedding(x)
        output, hidden = self.lstm(x, (h, c))  # , (h0, c0)
        output = self.dropout(output)
        # output = self.layer_norm(output)  # 归一化
        output = self.fc(output)
        return output, hidden

# 训练和生成函数（简化版）

def generate_poetry(model, char_to_id, id_to_char, start_text="", max_length=30):
    """
    使用贪婪解码生成诗歌。
    :param model: 训练好的模型
    :param char_to_id: 字符到 ID 的映射字典
    :param id_to_char: ID 到字符的映射字典
    :param start_text: 起始文本（可选）
    :param max_length: 最大生成长度
    :return: 生成的诗歌文本
    """
    model.eval()
    device = next(model.parameters()).device

    # 初始化输入序列
    input_ids = [char_to_id['<sos>']] + [char_to_id.get(c, char_to_id['<unk>']) for c in start_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_text
    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            output, _ = model(input_tensor)
            output = output[:, -1, :]  # 取最后一个时间步的输出
            predicted_id = torch.argmax(output, dim=-1).item()  # 选择概率最大的词

            if predicted_id == char_to_id['<eos>'] or len(generated_text) == 24:
                break

            generated_text += id_to_char[predicted_id]  # 将预测的字符添加到生成文本中
            input_tensor = torch.tensor([input_ids + [predicted_id]], dtype=torch.long).to(device)  # 更新输入序列

    return generated_text

def generation_by_hidden_head(model, char_to_id, id_to_char, head_text="", max_length=30):
    """
    使用贪婪解码生成诗歌。
    :param model: 训练好的模型
    :param char_to_id: 字符到 ID 的映射字典
    :param id_to_char: ID 到字符的映射字典
    :param start_text: 起始文本（可选）
    :param max_length: 最大生成长度
    :return: 生成的诗歌文本
    """
    model.eval()
    device = next(model.parameters()).device

    generated_text = ""
    fuhao = ['，', '。', '，', '。']
    with torch.no_grad():
        for i in range(len(head_text)):
            part = head_text[i]   # 部分诗句
            # 初始化输入序列
            input_ids = [char_to_id['<sos>']] + [char_to_id.get(c, char_to_id['<unk>']) for c in part]
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            for _ in range(4):
                output, _ = model(input_tensor)
                output = output[:, -1, :]  # 取最后一个时间步的输出
                predicted_id = torch.argmax(output, dim=-1).item()  # 选择概率最大的词

                # if predicted_id == char_to_id['<eos>'] or len(generated_text) == 6:
                #     break

                part += id_to_char[predicted_id]  # 将预测的字符添加到生成文本中
                input_tensor = torch.tensor([input_ids + [predicted_id]], dtype=torch.long).to(device)  # 更新输入序列
            part += fuhao[i]
            generated_text += part
    return generated_text


# 加载模型
def load_trained_model(model_path, vocab):
    model = LSTMTextGenerator(len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

import os
def train_lstm_model():
    model = LSTMTextGenerator(len(char2id)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    epochs = 1000
    loss_fn = nn.CrossEntropyLoss(ignore_index=char2id['<pad>'])
    # 检查模型是否已经存在
    if os.path.isfile(final_model_path):
        print(f"发现模型文件 {final_model_path}，正在加载...")
        model.load_state_dict(torch.load(final_model_path))
        print("模型加载完成")

    # 查看总参量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for num, (input_ids, labels, texts) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output, hidden = model(input_ids)
            # 调整输出和标签的形状
            # (batch_size * seq_len, vocab_size),  (batch_size * seq_len)
            loss = loss_fn(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            # if (num+1) % 50 == 0:
            #     print(f"\tBatch {num+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        # scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        poetry_output = generate_poetry(model, char2id, id2char, start_text="思去愁")
        print("生成的诗句:", poetry_output)
        # 保存模型
        torch.save(model.state_dict(), final_model_path)

if __name__ == '__main__':
    # train_lstm_model()
    # 使用示例
    trained_model = load_trained_model(final_model_path, char2id)
    poetry_output = generate_poetry(trained_model, char2id, id2char, start_text="思去愁")
    head_poetry = generation_by_hidden_head(trained_model, char2id, id2char, head_text="思去愁苦")
    print("生成的诗句:", poetry_output)
    print("藏头诗:", head_poetry)



