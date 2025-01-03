import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
class TorchLib:
    @classmethod
    def to_df(cls,tensor:torch.Tensor,columns=[])->pd.DataFrame:
        numpy_array = tensor.numpy()
        if columns:
            df = pd.DataFrame(numpy_array,columns=columns)
        else:
            df = pd.DataFrame(numpy_array)
        return df
    @classmethod
    def get_dataloader(self,df:pd.DataFrame,source_columns:list[str],target_columns:list[str],source_seq:int,target_seq:int=1,split=0.8,batch_size=32)->tuple[DataLoader,DataLoader]:
        assert(source_columns and target_columns)
        assert(source_seq and target_seq)
        source_data = df[source_columns].values
        target_data = df[target_columns].values
        data_len = len(df) - source_seq - target_seq + 1
        source_seq_data = torch.tensor([source_data[i:i+source_seq] for i in range(data_len)]).float()
        target_seq_data = torch.tensor([target_data[i+source_seq:i+source_seq+target_seq] for i in range(data_len)]).float()
        train_size = int(len(target_seq_data)*split)        
        
        x_train,x_test = source_seq_data[:train_size],source_seq_data[train_size:]
        y_train,y_test = target_seq_data[:train_size],target_seq_data[train_size:]

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return (train_loader,test_loader)
    @classmethod
    def train(self,model:nn.Module,train_loader:DataLoader,test_loader:DataLoader,epochs:int=100,lr:float=0.001,):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)
        best_loss = float('inf')
        for epoch in range(epochs):
            # 训练模型
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                # 使用前一时间步的输入作为目标序列
                output = model(batch_x.permute(1, 0, 2), batch_y.permute(1, 0, 2))
                train_loss = criterion(output, batch_y[1:])
                train_loss.backward()
                optimizer.step()            
            # 验证模型
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    output = model(batch_x.permute(1, 0, 2), batch_y.permute(1, 0, 2))
                    loss = criterion(output, batch_y[1:])
                    total_loss += loss.item()
            if (epoch + 1) % 10 == 0: 
                print(f'Epoch [{epoch + 1}/{epochs}],Train Loss: {train_loss.item():.4f} , Val Loss: {total_loss / len(test_loader):.4f}')
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = model
                torch.save(best_model.state_dict(),"transformer.pth")  

class TransformerModel(nn.Module):
    def __init__(self, *, source_dim:int, d_model:int, nhead:int=4, n_layers:int=6,dropout=0.1,target_dim:int=1,is_embedded:bool=False,is_positional:bool=False):
        super(TransformerModel, self).__init__()
        self.is_embedded = is_embedded
        self.is_positional = is_positional
        self.source_embedder = nn.Linear(source_dim, d_model)
        self.target_embedder = nn.Linear(target_dim, d_model)
        self.positional = PositionalEncoding(d_model,dropout=0)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=n_layers,num_decoder_layers=n_layers,dropout=dropout)
        self.fc = nn.Linear(d_model, target_dim)

    def forward(self, src, tgt):
        if not self.is_embedded:
            src = self.source_embedder(src)
            tgt = self.target_embedder(tgt)
        if not self.is_positional:
            src = self.positional(src)
            tgt = self.positional(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output[-1, :, :])  # 取最后一个时间步的输出
        return output

    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # 创建一个 (max_len, d_model) 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # 增加一个维度，使其形状为 (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 注册为 buffer，不会被视为模型参数

    def forward(self, x):
        # x 的形状为 (seq_len, batch_size, d_model)
        x = x + self.pe[:, :x.size(1), :]  # 将位置编码加到输入上
        return self.dropout(x)

if __name__ == '__main__':
    import sqlite3
    conn = sqlite3.connect("daily_pro.db")
    df = pd.read_sql("select * from daily where code='000001'",conn)
    epochs = 100
    lr = 0.001
    batch_size = 200
    source_seq = 20
    target_seq = 1
    train_loader,test_loader = TorchLib.get_dataloader(df,['o','h','l','c','zd'],['c'],source_seq,target_seq,batch_size=batch_size)
    model = TransformerModel(source_dim=5,target_dim=1,d_model=64)
    TorchLib.train(model,train_loader,test_loader,epochs,lr)
        