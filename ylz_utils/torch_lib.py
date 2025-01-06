import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
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
    def from_df(cls,df:pd.DataFrame)->torch.Tensor:
        mapping_records = {}
        df=df.copy()
        columns = df.columns
        for col in columns:
            print("???",col)
            if df[col].dtypes == 'object':
                mapping_dict = {}
                unique_values = df[col].unique()
                for index, value in enumerate(unique_values):
                    mapping_dict[value] = index
                # 使用map函数根据映射字典进行替换
                df[col] = df[col].map(mapping_dict)
                mapping_records[col] = {
                    'positions': df.columns.get_loc(col),
                    'mapping': mapping_dict
                }
        tensor = torch.from_numpy(df.values).float()        
        #print("!!!",tensor[:5])
        return (tensor,mapping_records)

    @classmethod
    def get_dataloader(cls,df:pd.DataFrame,source_columns:list[str],target_columns:list[str],source_seq:int,target_seq:int=1,split=0.8,batch_size=32)->tuple[DataLoader,DataLoader]:
        assert(source_columns and target_columns)
        assert(source_seq and target_seq)
        print("old df length=",len(df))
        df = df[source_columns+target_columns].dropna()
        print("no na df length=",len(df))
        source_data,source_mapping = cls.from_df(df[source_columns])
        target_data,target_mapping = cls.from_df(df[target_columns])
        data_len = len(df) - source_seq - target_seq + 1
        source_seq_data = torch.stack([source_data[i:i+source_seq] for i in range(data_len)])
        target_seq_data = torch.stack([target_data[i+source_seq:i+source_seq+target_seq] for i in range(data_len)])
        train_size = int(len(target_seq_data)*split)        
        x_train,x_test = source_seq_data[:train_size],source_seq_data[train_size:]
        y_train,y_test = target_seq_data[:train_size],target_seq_data[train_size:]

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return (train_loader,test_loader,source_mapping,target_mapping)
    @classmethod
    def train(cls,model:nn.Module,train_loader:DataLoader,test_loader:DataLoader,source_mapping,target_mapping,epochs:int=100,lr:float=0.001,file_name:str='model.pth'):
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
                train_loss = criterion(output, batch_y.squeeze(-1))
                train_loss.backward()
                optimizer.step()            
            # 验证模型
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    output = model(batch_x.permute(1, 0, 2), batch_y.permute(1, 0, 2))
                    loss = criterion(output, batch_y.squeeze(-1))
                    total_loss += loss.item()
            if (epoch + 1) % 1 == 0: 
                print(f'Epoch [{epoch + 1}/{epochs}],Train Loss: {train_loss.item():.4f} , Val Loss: {total_loss / len(test_loader):.4f}')
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = model
                torch.save({"mapping":{"source":source_mapping,"target":target_mapping},
                            "paramers":best_model.paramers,
                            "model":best_model.state_dict()},file_name)  
    @classmethod
    def load_model(cls,file_name:str="model.pth"):
        model_dict = torch.load(file_name)
        print(model_dict.keys())
        print("mapping=",model_dict["mapping"])
        print("paramers=",model_dict["paramers"])
        paramers = model_dict["paramers"]
        model = TransformerModel(**paramers) 
        model.load_state_dict(model_dict["model"])
        return model
    @classmethod
    def predict(cls,model:nn.Module,src,tgt):
        model.eval()
        src = src[-1,:,:].unsqueeze(0)
        tgt = tgt[-1,:,:].unsqueeze(0)
        temp = torch.zeros(1,1,1)
        with torch.no_grad():
            output = model(src.permute(1, 0, 2), temp.permute(1, 0, 2))
        print("src=",src,src.shape)
        print("tgt=",tgt,tgt.shape)
        print("output=",output)
        return output

class TransformerModel(nn.Module):
    def __init__(self, *, source_dim:int, d_model:int, nhead:int=4, n_layers:int=6,dropout=0.1,target_dim:int=1,is_embedded:bool=False,is_positional:bool=False):
        super(TransformerModel, self).__init__()
        self.is_embedded = is_embedded
        self.is_positional = is_positional
        self.paramers = {"source_dim":source_dim,"d_model":d_model,"target_dim":target_dim,
                         "nhead":nhead,"n_layers":n_layers,"dropout":dropout,
                         "is_embedded":is_embedded,"is_positional":is_positional}
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
    epochs = 50
    lr = 0.001
    batch_size = 200
    source_seq = 10
    target_seq = 1
    d_model=64
    source_column=['o','h','l','c','v','v_status','c_status','hs','zf','ma5c','ma10c','macd']
    source_column=['c']
    source_dim=len(source_column)
    target_column=['zd']
    target_column=['zd']
    target_dim=len(target_column)
    df=pd.DataFrame({"c":range(300),"zd":range(300)})
    print(df)
    train_loader,test_loader,source_mapping,target_mapping = TorchLib.get_dataloader(
            df,source_column,target_column,source_seq,target_seq,batch_size=batch_size)
    print(source_mapping,target_mapping)
    
    #model = TorchLib.load_model("model.pth") 
    #src,tgt=next(iter(test_loader))
    #pred = TorchLib.predict(model,src,tgt)
    #exit(1)
    
    model = TransformerModel(source_dim=source_dim,target_dim=target_dim,d_model=d_model)
    TorchLib.train(model,train_loader,test_loader,source_mapping,target_mapping,epochs,lr)
        
