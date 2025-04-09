import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, *, source_column:list[str],target_column:list[str],key_column:str,source_seq:int,target_seq:int=1,
                 d_model:int, nhead:int=4, n_layers:int=6,dropout=0.1,
                 is_embedded:bool=False,is_positional:bool=False,norm_type=None,mapping=None):
        super(TransformerModel, self).__init__()
        self.mapping=mapping
        self.source_column = source_column
        self.source_dim = len(source_column)
        self.target_column = target_column
        self.target_dim = len(target_column)
        self.key_column = key_column
        self.all_dim = self.source_dim + self.target_dim
        self.source_seq = source_seq
        self.target_seq = target_seq
        self.d_model = d_model
        self.is_embedded = is_embedded
        self.is_positional = is_positional
        self.norm_type = norm_type
        
        self.paramers = {"source_column":source_column,"target_column":target_column,"key_column":key_column,
                         "source_seq":source_seq,"target_seq":target_seq,
                         "d_model":d_model,"nhead":nhead,"n_layers":n_layers,"dropout":dropout,
                         "is_embedded":is_embedded,"is_positional":is_positional,"norm_type":norm_type}
        self.source_embedder = nn.Linear(self.source_dim, d_model)
        self.target_embedder = nn.Linear(self.target_dim, d_model)
        self.positional_train = PositionalEncoding(d_model,dropout=dropout)
        self.positional_eval = PositionalEncoding(d_model,dropout=0)
        self.layerNorm = nn.LayerNorm(self.d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=n_layers,num_decoder_layers=n_layers,dropout=dropout)
        self.fc = nn.Linear(d_model, self.target_dim)

    def forward(self, src, tgt):
        # 嵌入
        if not self.is_embedded:
            src = self.source_embedder(src)
            tgt = self.target_embedder(tgt)
        #位置编码
        if not self.is_positional:
            if self.training:
                src = self.positional_train(src)
                tgt = self.positional_train(tgt)
            else:
                src = self.positional_eval(src)
                tgt = self.positional_eval(tgt)
        #Layer Norm
        src = self.layerNorm(src)
        tgt = self.layerNorm(tgt)

        output = self.transformer(src,tgt)
        output = self.fc(output)
        return output

    def encode_mapping(self,df:pd.DataFrame,source_column=None,target_column=None,key_column=None)->torch.Tensor:
        if not source_column:
            source_column = self.source_column
        if not target_column:
            target_column = self.target_column
        if not key_column:
            key_column = self.key_column
        assert(source_column and target_column)
        source_mapping_records = {}
        target_mapping_records = {}
        print("old df length=",len(df))
        if not key_column:
            key_column = '__INDEX'
            df[key_column] = df.index
        df = df[source_column+target_column+[key_column]].dropna()
        print("no na df length=",len(df))

        # 判断是否是文本类型的训练
        # 是否需要jieba，20250409
        # if len(source_column)==1 and len(target_column)==1 and df[source_column[0]].dtypes == 'object' and df[target_column[0]].dtypes == 'object':
        #     df['source_jieba'] = df[source_column[0]].apply(lambda x:jieba.lcut(x))
        #     df['target_jieba'] = df[target_column[0]].apply(lambda x:jieba.lcut(x))
        #     df['source_len'] = df['source_jieba'].str.len()
        #     df['target_len'] = df['target_jieba'].str.len()
        #     max_source_len = df['source_len'].max().item()
        #     max_target_len = df['target_len'].max().item()
        #     total_token = df['source_jieba'].to_list() + df['target_jieba'].to_list()
        #     print(df)
        #     print(max_source_len,max_target_len)
        #     print(total_token)

        df=df.copy()
        columns = df.columns
        for col in columns:
            if col==key_column:
                continue
            if col in source_column: 
                source_mapping_dict = {}
                if df[col].dtypes == 'object':
                    col_type = 'object'
                    if self.mapping: #如果是predict目的，会采用模型导入。这种情况下才会有self.mapping
                        assert(self.mapping["source"][col]["type"]=="object")
                        mapping = self.mapping["source"][col]["mapping"]
                        df[col] = df[col].map(mapping)
                    else:
                        unique_values = df[col].unique()
                        for index, value in enumerate(unique_values):
                            source_mapping_dict[value] = index
                        # 使用map函数根据映射字典进行替换
                        df[col] = df[col].map(source_mapping_dict)
                else:
                    col_type = 'number'
                    source_mapping_dict["__NORM_TYPE"] = self.norm_type
                    if self.norm_type == 'std':
                        # 每列数值型进行标准化
                        mean = df[col].mean()
                        std = df[col].std()
                        df[col] = (df[col] - mean)/std
                        source_mapping_dict["__MEAN"] = mean
                        source_mapping_dict["__STD"] = std       
                source_mapping_records[col] = {
                    'positions': df.columns.get_loc(col),
                    'type':col_type,
                    'mapping': source_mapping_dict
                }
                
            if col in target_column:
                if col in source_column:
                    #col已经被编码，可以直接使用
                    target_mapping_records[col] = source_mapping_records[col]
                    continue
                target_mapping_dict = {}
                if df[col].dtypes == 'object':
                    col_type = 'object'
                    if self.mapping: #如果是predict目的，会采用模型导入。这种情况下才会有self.mapping
                        assert(self.mapping["target"][col]["type"]=="object")
                        mapping = self.mapping["target"][col]["mapping"]
                        df[col] = df[col].map(mapping)
                    else:
                        unique_values = df[col].unique()
                        for index, value in enumerate(unique_values):
                            target_mapping_dict[value] = index
                        # 使用map函数根据映射字典进行替换
                        df[col] = df[col].map(target_mapping_dict)
                else:
                    col_type = 'number'
                    target_mapping_dict["__NORM_TYPE"] = self.norm_type
                    if self.norm_type == 'std':
                        # 每列数值型标准化
                        mean = df[col].mean()
                        std = df[col].std()
                        df[col] = (df[col] - mean)/std               
                        target_mapping_dict["__MEAN"] = mean
                        target_mapping_dict["__STD"] = std
                target_mapping_records[col] = {
                    'positions': df.columns.get_loc(col),
                    'type':col_type,
                    'mapping': target_mapping_dict
                } 
        return (df,source_mapping_records,target_mapping_records)

    def set_dataloader(self,df:pd.DataFrame,
                       source_column=None,target_column=None,key_column=None,source_seq=None,target_seq=None,
                       split:float=0.8,batch_size:int=32,filter:list[str]=[])->tuple[DataLoader,DataLoader,dict,dict]:
        if not source_column:
            source_column = self.source_column
        if not target_column:
            target_column = self.target_column
        if not key_column:
            key_column = self.key_column
        if not source_seq:
            source_seq = self.source_seq
        if not target_seq:
            target_seq = self.target_seq
        assert(source_column and target_column)
        assert(source_seq and target_seq)
        df,source_mapping,target_mapping = self.encode_mapping(df)
        data_len = len(df) - source_seq - target_seq + 1
        # source_seq_data = torch.stack([source_data[i:i+source_seq] for i in range(data_len)])
        # target_seq_data = torch.stack([target_data[i+source_seq:i+source_seq+target_seq] for i in range(data_len)])
        if not key_column:
            key_column = '__INDEX'
        if filter:
            source_key_data = [df[i:i+source_seq][key_column].tolist()
                                for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq][key_column].iloc[-1]})]
            target_key_data = [df[i+target_seq:i+source_seq+target_seq][key_column].tolist()
                                for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq][key_column].iloc[-1]})]
            source_seq_data = torch.stack([torch.from_numpy(df[i:i+source_seq][source_column].values).float() 
                                    for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq][key_column].iloc[-1]})])
            target_seq_data = torch.stack([torch.from_numpy(df[i+target_seq:i+source_seq+target_seq][target_column].values).float() 
                                    for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq][key_column].iloc[-1]})])
        else:
            source_key_data = [df[i:i+source_seq][key_column].tolist() for i in range(data_len)]
            target_key_data = [df[i+target_seq:i+source_seq+target_seq][key_column].tolist() for i in range(data_len)]
            source_seq_data = torch.stack([torch.from_numpy(df[i:i+source_seq][source_column].values).float() 
                                    for i in range(data_len)])
            target_seq_data = torch.stack([torch.from_numpy(df[i+target_seq:i+source_seq+target_seq][target_column].values).float() 
                                    for i in range(data_len)])

        train_size = int(len(source_seq_data)*split)        
        x_train,x_test = source_seq_data[:train_size],source_seq_data[train_size:]
        x_train_key,x_test_key = source_key_data[:train_size],source_key_data[train_size:]
        y_train,y_test = target_seq_data[:train_size],target_seq_data[train_size:]
        y_train_key,y_test_key = target_key_data[:train_size],target_key_data[train_size:]
        train_dataset = TensorWithKeyDataset(x_train,y_train,x_train_key,y_train_key)
        test_dataset = TensorWithKeyDataset(x_test,y_test,x_test_key,y_test_key)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.mapping = {"source":source_mapping,"target":target_mapping}
        return (train_loader,test_loader,self.mapping)

    def model_train(self,*,train_loader:DataLoader=None,test_loader:DataLoader=None,epochs:int=100,lr:float=0.001,file_name:str='model.pth'):
        if not train_loader:
            train_loader = self.train_loader
        if not test_loader:
            test_loader = self.test_loader
        assert(train_loader and test_loader)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(),lr=lr)
        # 学习率调度器
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)                         
        best_loss = float('inf')
        for epoch in range(epochs):
            # 训练模型
            self.train()
            for batch_x, batch_y,_,_ in train_loader:
                optimizer.zero_grad()
                # 使用前一时间步的输入作为目标序列
                src = batch_x.permute(1, 0, 2)
                #tgt = batch_y.permute(1, 0, 2)
                #tgt = torch.rand(batch_y.shape).permute(1,0,2)
                #tgt = batch_x[:,-1,-1].permute(1,0,2)
                tgt = torch.cat([batch_y[:,:-1,:],torch.rand(batch_y.size(0),1,batch_y.size(2))],dim=1).permute(1,0,2)
                output = self(src,tgt)
                train_loss = criterion(output[-1,:], batch_y.squeeze(-1))
                train_loss.backward()
                optimizer.step()
                #scheduler.step() 
            # 验证模型
            self.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y,_,_ in test_loader:
                    src = batch_x.permute(1, 0, 2)
                    #tgt = batch_y.permute(1, 0, 2)
                    #tgt = torch.rand(batch_y.shape).permute(1,0,2)
                    #tgt = batch_x[:,-1,-1].permute(1,0,2)
                    tgt = torch.cat([batch_y[:,:-1,:],torch.rand(batch_y.size(0),1,batch_y.size(2))],dim=1).permute(1,0,2)
                    output = self(src,tgt)
                    loss = criterion(output[-1,:], batch_y.squeeze(-1))
                    total_loss += loss.item()
            if (epoch + 1) % 1 == 0: 
                print(f'Epoch [{epoch + 1}/{epochs}],Train Loss: {train_loss.item():.4f} , Val Loss: {total_loss / len(test_loader):.4f} , Best Loss: {best_loss}')
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = self
                torch.save({"mapping":self.mapping,
                            "model_paramers":{"source_column":self.source_column,"target_column":self.target_column,"key_column":self.key_column,
                                        "source_seq":self.source_seq,"target_seq":self.target_seq,
                                        **best_model.paramers},
                            "model":best_model.state_dict()},file_name) 
        print(f"The best val loss is {best_loss}") 
    @classmethod
    def load_model(cls,file_name:str="model.pth"):
        model_dict = torch.load(file_name)
        mapping =model_dict["mapping"]
        model_paramers = model_dict["model_paramers"]
        model = TransformerModel(**model_paramers,mapping=mapping) 
        model.load_state_dict(model_dict["model"])
        return model
    
    def model_predict(self,loader:DataLoader=None):
        if not loader:
            loader = self.train_loader
        assert(loader)
        self.eval()
        with torch.no_grad():
            for batch_x, batch_y,batch_x_key,batch_y_key in loader:
                src = batch_x.permute(1,0,2)
                #tgt = batch_y.permute(1,0,2)
                tgt = torch.cat([batch_y[:,:-1,:],torch.rand(batch_y.size(0),1,batch_y.size(2))],dim=1).permute(1,0,2)
                #tgt = batch_x[:,-1,:-1].permute(1,0,2)
                output = self(src, tgt)
                if self.mapping:
                    if self.mapping["target"][self.target_column[0]]["type"]=="number" and self.mapping["target"][self.target_column[0]]["mapping"]['__NORM_TYPE']=="std":
                        mean = self.mapping["target"][self.target_column[0]]['mapping']['__MEAN'] 
                        std = self.mapping["target"][self.target_column[0]]['mapping']['__STD'] 
                        print(batch_x_key,"1===>",batch_y_key,"\nbatch_y=",((batch_y*std)+mean)[:,:],"\noutput=",((output*std)+mean)[-1])
                    else:
                        print(batch_x_key,"2===>",batch_y_key,"\nbatch_y=",batch_y[:,-1],"\noutput=",output[-1])
        return output
    def subseqent_mask(self,size):
        attn_shape = (1,size,size)
        mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
        return torch.from_numpy(1 - mask)
    
class TorchLib:
    @classmethod
    def to_df(cls,tensor:torch.Tensor,columns=[])->pd.DataFrame:
        numpy_array = tensor.numpy()
        if columns:
            df = pd.DataFrame(numpy_array,columns=columns)
        else:
            df = pd.DataFrame(numpy_array)
        return df

class TensorWithKeyDataset(Dataset):
    def __init__(self, source_seq_data, target_seq_data,source_key_data,target_key_data):
        assert len(source_seq_data) == len(target_seq_data) == len(source_key_data) == len(target_key_data), "all data must have the same length."
        self.source_seq_data = source_seq_data
        self.target_seq_data = target_seq_data
        self.source_key_data = source_key_data
        self.target_key_data = target_key_data
        #print(self.source_key_data)
        #print(self.target_key_data)
    def __len__(self):
        return len(self.source_seq_data)

    def __getitem__(self, idx):
        # 返回字符串和对应的数值
        return self.source_seq_data[idx], self.target_seq_data[idx] ,self.source_key_data[idx],self.target_key_data[idx]
class Embedding(torch.nn.Module):
    def __init__(self,d_model:int,vocab:int):
        super(Embedding,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model        
    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)
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

