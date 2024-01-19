import torch
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class timeRNN(torch.nn.Module):
    
    def __init__(self, dataset_name, time_interval, h_dim, dropout):
        super(timeRNN, self).__init__()
        self.dataset = dataset_name
        self.time_interval = time_interval
        self.pad_len = 8
        self.pad_id = 52
        self.time_element_num = 53  # year-10 month-12 day-10 hour-10 min-10 + padding(last)
        self.h_dim = h_dim

        self.time_embedding = torch.nn.Parameter(torch.Tensor(self.time_element_num, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.time_embedding)

        self.gru_net = torch.nn.GRU(input_size=h_dim, hidden_size=h_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.line_layer = torch.nn.Linear(self.pad_len * h_dim, 1)

    def forward(self, time_list):
        batch_size = len(time_list)

        t_num = time_list[0].item()
        # t_num = time_list[0]
        real_time = self.num_to_time(t_num)
        time_seq = self.time_to_seq(real_time)

        time_emb = self.time_embedding[time_seq]
        time_emb = time_emb.unsqueeze(0)
        time_emb = time_emb.repeat(batch_size, 1, 1)
        time_emb, h0 = self.gru_net(time_emb)
        time_emb = self.dropout(time_emb)
        time_emb = time_emb.reshape(batch_size, -1)
        time_emb = self.line_layer(time_emb)
        return time_emb

    def time_to_seq(self, t: datetime):
        res = []
        years = ['y' + ele for ele in str(t.year)]
        month = ['m' + str(t.month)]
        day = ['d' + ele for ele in str(t.day)]
        hour = ['h' + ele for ele in str(t.hour)]
        minute = ['u' + ele for ele in str(t.minute)]

        res.extend(years)   # WIKI and YAGO
        if 'ICEWS14' in self.dataset or 'ICEWS05-15' in self.dataset or 'ICEWS18' in self.dataset:
            res.extend(month)
            res.extend(day)
        if 'GDELT' in self.dataset:
            res.extend(hour)
            res.extend(minute)

        res = [self.time_dic(ele) for ele in res]
        res = self.pad_time(res)

        return res

    def pad_time(self, res: list):
        if len(res) < self.pad_len:
            res.extend([self.pad_id] * (self.pad_len - len(res)))
        return res

    def num_to_time(self, t_num):
        now_date = None

        # time_interval = 24 hours
        if 'ICEWS14' in self.dataset:
            data_str = '2014-01-01'
            format = '%Y-%m-%d'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + timedelta(days=t_num)
        if 'ICEWS05-15' in self.dataset:
            data_str = '2015-01-01'
            format = '%Y-%m-%d'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + timedelta(days=t_num)
        if 'ICEWS18' in self.dataset:
            data_str = '2018-01-01'
            format = '%Y-%m-%d'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + timedelta(days=t_num)

        # time_interval = 1 year
        if 'WIKI' in self.dataset:
            data_str = '1810'
            format = '%Y'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + relativedelta(years=t_num)
        if 'YAGO' in self.dataset:
            data_str = '1840'
            format = '%Y'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + relativedelta(years=t_num)

        # time_interval = 15 minutes
        if 'GDELT' in self.dataset:
            data_str = '1979-01-01 00:00'
            format = '%Y-%m-%d %H:%M'
            start_date = datetime.strptime(data_str, format)
            now_date = start_date + timedelta(minutes=t_num * 15)

        return now_date

    @staticmethod
    def time_dic(time_piece):
        d = [
            'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9',
            'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
            'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
            'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',
            'u0', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9',
            'pad'
        ]   # 52
        return d.index(time_piece)


# r = timeRNN('GDELT', 1, 200, 0.3)
# r([178, 178])
