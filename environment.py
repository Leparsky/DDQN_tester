# номер свечки за день попробовать добавить к состоянию
# обьем попробовать добавить к состоянию
# изменение обьема  попробовать добавить к состоянию
import pandas as pd
import numpy as np

class Environment:
    def __init__(self, args):
        #self.data = data
        self.args = args
        self.history_t = args.history_win

        self.state_size =  self.history_t+1
        if self.args.usevol:
            self.state_size += self.history_t
            self.history_vol = [0 for _ in range(self.history_t)]

        if args.candlenum:
            self.state_size += 3

        if args.allprices or args.allprices2:
            self.state_size += self.history_t*3
            self.open_vol = [0 for _ in range(self.history_t)]
            self.high_vol = [0 for _ in range(self.history_t)]
            self.low_vol = [0 for _ in range(self.history_t)]
          
        self.action_size = 3
        self.t = 0
        self.done = False
        self.profits = 0
        self.position = 0
        # self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]

    def GetStockDataVecFN(self, key=r'D:\PycharmProjects\RIZ8\SPFB.RTS-6.18(5M).csv', append=False):
        #append между разными фьючерсами лучше не делать, как мне кажется если поразмышлять логически
        dateparse = lambda x, y: pd.datetime.combine(pd.datetime.strptime(x, '%Y%m%d'),pd.datetime.strptime(y, '%H%M%S').time())
        df = pd.read_csv(key, delimiter=',', index_col=['datetime'],parse_dates={'datetime': ['<DATE>', '<TIME>']}, date_parser=dateparse)
        if append:
            self.data = pd.concat(self.data, df, ignore_index=True)
        else:
            self.data = df
        self.reset()

    '''def getStockDataVecFN(self,key =  r'D:\PycharmProjects\RIZ8\SPFB.RTS-6.18(5M).csv'):
            dateparse = lambda x, y: pd.datetime.combine(pd.datetime.strptime(x, '%Y%m%d'),
                                                         pd.datetime.strptime(y, '%H%M%S').time())
            self.data = pd.read_csv(key, delimiter=',', index_col=['datetime'],
                               parse_dates={'datetime': ['<DATE>', '<TIME>']}, date_parser=dateparse)
    '''
    def get_state_size(self):
        return self.state_size

    def get_action_size(self):
        return self.action_size

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.position = 0
        # self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        if self.args.usevol:
            self.history_vol  = [0 for _ in range(self.history_t)]
        if self.args.allprices or self.args.allprices2: 
            self.open_vol = [0 for _ in range(self.history_t)]
            self.hidh_vol = [0 for _ in range(self.history_t)]
            self.low_vol = [0 for _ in range(self.history_t)]
        for _ in range(self.history_t-1):  # step(0) - act = 0: stay
            self.step(0)
        res = self.step(0)
        return res[0]

    def step(self, act):
        datacol = '<CLOSE>'
        datacol_vol  = '<VOL>'
        opencol = '<OPEN>'
        highcol = '<HIGH>'
        lowcol = '<LOW>'
        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val=0
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:  # buy
            if self.position == 0:  # если не в позиции то добпвляем лонг с "+"
                self.position = self.data.iloc[self.t, :][datacol]
                buy = self.position
                # add value to long positions  (we enter at the candle close)
            elif self.position > 0:  # есть открытая длинная позиция
                reward = -100
            else:
                profit = (-self.data.iloc[self.t, :][datacol] - self.position)
                self.profits += profit
                buy = self.data.iloc[self.t, :][datacol]
                self.position = 0

                # Есть открытая короткая позиция

        elif act == 2:  # sell
            if self.position == 0:  # если не в позиции то добпвляем Short с "-"
                self.position = -self.data.iloc[self.t, :][datacol]
                # add value to long positions  (we enter at the candle close)
                sell = - self.position
            elif self.position < 0:  # есть открытая короткая позиция
                reward = -100
            else:
                profit = (self.data.iloc[self.t, :][datacol] - self.position)
                # Есть открытая короткая позиция
                self.profits += profit  # Remember profit
                sell = self.data.iloc[self.t, :][datacol]
                self.position = 0
        if self.position < 0:
            position_val = (-self.data.iloc[self.t, :][datacol] - self.position)
        elif  self.position > 0:
            position_val = (self.data.iloc[self.t, :][datacol] - self.position)
            # set next time
        self.t += 1
        # self.position_value = 0
        # for p in self.positions:  # calculate position value
        # self.position_value += (self.data.iloc[self.t, :][datacol] - p)
        self.history.pop(0)
        self.done = True if self.t>=len(self.data)-1 else False
        # method takes a single argument (index) and removes the element present at that index from the list. ...
        self.history.append(self.data.iloc[self.t, :][datacol] - self.data.iloc[(self.t - 1), :][
            datacol])  # add the diferrence between cureent CLose Value and prior Close Value

        result = [position_val + 0.0]

        if self.args.candlenum:
            result += [self.data.index[self.t].day + 0.0, self.data.index[self.t].dayofweek+1 + 0.0,(self.data.index[self.t].hour*60 + self.data.index[self.t].minute -600 )/5 +1]

        result +=self.history
        if self.args.usevol:
            self.history_vol.pop(0)
            self.history_vol.append(self.data.iloc[self.t, :][datacol_vol] - self.data.iloc[(self.t - 1), :][
                datacol_vol])  # add the diferrence between cureent CLose Value and prior Close Value
            result+= self.history_vol


        if self.args.allprices:

            self.open_vol.pop(0)
            self.open_vol.append(self.data.iloc[self.t, :][opencol] - self.data.iloc[(self.t - 1), :][
                opencol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.open_vol

            self.high_vol.pop(0)
            self.high_vol.append(self.data.iloc[self.t, :][highcol] - self.data.iloc[(self.t - 1), :][
                highcol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(self.data.iloc[self.t, :][lowcol] - self.data.iloc[(self.t - 1), :][
                lowcol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.low_vol
        elif self.args.allprices2:
            self.open_vol.pop(0)
            close_t = self.data.iloc[self.t, :][datacol]
            self.open_vol.append(self.data.iloc[self.t, :][opencol] - close_t)  # add the diferrence
            result += self.open_vol

            self.high_vol.pop(0)
            self.high_vol.append(self.data.iloc[self.t, :][highcol]  - close_t)  # add the diferrence
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(self.data.iloc[self.t, :][lowcol]  - close_t)  # add the diferrence
            result += self.low_vol

        # clipping reward
        if profit != 0:
            reward = profit
        elif position_val < -300:
            reward += position_val*3

        return np.array(result), reward, self.done, buy, sell, profit  # obs, reward, done,profit

    def step_old(self, act):
        datacol = '<CLOSE>'
        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val=0
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:  # buy
            if self.position == 0:  # если не в позиции то добпвляем лонг с "+"
                self.position = self.data.iloc[self.t, :][datacol]
                buy = self.position
                # add value to long positions  (we enter at the candle close)
            elif self.position > 0:  # есть открытая длинная позиция
                reward = -100
            else:
                profit = (-self.data.iloc[self.t, :][datacol] - self.position)
                self.profits += profit
                buy = self.data.iloc[self.t, :][datacol]
                self.position = 0

                # Есть открытая короткая позиция

        elif act == 2:  # sell
            if self.position == 0:  # если не в позиции то добпвляем Short с "-"
                self.position = -self.data.iloc[self.t, :][datacol]
                # add value to long positions  (we enter at the candle close)
                sell = - self.position
            elif self.position < 0:  # есть открытая короткая позиция
                reward = -100
            else:
                profit = (self.data.iloc[self.t, :][datacol] - self.position)
                # Есть открытая короткая позиция
                self.profits += profit  # Remember profit
                sell = self.data.iloc[self.t, :][datacol]
                self.position = 0
        if self.position < 0:
            position_val = (-self.data.iloc[self.t, :][datacol] - self.position)
        elif  self.position > 0:
            position_val = (self.data.iloc[self.t, :][datacol] - self.position)
            # set next time
        self.t += 1
        # self.position_value = 0
        # for p in self.positions:  # calculate position value
        # self.position_value += (self.data.iloc[self.t, :][datacol] - p)
        self.history.pop(0)
        # method takes a single argument (index) and removes the element present at that index from the list. ...
        self.history.append(self.data.iloc[self.t, :][datacol] - self.data.iloc[(self.t - 1), :][
            datacol])  # add the diferrence between cureent CLose Value and prior Close Value

        # clipping reward
        if profit != 0:
            reward = profit
        elif position_val < -300:
            reward += position_val*3
        return np.array([position_val] + self.history), reward, self.done, buy, sell, profit  # obs, reward, done,profit
