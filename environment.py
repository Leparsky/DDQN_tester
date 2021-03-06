# номер свечки за день попробовать добавить к состоянию
# обьем попробовать добавить к состоянию
# изменение обьема  попробовать добавить к состоянию
import pandas as pd
import numpy as np
import math
import pandas as pd

class Environment:
    def __init__(self, args):
        #self.data = None
        self.args = args
        self.history_t = args.history_win
        self.state_size = self.history_t + 1



        if self.args.madifference:
            self.state_size += self.history_t
            self.madifference_vol = [0 for _ in range(self.history_t)]

        if self.args.ma1 and not self.args.hidema:
            self.state_size += self.history_t
            self.ma1_vol = [0 for _ in range(self.history_t)]

        if self.args.ma2 and not self.args.hidema:
            self.state_size += self.history_t
            self.ma2_vol = [0 for _ in range(self.history_t)]
        #args.ma1, args.ma2,

        if self.args.usevol:
            self.state_size += self.history_t
            self.history_vol = [0 for _ in range(self.history_t)]

        if args.candlenum:
            self.state_size += 3

        if args.allprices or args.allprices2 or args.allprices3:
            self.state_size += self.history_t * 3
            self.open_vol = [0.0 for _ in range(self.history_t)]
            self.high_vol = [0.0 for _ in range(self.history_t)]
            self.low_vol = [0.0 for _ in range(self.history_t)]

        self.action_size = 3
        self.t = 0
        self.done = False
        self.profits = 0
        self.position = 0
        # self.position_value = 0
        self.history = [0.0 for _ in range(self.history_t)]

        self.datacol_vol = '<VOL>'
        self.opencol = '<OPEN>'
        self.highcol = '<HIGH>'
        self.lowcol = '<LOW>'
        self.datacol = '<CLOSE>'
        self.datacol = '<CLOSE>'
        self.tickercol = '<TICKER>'
    def clearDataVecFN(self):
        self.data = self.data.iloc[0:0]
    def check_data_integrity4prediction_at(self,datatime):
        tt = self.data.loc[datatime]
        tt1 = self.data.loc[datatime]
    def adddatacandle(self,datetime,ticker,per, open,high,low,close,vol):
        try:
            if self.data:
                self.data.append({'<TICKER>': ticker,
                            '<PER>' : per,
                            '<OPEN>': open,
                            '<HIGH>': high,
                            '<LOW>' : low,
                            '<CLOSE>' : close,
                            '<VOL>' : vol,
                            'datetime': datetime})
        except :
            self.data = pd.DataFrame({'<TICKER>': ticker,
                                      '<PER>': per,
                                      '<OPEN>': open,
                                      '<HIGH>': high,
                                      '<LOW>': low,
                                      '<CLOSE>': close,
                                      '<VOL>': vol
                                      }, index=[datetime])
            #self.data.index = [datetime]
            self.data.indexname = "datetime"
            if self.args.ma1:
                self.data['ma1'] = self.data[self.datacol].rolling(self.args.ma1).mean()
                # self.data = self.data.join(MA)

            if self.args.ma2:
                self.data['ma2'] = self.data[self.datacol].rolling(self.args.ma2).mean()
                # MA = pd.Series(pd.rolling_mean(df[self.datacol], self.args.ma2), name='ma2')
                # self.data = self.data.join(MA)

    def GetStockDataVecFN(self, key=r'D:\PycharmProjects\RIZ8\SPFB.RTS-6.18(5M).csv', append=False):
        # append между разными фьючерсами лучше не делать, как мне кажется если поразмышлять логически
        dateparse = lambda x, y: pd.datetime.combine(pd.datetime.strptime(x, '%Y%m%d'),
                                                     pd.datetime.strptime(y, '%H%M%S').time())
        df = pd.read_csv(key, delimiter=',', index_col=['datetime'], parse_dates={'datetime': ['<DATE>', '<TIME>']},
                         date_parser=dateparse)
        try:
            if append:
                self.data.append(df, ignore_index=True)
            else:
                self.data = df
        except AttributeError:
            self.data = df

        if self.args.ma1:
            self.data['ma1'] = self.data[self.datacol].rolling(self.args.ma1).mean()
            #self.data = self.data.join(MA)

        if self.args.ma2:
            self.data['ma2'] = self.data[self.datacol].rolling(self.args.ma2).mean()
            #MA = pd.Series(pd.rolling_mean(df[self.datacol], self.args.ma2), name='ma2')
            #self.data = self.data.join(MA)

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
        self.positions = []
        # self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        if self.args.ma1 and not self.args.hidema:
            self.ma1_vol = [0 for _ in range(self.history_t)]

        if self.args.ma2 and not self.args.hidema:
            self.ma2_vol = [0 for _ in range(self.history_t)]

        if self.args.madifference:
            self.madifference_vol = [0 for _ in range(self.history_t)]

        if self.args.usevol:
            self.history_vol = [0 for _ in range(self.history_t)]
        if self.args.allprices or self.args.allprices2 or self.args.allprices3:
            self.open_vol = [0 for _ in range(self.history_t)]
            self.hidh_vol = [0 for _ in range(self.history_t)]
            self.low_vol = [0 for _ in range(self.history_t)]
        for _ in range(self.history_t + np.max([self.args.ma1, self.args.ma2]) - 1):  # step(0) - act = 0: stay
            self.step(0)
        res = self.step(0)
        return res[0]

    def gotnextstate(self):

        result = []
        self.t += 1
        self.done = True if self.t >= len(self.data) - 1 else False

        if self.args.candlenum:
            result += [self.data.index[self.t].day + 0.0, self.data.index[self.t].dayofweek + 1 + 0.0,
                       (self.data.index[self.t].hour * 60 + self.data.index[self.t].minute - 600) / 5 + 1]
        if self.args.ma1 and self.t > self.args.ma1 and not self.args.hidema:
            self.ma1_vol.pop(0)
            self.ma1_vol.append(self.data.iloc[self.t, :]['ma1'] - self.data.iloc[(self.t - 1), :]['ma1'])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.ma1_vol
        if self.args.ma2 and self.t > self.args.ma2 and not self.args.hidema:
            self.ma2_vol.pop(0)
            self.ma2_vol.append(self.data.iloc[self.t, :]['ma2'] - self.data.iloc[(self.t - 1), :]['ma2'])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.ma2_vol

        if self.args.madifference and self.args.ma2 and self.t > self.args.ma2 and self.args.ma1 and self.t > self.args.ma1:
            self.madifference_vol.pop(0)
            self.madifference_vol.append(self.data.iloc[self.t, :]['ma2'] - self.data.iloc[self.t, :]['ma1'])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.madifference_vol

        if self.args.usevol:
            self.history_vol.pop(0)
            self.history_vol.append(self.data.iloc[self.t, :][self.datacol_vol] - self.data.iloc[(self.t - 1), :][
                self.datacol_vol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.history_vol

        if self.args.allprices:

            self.history.pop(
                0)  # method takes a single argument (index) and removes the element present at that index from the list. ...
            self.history.append(self.data.iloc[self.t, :][self.datacol] - self.data.iloc[(self.t - 1), :][
                self.datacol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.history

            self.open_vol.pop(0)
            self.open_vol.append(self.data.iloc[self.t, :][self.opencol] - self.data.iloc[(self.t - 1), :][
                self.opencol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.open_vol

            self.high_vol.pop(0)
            self.high_vol.append(self.data.iloc[self.t, :][self.highcol] - self.data.iloc[(self.t - 1), :][
                self.highcol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(self.data.iloc[self.t, :][self.lowcol] - self.data.iloc[(self.t - 1), :][
                self.lowcol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.low_vol

        elif self.args.allprices2:

            self.history.pop(
                0)  # method takes a single argument (index) and removes the element present at that index from the list. ...
            self.history.append(self.data.iloc[self.t, :][self.datacol] - self.data.iloc[(self.t - 1), :][
                self.datacol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.history

            self.open_vol.pop(0)
            close_t = self.data.iloc[self.t, :][self.datacol]
            self.open_vol.append(self.data.iloc[self.t, :][self.opencol] - close_t)  # add the diferrence
            result += self.open_vol

            self.high_vol.pop(0)
            self.high_vol.append(self.data.iloc[self.t, :][self.highcol] - close_t)  # add the diferrence
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(self.data.iloc[self.t, :][self.lowcol] - close_t)  # add the diferrence
            result += self.low_vol

        elif self.args.allprices3:
            self.history.pop(
                0)  # method takes a single argument (index) and removes the element present at that index from the list. ...
            self.history.append(
                math.log10(self.data.iloc[self.t - 1, :][self.datacol] / self.data.iloc[(self.t), :][self.datacol]))
            result += self.history

            self.open_vol.pop(0)
            self.open_vol.append(
                math.log10(self.data.iloc[self.t - 1, :][self.opencol] / self.data.iloc[(self.t), :][self.opencol]))
            result += self.open_vol

            self.high_vol.pop(0)
            self.high_vol.append(
                math.log10(self.data.iloc[self.t - 1, :][self.highcol] / self.data.iloc[(self.t), :][self.highcol]))
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(
                math.log10(self.data.iloc[self.t - 1, :][self.lowcol] / self.data.iloc[(self.t), :][self.lowcol]))
            result += self.low_vol
        else:
            self.history.pop(
                0)  # method takes a single argument (index) and removes the element present at that index from the list. ...
            self.history.append(self.data.iloc[self.t, :][self.datacol] - self.data.iloc[(self.t - 1), :][
                self.datacol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.history

        return result

    def step(self, act):
        return self.step1(act)

    def step1(self, act):
        #лонг

        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val = 0
        # act = 0: stay, 1: buy, 2: sell

        ticker = self.data.iloc[self.t, :][self.tickercol]
        if act == 1:  # buy
            self.positions.append(
                self.data.iloc[self.t, :][self.datacol])  # add value to long positions (we enter at the candle close)
        elif act == 2:  # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0  # calculate profit/loss current position

                for p in self.positions:
                    profits += (self.data.iloc[self.t, :][self.datacol] - p)
                reward += profits
                self.profits += profits  # Remember profit
                profit = profits
                self.positions = []  # Close all long positions

        result = self.gotnextstate()

        position_val = 0
        for p in self.positions:  # calculate position value
            position_val += (self.data.iloc[self.t, :][self.datacol] - p)




        result = [position_val + 0.0] + result


        if ticker != self.data.iloc[self.t, :][self.tickercol]:
            for _ in range(self.history_t):  # step(0) - act = 0: stay
                result = [0.0] + self.gotnextstate()
            reward = 0
            position_val = 0
            self.position = 0
            profit = position_val

        # clipping reward

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return np.array(result), reward, self.done, buy, sell, profit  # obs, reward, done,profit

    def step0(self, act):
        #-шорт и лонг

        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val = 0
        # act = 0: stay, 1: buy, 2: sell

        ticker = self.data.iloc[self.t, :][self.tickercol]
        if act == 1:  # buy
            if self.position == 0:  # если не в позиции то добпвляем лонг с "+"
                self.position = self.data.iloc[self.t, :][self.datacol]
                buy = self.position
                # add value to long positions  (we enter at the candle close)
            elif self.position > 0:  # есть открытая длинная позиция
                reward = -100
            else:
                profit = (-self.data.iloc[self.t, :][self.datacol] - self.position)
                self.profits += profit
                buy = self.data.iloc[self.t, :][self.datacol]
                self.position = 0
                # Есть открытая короткая позиция

        elif act == 2:  # sell
            if self.position == 0:  # если не в позиции то добпвляем Short с "-"
                self.position = -self.data.iloc[self.t, :][self.datacol]
                # add value to long positions  (we enter at the candle close)
                sell = - self.position
            elif self.position < 0:  # есть открытая короткая позиция
                reward = -100
            else:
                profit = (self.data.iloc[self.t, :][self.datacol] - self.position)
                # Есть открытая короткая позиция
                self.profits += profit  # Remember profit
                sell = self.data.iloc[self.t, :][self.datacol]
                self.position = 0
        if self.position < 0:
            position_val = (-self.data.iloc[self.t, :][self.datacol] - self.position)
        elif self.position > 0:
            position_val = (self.data.iloc[self.t, :][self.datacol] - self.position)
            # set next time
            # !!! self.t += 1

        result = self.gotnextstate()

        if self.args.stop > 0:
            if self.position < 0:
                lowest_position_val = (-1 * self.position - self.data.iloc[self.t, :][self.highcol])
            elif self.position > 0:
                lowest_position_val = (self.data.iloc[self.t, :][self.lowcol] - self.position)
            else:
                lowest_position_val = 0

            if -1 * self.args.stop > lowest_position_val:
                profit = self.args.stop * (-1.1)
                reward = lowest_position_val
                position_val = 0
                self.position = 0
        result = [position_val + 0.0] + result
            # clipping reward
        if profit != 0:
            reward = profit
        #elif position_val < -self.args.stop / 2:
        #    reward += position_val * 3

        if ticker != self.data.iloc[self.t, :][self.tickercol]:
            for _ in range(self.history_t):  # step(0) - act = 0: stay
                result = [0.0] + self.gotnextstate()
            reward = 0
            position_val = 0
            self.position = 0
            profit = position_val

        return np.array(result), reward, self.done, buy, sell, profit  # obs, reward, done,profit

    def step_old2(self, act):
        datacol = '<CLOSE>'
        tickercol = '<TICKER>'
        datacol_vol = '<VOL>'
        opencol = '<OPEN>'
        highcol = '<HIGH>'
        lowcol = '<LOW>'
        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val = 0
        # act = 0: stay, 1: buy, 2: sell

        ticker = self.data.iloc[self.t, :][tickercol]
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
        elif self.position > 0:
            position_val = (self.data.iloc[self.t, :][datacol] - self.position)
            # set next time
        self.t += 1

        if self.args.stop > 0:
            if self.position < 0:
                lowest_position_val = (-1 * self.position - self.data.iloc[self.t, :][highcol])
            elif self.position > 0:
                lowest_position_val = (self.data.iloc[self.t, :][lowcol] - self.position)
            else:
                lowest_position_val = 0
            if -1 * self.args.stop > lowest_position_val:
                profit = self.args.stop * 1.1
                reward = lowest_position_val
                position_val = 0
                self.position = 0
        if ticker != self.data.iloc[self.t, :][tickercol]:
            for _ in range(self.history_t):  # step(0) - act = 0: stay
                self.step(0)
            reward = 0
            position_val = 0
            self.position = 0
            profit = position_val

        # self.position_value = 0
        # for p in self.positions:  # calculate position value
        # self.position_value += (self.data.iloc[self.t, :][datacol] - p)
        self.history.pop(0)
        self.done = True if self.t >= len(self.data) - 1 else False
        # method takes a single argument (index) and removes the element present at that index from the list. ...
        self.history.append(self.data.iloc[self.t, :][datacol] - self.data.iloc[(self.t - 1), :][
            datacol])  # add the diferrence between cureent CLose Value and prior Close Value

        result = [position_val + 0.0]

        if self.args.candlenum:
            result += [self.data.index[self.t].day + 0.0, self.data.index[self.t].dayofweek + 1 + 0.0,
                       (self.data.index[self.t].hour * 60 + self.data.index[self.t].minute - 600) / 5 + 1]

        result += self.history
        if self.args.usevol:
            self.history_vol.pop(0)
            self.history_vol.append(self.data.iloc[self.t, :][datacol_vol] - self.data.iloc[(self.t - 1), :][
                datacol_vol])  # add the diferrence between cureent CLose Value and prior Close Value
            result += self.history_vol

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
            self.high_vol.append(self.data.iloc[self.t, :][highcol] - close_t)  # add the diferrence
            result += self.high_vol

            self.low_vol.pop(0)
            self.low_vol.append(self.data.iloc[self.t, :][lowcol] - close_t)  # add the diferrence
            result += self.low_vol

        # clipping reward
        if profit != 0:
            reward = profit
        elif position_val < -self.args.stop / 2:
            reward += position_val * 3

        return np.array(result), reward, self.done, buy, sell, profit  # obs, reward, done,profit

    def step_old(self, act):
        datacol = '<CLOSE>'
        reward = 0  # stay
        profit = 0
        buy = 0
        sell = 0
        position_val = 0
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
        elif self.position > 0:
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
            reward += position_val * 3
        return np.array([position_val] + self.history), reward, self.done, buy, sell, profit  # obs, reward, done,profit
