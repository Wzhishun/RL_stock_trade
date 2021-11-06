import gym
from gym import spaces
import numpy as np
import random
import pandas as pd
import pickle
import time
import datetime
import plotly
import plotly.graph_objs as go

np.random.seed(1)
from sklearn import preprocessing

class StockEnv(gym.Env):
    def __init__(self, INITIAL_ACCOUNT_BALANCE=1000000, type='train', train_size=480, ran=0):
        self.ran = ran

        start = time.clock()
        super(StockEnv, self).__init__()
        self.type = type
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.train_size = train_size
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(30, 6),  # 48+30+30
                                            dtype=np.float32)

        self.stock_list = ['002230', '002405', '002253', '000333', '002050', '002242',
                           '603288', '000895', '600305', '600030', '000776', '000686',
                           '600276', '002001', '600267','510050', '510300', '510500']
        self.type_list = ["SZ", "SZ", "SZ", "SZ", "SZ", "SZ",
                          "SH", "SZ", "SH", "SH", "SZ", "SZ",
                          "SH", "SZ", "SH", "ETF", "ETF", "ETF"]
        self.total_stock_1d_list = []
        self.total_stock_5m_list = []
        self.total_stock_1w_list = []
        self.stock_data = pd.read_csv(
            "./StockData/" + self.type_list[self.ran] + self.stock_list[self.ran] + ".csv",
            index_col=0)  # 读取个股数据
        self.total_stock_1d_list.append(self.stock_data)
        end = time.clock()
        print('Running time: %s Seconds' % (end - start))

    def _next_observation(self):

        array_1d = preprocessing.minmax_scale(
            np.array(self.total_stock_1d_list[0].iloc[self.current_step - 29:self.current_step + 1, 0:6]),
            feature_range=(-1, 1), axis=0, copy=False)

        obs = array_1d
        return obs

    def reset(self):
        '''
        self.total_asset
        self.stock_num
        self.avg_price
        self.cash
        self.current_step
        '''

        self.stock_choose = self.stock_list[self.ran]
        self.price_list = self.total_stock_1d_list[0].open.to_list()
        self.range1 = 480
        self.range2 = len(self.price_list) - 480 - 2
        self.range3 = len(self.price_list) - 240 - 2

        if self.type == 'train':
            self.current_step = random.randint(self.range1, self.range2)
        elif self.type == 'test':
            self.current_step = self.range2

        self.total_asset = self.INITIAL_ACCOUNT_BALANCE
        self.stock_num = 0
        self.cash = self.INITIAL_ACCOUNT_BALANCE
        self.init_step = self.current_step
        self.init_price = self.price_list[self.current_step + 1]
        self.last_price = self.init_price
        self.this_price = self.init_price
        self.current_price = self.init_price
        self.done = False
        self.reward_sum = 0
        # plot add
        self.action_list = []
        self.date_list = []
        self.stock_num_list = []
        self.reward_list = []
        self.asset_match_list = []
        self.price_match_list = []
        self.fee_sum = 0

        self.percent_list = []

        self.asset_rate_list = []
        self.price_rate_list = []
        return self._next_observation()

    def _take_action(self, action):
        self.last_asset = self.total_asset
        action_type = action[0]
        self.last_price = self.this_price
        self.current_price = self.price_list[self.current_step + 1]
        self.this_price = self.current_price
        self.total_asset = self.current_price * self.stock_num + self.cash
        # plot add
        self.action_list.append(action_type)
        self.date_list.append(self.stock_data.index[self.current_step])
        self.asset_match_list.append(self.total_asset / self.INITIAL_ACCOUNT_BALANCE)
        self.price_match_list.append(self.current_price / self.init_price)
        self.stock_num_list.append(self.stock_num)
        if action_type > 0:
            buy_num = int(self.cash * action_type / (self.current_price * 1.00025 * 100)) * 100
            if buy_num * self.current_price >= 10000:
                self.cash = self.cash - buy_num * self.current_price * 1.00025
                self.fee_sum = self.fee_sum + buy_num * self.current_price * 0.00025
            elif buy_num * self.current_price < 10000 and buy_num > 0:
                self.cash = self.cash - buy_num * self.current_price - 5
                self.fee_sum = self.fee_sum + 5
            self.stock_num = self.stock_num + buy_num
            self.total_asset = self.cash + self.stock_num * self.current_price
        elif action_type <= 0:
            sell_num = int(self.stock_num * abs(action_type) / 100) * 100
            if sell_num * self.current_price >= 10000:
                self.cash = self.cash + sell_num * self.current_price * (1 - 0.00125)
                self.fee_sum = self.fee_sum + sell_num * self.current_price * 0.00125
            elif sell_num * self.current_price < 10000 and sell_num > 0:
                self.cash = self.cash + sell_num * self.current_price * (1 - 0.001) - 5
                self.fee_sum = self.fee_sum + sell_num * self.current_price * 0.001 + 5
            self.stock_num = self.stock_num - sell_num
            self.total_asset = self.cash + self.stock_num * self.current_price
        self.asset_rate_list.append(round(self.total_asset / self.last_asset - 1, 4))
        self.price_rate_list.append(round(self.current_price / self.last_price - 1, 4))

    def step(self, action):
        if self.current_step >= self.init_step + self.train_size or self.total_asset <= 0:
            self.done = True
        self._take_action(action)
        self.current_step += 1
        reward = ((self.total_asset - self.last_asset) / self.last_asset -
                  (self.this_price - self.last_price) / self.last_price) * 100
        self.reward_sum = self.reward_sum + reward
        # plot add
        self.percent_list.append(round(self.stock_num*self.current_price/self.total_asset,4))
        self.reward_list.append(self.reward_sum)
        obs = self._next_observation()
        return obs, reward, self.done, {}

    def render(self,mode="human"):
        step = 479
        if step == self.train_size - 1:
            table_trace1 = go.Table(
                domain=dict(x=[0.7, 1], y=[0.7, 1]),
                columnwidth=[20, 20, 20, 20, 20],
                columnorder=[0, 1, 2, 3, 4, 5],
                header=dict(height=50,
                            values=[['<b>开始日期</b>'], ['<b>收益率</b>'], ['<b>基准收益率</b>'],
                                    ['<b>超额收益率</b>'], ['<b>手续费率</b>']],
                            line=dict(color='#DCDCDC'),
                            align=['left'] * 7,

                            font=dict(color=['#00ccff'] * 8, size=14),
                            fill=dict(color='#000000')),
                cells=dict(values=[self.date_list[0],
                                   round(self.asset_match_list[-1] - 1, 4),
                                   round(self.price_match_list[-1] - 1, 4),
                                   round(self.asset_match_list[-1] - self.price_match_list[-1], 4),
                                   round(self.fee_sum / self.INITIAL_ACCOUNT_BALANCE, 4)],
                           line=dict(color='#DCDCDC'),
                           align=['left'] * 8,
                           font=dict(color=['#00ccff'] * 8, size=12),
                           format=[None] + [", .2f"] * 6,
                           prefix=[None] * 7,
                           suffix=[None] * 7,
                           height=49,
                           fill=dict(color=['#000000', '#000000']))
            )
            trace1 = go.Scatter(
                x=self.date_list,
                y=self.price_match_list,
                name='bentchmark',
                line=dict(width=2, color='#00ccff'),
            )
            trace2 = go.Scatter(
                x=self.date_list,
                y=self.asset_match_list,
                name='net_value',
                line=dict(width=2, color='red'),
            )
            amount_scatter = go.Scatter(x=self.date_list,
                                        y=self.stock_num_list,
                                        name='long position',
                                        line=dict(color='rgba(255,188,95,0.4)'),
                                        mode='lines',
                                        fill='tozeroy',
                                        fillcolor='rgba(255,188,95,0.2)',
                                        xaxis='x',
                                        yaxis='y2',
                                        opacity=0.6)

            trace3 = go.Scatter(
                x=self.date_list,
                y=self.reward_list,
                name='reward',
                line=dict(color='rgba(0,204,255,0.4)'),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(0,204,255,0.2)',
                xaxis='x3',
                yaxis='y3',
                opacity=0.6)
            trace4 = go.Bar(
                x=self.date_list,
                y=self.action_list,
                name="action",
                xaxis='x4',
                yaxis='y4'
            )

            layout = go.Layout(
                title="股票交易框架 TRPO" + "   初始资金：" + str(self.INITIAL_ACCOUNT_BALANCE) + "  标的:" + self.stock_choose,
                titlefont={'size': 22, 'color': '#00ccff'},
                legend={'x': 1.1},
                width=1900,
                height=1100,
                # Top left
                xaxis=dict(
                    domain=[0, 0.67], type="category",
                    showgrid=False, zeroline=False, anchor='y1',
                ),

                yaxis=dict(

                    domain=[0.4, 1],
                    titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
                    showgrid=False, position=0.03, zeroline=False,
                    anchor='free', side="left",
                ),
                yaxis2=dict(overlaying='y', side='right',
                            titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
                            showgrid=False, position=0.03, zeroline=False,
                            anchor='free'),
                xaxis4=dict(
                    domain=[0, 0.67], type="category", showticklabels=False,
                    showgrid=False, zeroline=False, anchor='y4',
                ),
                yaxis4=dict(

                    domain=[0.17, 0.32],
                    titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
                    showgrid=False, position=0.03, zeroline=False,
                    side="right", anchor='x4',
                ),
                xaxis3=dict(
                    domain=[0.7, 1], type="category", showticklabels=False,
                    showgrid=False, zeroline=False,
                ),

                yaxis3=dict(

                    domain=[0.45, 0.7],
                    titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
                    showgrid=False, position=0.03, zeroline=False,
                    side="left", anchor='x3',
                ),

                paper_bgcolor='#000000',
                plot_bgcolor='#000000'
            )
            data = [table_trace1, trace1, trace2, trace4, amount_scatter, trace3]
            time_now = datetime.datetime.today()
            time_now = str(time_now).replace(" ", "_")
            time_now = str(time_now).replace(".", "_")
            time_now = str(time_now).replace(":", "_")
            time_now = str(time_now).replace("-", "_")

            fig = go.Figure(data=data, layout=layout)

            plotly.offline.plot(fig, filename="./result_figure/" + self.stock_choose + time_now + '.html',
                                auto_open=False)
            # todo:
            import math
            def beta(list_stock, list_index):
                return (np.cov(list_stock, list_index))[0][1] / np.var(list_index)

            # 计算夏普比率
            def sharp_ratio(list, rf=0.02, days=240):
                a = pd.DataFrame(list) - rf / days
                return ((a.mean() * math.sqrt(days)) / a.std())[0]

            def max_drawdown(return_list):
                try:
                    index_j = np.argmax(np.maximum.accumulate(return_list) - return_list)  # 结束位置
                    index_i = np.argmax(return_list[:index_j])  # 开始位置
                    d = return_list[index_j] - return_list[index_i]  # 最大回撤
                except BaseException as e:
                    d = 0
                return d

            self.sharp_ratio = round(sharp_ratio(self.asset_rate_list), 4)
            self.beta = round(beta(self.asset_rate_list, self.price_rate_list), 4)
            self.max_drawdown = abs(round(max_drawdown(np.array(self.asset_rate_list)), 4))
            # round(self.fee_sum / self.INITIAL_ACCOUNT_BALANCE, 4),
            print(self.stock_choose,
                  round(self.price_match_list[-1] - 1, 4),
                  round(self.asset_match_list[-1] - 1, 4),
                  self.sharp_ratio,
                  self.beta,
                  self.max_drawdown,
                  round(beta(self.price_rate_list, self.price_rate_list), 4),
                  round(sharp_ratio(self.price_rate_list), 4),
                  round(max_drawdown(self.price_rate_list), 4))
        return self.asset_match_list, self.price_match_list, self.percent_list,self.sharp_ratio,self.beta,self.max_drawdown
