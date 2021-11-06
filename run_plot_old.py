# import seaborn as sns

import StockEnvPlot
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np

stock_list = ['002230', '002405', '002253', '000333', '002050', '002242',
              '603288', '000895', '600305', '600030', '000776', '000686',
              '600276', '002001', '600267','510050', '510300', '510500']
type_list = ["SZ", "SZ", "SZ", "SZ", "SZ", "SZ",
             "SH", "SZ", "SH", "SH", "SZ", "SZ",
             "SH", "SZ", "SH", "ETF", "ETF", "ETF"]
policy_type_list = ["DensePolicy", "DenseGarchPolicy",
                    "MultiFrequencyParallelDensePolicy",
                    "MultiFrequencyParallelDenseGarchPolicy"]# "DenseParallel",
ran = 6  # 0--14
test = "test1"  # 测试序号，test1,2,3,4...
seed_num = 0  # 随机数种子，0,1,2...
profit_array = []
percent_array=[]
for j in range(0, 1):
    profit = []
    base = []
    percent=[]
    policy_type_number = j  # 政策网络选择 0，1，2，3
    if j==0:
        env = Env.StockEnvPlot.StockEnv(
            type='test', ran=ran)
        env = DummyVecEnv([lambda: env])
        model = PPO2.load("./ModelSave/" +
                          policy_type_list[policy_type_number] + "/" + "PPO2" + type_list[ran]
                          + stock_list[ran] + "_" + policy_type_list[policy_type_number] + "_" +
                          test + "_seed" + str(seed_num) + "_1986560_steps.zip",
                          env, seed=0)
    if j == 1:
        env = Env.StockGarchEnvPlot.StockEnv(
            type='test', ran=ran)
        env = DummyVecEnv([lambda: env])
        model = PPO2.load("./ModelSave/" +
                          policy_type_list[policy_type_number] + "/" + "PPO2" + type_list[ran]
                          + stock_list[ran] + "_" + policy_type_list[policy_type_number] + "_" +
                          test + "_seed" + str(seed_num) + "_1986560_steps.zip",
                          env, seed=0)
    if j == 2:
        env = Env.StockParallelEnvPlot.StockEnv(
            type='test', ran=ran)
        env = DummyVecEnv([lambda: env])
        model = PPO2.load("./ModelSave/" +
                          policy_type_list[policy_type_number] + "/" + "PPO2" + type_list[ran]
                          + stock_list[ran] + "_" + policy_type_list[policy_type_number] + "_" +
                          test + "_seed" + str(seed_num) + "_test_best_model.zip",
                          env, seed=0)
    if j == 3:

        env = Env.StockParallelGarchEnvPlot.StockEnv(
            type='test', ran=ran)
        env = DummyVecEnv([lambda: env])
        model = PPO2.load("./ModelSave/" +
                          policy_type_list[policy_type_number] + "/" + "PPO2" + type_list[ran]
                          + stock_list[ran] + "_" + policy_type_list[policy_type_number] + "_" +
                          test + "_seed" + str(seed_num) + "_test_best_model.zip",
                          env, seed=0)

    # PPO2:
    # 0 _1986560 _1986560 _1679360 _test_best_model
    # 1 _1986560 _1986560 _1679360 _test_best_model
    # 2 _1986560 _1986560 _1679360 _test_best_model
    # 3 _1986560 _test_best_model _1679360 _test_best_model
    # 4 _1986560 _1986560 _test_best_model _test_best_model 3，1
    # 5 _1986560 _1986560 _test_best_model _test_best_model
    # 6
    # 7
    # 8
    # 9
    #10
    #11
    #12
    #13
    #14
    #15
    #16
    #17
    #18



    print(stock_list[ran])
    obs = env.reset()
    for i in range(480):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i == 479:
            t = env.render()
            profit.append(t[0])
            base.append(t[1])
            percent.append(t[2])
        env.close()

    env.close()
    profit = np.array(profit)
    profit_array.append(profit)
    percent = np.array(percent)
    percent_array.append(percent)
base = np.array(base)

policy_type_list2 = ["DNN", "DNN-GARCH", "MCT", "MCTG"]
# print(profit, base)

import pandas as pd
pd.DataFrame(base.reshape(1,480)).to_csv("base.csv")
pd.DataFrame(np.array(profit_array).reshape(4,480)).to_csv("profit_array.csv")
pd.DataFrame(np.array(percent_array).reshape(4,480)).to_csv("percent_array.csv")
'''
ax = plt.subplot(1, 1, 1)
ax.set_title(type_list[ran]+ stock_list[ran])
ax.set_xlabel('Step')
ax.set_ylabel('Net Value')
color=['Greys','g','pink','r']
sns.tsplot(data=base, time=np.arange(0, base.shape[1]), ax=ax, color='b',condition='B&H')
for i in range(0,4):
    sns.tsplot(data=profit_array[i], time=np.arange(0, profit_array[i].shape[1]), ax=ax,
               condition=policy_type_list2[i],color=color[i])
# seborn绘图
plt.show()'''

'''import plotly
import plotly.graph_objs as go
import datetime
trace1 = go.Scatter(x=np.arange(0, len(base)), y=base, name='B & H', line=dict(width=2, color='#00ccff'), xaxis='x', yaxis='y1')
trace2 = go.Scatter(x=np.arange(0, len(base)), y=profit_array[0], name='B & H', line=dict(width=2, color='#00ccff'), xaxis='x', yaxis='y1')
trace3 = go.Scatter( x=np.arange(0, len(base)), y=profit_array[1], name='B & H', line=dict(width=2, color='#00ccff'), xaxis='x', yaxis='y1')
trace4 = go.Scatter( x=np.arange(0, len(base)), y=profit_array[2], name='B & H', line=dict(width=2, color='#00ccff'), xaxis='x', yaxis='y1')
trace5 = go.Scatter(
    x=np.arange(0, len(base)),
    y=profit_array[3],
    name='B & H',
    line=dict(width=2, color='#00ccff'),
    xaxis='x',
    yaxis='y1')
trace11 = go.Bar(
    x=np.arange(0, len(base)),
    y=profit_array[3],
    name="action",
    xaxis='x',
    yaxis='y2')
trace1bar = go.Bar(
    x=np.arange(0, len(base)),
    y=percent_array[0],
    name="action",
    xaxis='x',
    yaxis='y2')
trace2bar = go.Bar(
    x=np.arange(0, len(base)),
    y=percent_array[1],
    name="action",
    xaxis='x',
    yaxis='y2')
trace3bar = go.Bar(
    x=np.arange(0, len(base)),
    y=percent_array[2],
    name="action",
    xaxis='x',
    yaxis='y2')
trace4bar = go.Bar(
    x=np.arange(0, len(base)),
    y=percent_array[3],
    name="action",
    xaxis='x',
    yaxis='y2')
layout = go.Layout(
    width=1000,
    height=2000,
    # Top left
    xaxis=dict(
        domain=[0, 1], type="category",
        showgrid=False, zeroline=False, anchor='y1',
    ),
    yaxis=dict(
        domain=[0.4, 1],
        titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
        showgrid=False, position=0.03, zeroline=False,
        anchor='free', side="left",
    ),

    yaxis2=dict(domain=[0, 0.4],
                titlefont={'color': '#FFFFFF'}, tickfont={'color': '#FFFFFF'},
                showgrid=False, position=0.03, zeroline=False,
                anchor='free'),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000'
)
data = [trace1, trace2, trace3, trace4, trace5]
time_now = datetime.datetime.today()
time_now = str(time_now).replace(" ", "_")
time_now = str(time_now).replace(".", "_")
time_now = str(time_now).replace(":", "_")
time_now = str(time_now).replace("-", "_")
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename=time_now + '.html', auto_open=False)
'''