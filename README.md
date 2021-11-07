# Test1106
Test1106
You can see the lab in wandb file. It is public.
https://wandb.ai/prcwzs/Test1106/workspace?workspace=user-prcwzs

Due to my poor computer power, I just train 15 stocks not all securities.

I use PPO (Proximal Policy Optimization) + MLP (Multilayer Perceptron) to create a RL (Reinforcement Learning) agent to trade stock you provided. The RL trading decision marking process can be described as follows:

I do not choose to use traditional indiactor to trade stock cause I do think it will lead to the result of strategy homogeneity. However, using machine learning methods will face black box problems, which may cause systemic risks.StockEnv.py is a Chinese stock trading env where transaction fee is 0.125%. StockPlotEnv.py is major in plot the trading image. quant.py is in charge of data process. PPO_run1.py is tha main fuction of running RL process. Test.zip includes other train models and data. run_plot.py is a plot process.


packages:

pickle

tensorflow-1.15

stable-baselines

pickle

scikit-learn

gym

In the process of data processing, the last price and volume based on the current 10 trading days are used as the agent's observation data, and trading decisions are made (-1 to 1, representing the ratio of selling or buying and trading). The training set is the data from the initial day to the last 480 trading day, and the test set is the last 480 trading days.
The agent uses the excess return rate as its reward function, and uses the return rate, beta, Sharpe ratio and maximum drawdown for the indicators analyzed by the reinforcement learning method.
![8f2cb5b0e82341a2294bac58d90a84f](https://user-images.githubusercontent.com/49648647/140630360-c8b967a2-4930-4c89-9035-e5496566c8c8.png)

As shown in the following chart, the red line is the revenue curve of our agent, and the blue line is the benchmark revenue curve. The orange bars are the positions of our reinforcement learning agent. The pink bars are the trading actions of the reinforcement learning agent. The line graph on the right is the reward function of the reinforcement learning agent.

![image](https://user-images.githubusercontent.com/49648647/140601769-ee73b35d-a663-40ba-b87b-e493f20e74c0.png)

![image](https://user-images.githubusercontent.com/49648647/140602096-d8e79621-7a8f-41a2-8a43-47bfb3e7ad58.png)

![image](https://user-images.githubusercontent.com/49648647/140602112-12a0c9cb-b346-49cf-8bab-21540fafd3e3.png)

Although the performance of the agent varies greatly among multiple stocks, and it has not achieved particularly good returns, it can be improved based on the existing model. Such as using more suitable algorithms (LSTM, ATTENTION), richer data (open, high, low, close, KDJ, MACD, RSI...) and other sentiment indicators, or as an auxiliary decision-making for traders.

