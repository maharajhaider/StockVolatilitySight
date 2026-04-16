  
**The Problem To Solve**

Market volatility measures how much asset prices fluctuate over time and it is commonly used as an indicator of financial uncertainty and risk. Low volatility periods typically correspond to relative stable market conditions with small fluctuation of asset prices while high volatility periods are associated with large price swings and less predictable behavior. Therefore, being able to accurately predict market volatility helps investors, traders and financial systems estimate future risk and make better decisions under uncertainty. 

For this project, we focus on the problem of predicting future stock market volatility based on historical market data. In particular, we will investigate whether regime-aware models that distinguish between low volatile and high volatile market conditions can improve predictive accuracy on the stock market volatility compared with a baseline model.

**Method**

Cont (2007) identifies the volatility clustering property; the scale of price changes remains similar over a period. From this idea, we propose a hybrid model which first employs a hidden markov model (HMM) to characterize our inputs into either a calm or volatile regime. This regime information will be used to separate our original dataset, which will allow each regime specific dataset to be separately trained using an LSTM model. Finally, our output, the target stock volatility will be a weighted average of the probability of each regime given inputs, and the stock volatility output from the LSTM. The target volatility is the aggregate volatility over a rolling 21 day window, specifically:

## t=1m-1j=1m(rt+j-rt)2

Where m is the rolling window period, and r is the observed log return on a specific day subtracted by the mean return over that window as an adaptation from equations derived by Andersen and Bollerslev (1998) to reflect the importance of the mean for non-high frequency data. Our model will target the volatility of distinct ETFs (SPY, XEQT) and individual hand picked stocks (MAG7). The process will be detailed end-to-end in 5 separate phases: feature selection and data engineering, regime detection (HMM), LSTM training, ensemble prediction, evaluation.

Some inputs are intuitive; we plan to train our models on 15-20 years of daily OHLCV (Open, High, Low, Close, Volume) data. Comparing relative intra and inter day changes on these parameters will of course help our model determine the volatility of our data. We also suggest including secondary market indicators such as relative put and call volume as they are found linked to stock volatility by Ho et al. (2012). Finally, as per Gupta et al. (2023) there is significant value in non-price sentiment related inputs in predicting volatility \- for this we will use the AAII weekly sentiment surveys. After some basic feature selection to avoid correlated redundant features, we will engineer the inputs to make them more interpretable. This involves simple transforms, such as calculating daily log returns, absolute returns, intraday ranges, log volume, and other measures that help manipulate the data into more suitable forms. We must also note our data will be split chronologically into train, validation, and test sets to avoid temporal leakage in our data (ie. training will be 2005-2015, validation 2015-2020, remaining is test). The validation set will be used for hyper-parameter tuning for both the HMM and LSTM.

With the data in place, we will train our HMM through the hmmlearn python library. Our current idea is to create a GaussianHMM with 2 hidden states the model will learn. There is however additional work that needs to be done to determine if a Gaussian distribution is truly appropriate through analysis on data normality. Furthermore, Zhang et al. (2019) found GaussianMixtures as having higher log-likelihood with a lower BIC  when compared to a simple Gaussian. Thus, we suggest comparison between the simple and mixture versions. Additionally, there are nuances in how we set up the HMM. Specifically, higher-order and multi-layer markov chains boast higher accuracies at the cost of complexity (Zhang et al., 2019; Lee, 2017). We will explore higher-order markov models as a means of capturing longer term regime transitions if time permits. With the HMM trained we will utilize Viterbi decoding to assign the most likely sequence of regimes, and plot the resulting state as a sanity check against expected historical trends.

Considering different regimes are likely to have different return and volatility distributions, the HMM outputs will be used to separate data into one of the two regimes before training individual LSTM models on each. Each LSTM will train on a 21-day rolling window of data, outputting a predicted volatility. In addition, we propose a baseline LSTM trained on the entire dataset as a comparison measure.

The HMM per-regime probability will be used to produce a final weighted prediction of volatility. With our output as:   
yfinal \= P(st \= calm | O(1:T)) \*ycalm \+ P(st \= volatile | O(1:T)) \*yvolatile     
Where the regime probabilities are obtained via predict\_proba() which utilizes the forward-backward algorithm to predict soft probabilities. We have also considered using a final learned gating network to go beyond simple weighting in cases where the scale of volatility between the two states is so different that the relative error would be large for smaller volatility regimes, however this is currently considered out of scope. 

For evaluation purposes, we will compare our model against the baseline LSTM using MSE, MAE, and RMSE.  We are also considering comparing our model to a simpler model for reference purposes, for example a basic tree based regression or neural net (Christensen et al., 2023). 

**Contributions**

The project’s primary contribution will be to discover if using a hidden markov model to separate the data into two different regimes (Volatile Market and Calm Market) followed by deep learning yields better accuracy than an end to end deep learning model. 

In the past, there has been significant research into financial market prediction. Zhang et al. (2019) looked at improving stock market price trend prediction using HMM. On the other hand, Fischer and Krauss (2018) looked into using LSTM for financial market prediction.  They do not look at combining the methods to explore if that adds value. In addition, Vallarino (2025) investigates using a RNN for volatile stocks and linear regression for stable stocks. They do not look at market wide volatility when determining the model to use. The market-wide volatility is often asymmetric where the markets tend to change  in a volatile market a lot quicker than they do in a calm market. Given this asymmetry,  it would be beneficial to see if separating the time period into the two regimes improves volatility predictions. 

Jiang et al. (2023) motivates using HMM-ALSTM where the HMM states are added as an additional feature to LSTM. We instead plan to partition the data and  train two separate LSTM, one for each state to try to yield better results. This distinct architecture will allow each LSTM to specialize in their respective regime’s volatility. Comparing it to the baseline that uses LSTM end to end will help us explore how much value the regime separation by HMM adds.

**References**

Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International economic review*, 885-905.

Christensen, K., Siggaard, M., & Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics*, *21*(5), 1680-1727.

Cont, R. (2007). Volatility clustering in financial markets: empirical facts and agent-based models. In *Long memory in economics* (pp. 289-309). Berlin, Heidelberg: Springer Berlin Heidelberg.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European journal of operational research*, *270*(2), 654-669.

Gupta, R., Nel, J., & Pierdzioch, C. (2023). Investor confidence and forecastability of US stock market realized volatility: Evidence from machine learning. *Journal of Behavioral Finance*, *24*(1), 111-122.

Ho, K. Y., Zheng, L., & Zhang, Z. (2012). Volume, volatility and information linkages in the stock and option markets. *Review of Financial Economics*, *21*(4), 168-174.

Jiang, J., Wu, L., Zhao, H., Zhu, H., & Zhang, W. (2023). Forecasting movements of stock time series based on hidden state guided deep learning approach. *Information Processing & Management*, *60*(3), 103328\.

Lee, J. (2017). Multi-level hidden Markov model and ULSTM network \[Doctoral dissertation, Seoul National University\]. 

Vallarino, D. (2025). Adaptive Market Intelligence: A Mixture of Experts Framework for Volatility-Sensitive Stock Forecasting. *arXiv preprint arXiv:2508.02686*.

Zhang, M., Jiang, X., Fang, Z., Zeng, Y., & Xu, K. (2019). High-order Hidden Markov Model for trend prediction in financial time series. *Physica A: Statistical Mechanics and its Applications*, *517*, 1-12.

                                                             
