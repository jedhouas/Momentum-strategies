# -*- coding: utf-8 -*-
"""
Created on Sun May 16 10:55:21 2021

@author: jhouas
"""

import math
import numpy as np
import pandas as pd
import blpapi
import pdblp
import matplotlib.pyplot as plt
import datetime
from datetime import date

con = pdblp.BCon(debug=True, port=8194, timeout=5000)
con.start()
                
list_of_stocks = ['RJHI AB Equity' ,'SNB AB Equity' ,'SABIC AB Equity' ,'STC AB Equity' ,'RIBL AB Equity' ,'BSFR AB Equity' ,'SABB AB Equity' ,'ALINMA AB Equity' ,'ALMARAI AB Equity' ,'JOMAR AB Equity' ,'SECO AB Equity' ,'ALBI AB Equity' ,'MAADEN AB Equity' ,'JARIR AB Equity' ,'SAFCO AB Equity' ,'EEC AB Equity' ,'SAVOLA AB Equity' ,'SIPCHEM AB Equity' ,'ARNB AB Equity' ,'KAYAN AB Equity' ,'APPC AB Equity' ,'YANSAB AB Equity' ,'BJAZ AB Equity' ,'SIIG AB Equity' ,'SULAIMAN AB Equity' ,'NIC AB Equity' ,'MOUWASAT AB Equity' ,'NSCSA AB Equity' ,'MCDCO AB Equity' ,'ALARKAN AB Equity' ,'AOTHAIM AB Equity' ,'SACCO AB Equity' ,'BUPA AB Equity' ,'RESEARCH AB Equity' ,'ZAINKSA AB Equity' ,'SIBC AB Equity' ,'EXTRA AB Equity' ,'SEERA AB Equity' ,'TAWUNIYA AB Equity' ,'YACCO AB Equity' ,'SOCCO AB Equity' ,'PETROCH AB Equity' ,'YNCCO AB Equity' ,'TAIBA AB Equity' ,'EMAAR AB Equity' ,'ALMRAKEZ AB Equity' ,'QACCO AB Equity' ,'SPIMACO AB Equity' ,'ALDREES AB Equity' ,'PETROR AB Equity' ,'ARCCO AB Equity' ,'CITYC AB Equity' ,'CATERING AB Equity' ,'NADEC AB Equity' ,'ADCO AB Equity' ,'NAJRAN AB Equity' ,'ALCO AB Equity' ,'SISCO AB Equity' ,'EACCO AB Equity' ,'SCERCO AB Equity' ,'SADAFCO AB Equity' ,'SGS AB Equity' ,'CHEMICAL AB Equity' ,'BUDGET AB Equity' ,'NORTHCEM AB Equity' ,'ARCCI AB Equity' ,'ALHAMMAD AB Equity' ,'JADCO AB Equity' ,'DUR AB Equity' ,'BINDAWOO AB Equity' ,'SARCO AB Equity' ,'NGIC AB Equity' ,'SAPTCO AB Equity' ,'JADWAREI AB Equity' ,'MAHARAH AB Equity' ,'SCH AB Equity' ,'DALLAH AB Equity' ,'YAMAMAH AB Equity' ,'AMLAK AB Equity' ,'BAWAN AB Equity' ,'ALASEEL AB Equity' ,'LEEJAM AB Equity' ,'JOUF AB Equity' ,'SACO AB Equity' ,'HCC AB Equity' ,'ALANDALU AB Equity' ,'SAIC AB Equity' ,'TACCO AB Equity' ,'CHEMANOL AB Equity' ,'ALHOKAIR AB Equity' ,'UACC AB Equity' ,'ALRAJHI AB Equity' ,'ALBABTAI AB Equity' ,'SRECO AB Equity' ,'AADC AB Equity' ,'MEH AB Equity' ,'FPCO AB Equity' ,'ATAA AB Equity' ,'RYDREIT AB Equity' ,'ZIIC AB Equity' ,'KINGDOM AB Equity' ,'CARE AB Equity' ,'ATTMCO AB Equity' ,'MIS AB Equity' ,'MEDGULF AB Equity' ,'HB AB Equity' ,'NCLE AB Equity' ,'KEC AB Equity' ,'SHAKER AB Equity' ,'DERAYAHR AB Equity' ,'MALATH AB Equity' ,'SFICO AB Equity' ,'EIC AB Equity' ,'WALAA AB Equity' ,'AHFCO AB Equity' ,'HERFY AB Equity' ,'ZOUJAJ AB Equity' ,'AWPT AB Equity' ,'AXA AB Equity' ,'SVCP AB Equity' ,'MEPC AB Equity' ,'ASTRA AB Equity' ,'GIZACO AB Equity' ,'NGCO AB Equity' ,'SAUDIRE AB Equity' ,'SCACO AB Equity' ,'BATIC AB Equity' ,'ANAAM AB Equity' ,'ALKHABEE AB Equity' ,'TAACO AB Equity' ,'LAZURDE AB Equity' ,'TAPRCO AB Equity' ,'ASLAK AB Equity' ,'NAMA AB Equity' ,'BCI AB Equity' ,'MUSHREIT AB Equity' ,'SAAC AB Equity' ,'SHIELD AB Equity' ,'JAZTAKAF AB Equity' ,'SPPC AB Equity' ,'ALKHLEEJ AB Equity' ,'ALETIHAD AB Equity' ,'ABOMOATI AB Equity' ,'GACO AB Equity' ,'AATD AB Equity' ,'REDSEA AB Equity' ,'APCO AB Equity' ,'EAT AB Equity' ,'SIDC AB Equity' ,'MESC AB Equity' ,'TAKWEEN AB Equity' ,'SPM AB Equity' ,'SSP AB Equity' ,'THEEB AB Equity' ,'AMANA AB Equity' ,'BAAZEEM AB Equity' ,'FIPCO AB Equity' ,'ASACO AB Equity' ,'ALLIANZ AB Equity' ,'ARABSEA AB Equity' ,'AICC AB Equity' ,'SIECO AB Equity' ,'NMMCC AB Equity' ,'SADR AB Equity' ,'ACIG AB Equity' ,'SAGR AB Equity' ,'SMARKETI AB Equity' ,'BONYAN AB Equity' ,'UCA AB Equity' ,'SEDCO AB Equity' ,'WATAN AB Equity' ,'ABDICO AB Equity' ,'NASEEJ AB Equity' ,'MEFIC AB Equity' ,'TECO AB Equity' ,'ALABDUL AB Equity' ,'SALAMA AB Equity' ,'ENAYA AB Equity' ,'JADWA AB Equity' ,'RAYDAN AB Equity' ,'OASIS AB Equity' ,'ACE AB Equity' ,'BURUJ AB Equity' ,'MAATHER AB Equity' ,'ATC AB Equity' ,'SWICORP AB Equity' ,'TALEEM AB Equity' ,'SAICO AB Equity' ,'ALKATHIR AB Equity' ,'GULFUNI AB Equity' ,'ALINMATO AB Equity' ,'ALAHLI AB Equity' ,'ALOMRAN AB Equity' ,'ALALAMIY AB Equity' ,'MULKIA AB Equity' ,'SABBT AB Equity' ,'ALJAZIRA AB Equity' ,'GGCI AB Equity' ,'MASHAAR AB Equity']

res_list_stocks = 
start_date = 20100101
end_date = 20210101

#data = con.bdh(list_of_stocks, ['PX_LAST', 'VOLUME'], '20150629', '20200630', longdata=True)
def get_stocks_data(list_of_stocks, start_date, end_date, data_type='PX_LAST'):
    data_by_stock = []    
    #for x in list_of_stocks:
    #    data_by_stock.append(data[data['ticker'] == x].reset_index())
    
    for x in list_of_stocks:
        data = con.bdh(x, data_type, start_date, end_date, longdata=True)
        data.index = data.date
        data = data.drop(['date' , 'ticker', 'field'], axis=1)
        data = data.rename(columns={'value':x})
        data_by_stock.append(data)
        
    data = pd.concat(data_by_stock, axis=1)
    data.dropna(axis=1, inplace=True)
    return(data)

data = get_stocks_data(list_of_stocks, start_date, end_date, 'PX_LAST')

def get_logreturns(data):
    returns = np.log(data) - np.log(data.shift(1))
    returns.dropna(inplace=True)
    return(returns)

def get_returns(data):
    returns = data/data.shift(1) - 1
    returns.dropna(inplace=True)
    return(returns)

returns_0 = get_returns(data)

def TSMStrategy(returns, period=1, shorts=False):
    if shorts:
        position = returns.rolling(period).mean().map(
            lambda x: -1 if x <= 0 else 1)
    else:
        position = returns.rolling(period).mean().map(
            lambda x: 0 if x <= 0 else 1)
    performance = position.shift(1) * returns
    return performance


def backtester(returns, map, init=100, rebal=75, shorts= False):
  returns = returns.loc[map.index, : ]
  index = [100]
  i = 0
  while i < len(map):
    N = sum(map.iloc[i,:])
    weight = pd.Series([index[-1]/N for j in range(map.shape[1])], index = map.columns) * map.iloc[i,:]
    i += 1
    k = 1
    while i+1 < len(map) and k <= rebal:
      weight = returns.iloc[i+1,:].apply(lambda x: exp(x)) * weight
      index.append(sum(weight))
      i +=1
      k +=1
  return(pd.Series(index))

def to_datetime(x):
    return(datetime.datetime.strptime(str(x), '%Y%m%d'))

def to_inttime(x):
    return(int(x.strftime('%Y%m%d')))

           
# =============================================================================
# Momentum function calculator and evaluator - Single stock:
# =============================================================================

from scipy.stats import linregress
import seaborn as sns

def momentum_1(x):
    y = np.arange(len(x))
    slope, _, rvalue, _, _ = linregress(y, x)
    return(((1+slope) ** 252)*(rvalue**2))

    
def momentum(data, mom_function=np.mean, period=25):
    returns = np.log(data) - np.log(data.shift(1))
    returns.dropna(inplace=True)
    return(returns.rolling(period).apply(mom_function).dropna())

def mom_eval(list_feats_fun, feats_names_list, period, lags_eval): # list feats starts with the prices ts
    ts = {}
    for i, (f, fun) in enumerate(list_feats_fun):
        ts[feats_names_list[i]] = momentum(f,fun, period)
    returns = get_returns(list_feats_fun[0][0])
    returns = returns.loc[ts['Price'].index, :]
    for l in lags_eval:
        ts["Lagged price t + "+str(l)] = returns.shift(-l)
    dataset = pd.DataFrame(ts)
    corr = dataset.corr()
    heat_map = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
    heat_map.set_size(10,6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    # plots = {}
    # i = 0
    # while i < len(list_feats_fun[0][0])-25:
    #     corr_period = dataset.iloc[i:i+period,:].corr()
    #     for i in range(len(dataset.columns)-1):
    #         for j in range(i+1, len(dataset.columns)):
    #             plot_name = dataset.columns[i]+"_"+dataset.columns[j]
    #             if (plot_name in plots.keys()) == False:
    #                 plots[dataset.columns[i]+"_"+dataset.columns[j]] = [corr_period.iloc[i,j]]
    #             else:
    #                 plots[dataset.columns[i]+"_"+dataset.columns[j]].append(corr_period.iloc[i,j])
    #     i += 25
    #     plt.plot(plots)
    #     plt.show()
        return(corr)

def get_list_fun(x, fun_price, fun_volume, start_date, end_date):
    price = get_stocks_data([x], start_date, end_date, data_type='PX_LAST')
    volume = get_stocks_data([x], start_date, end_date, data_type='VOLUME')
    return([(price, fun_price), (volume, fun_volume)])


def get_correl_indicator(x, fun_price, fun_volume, start_date , end_date, lags_eval,  period):
        
    list_feats_fun = get_list_fun(x, fun_price, fun_volume, to_inttime(to_datetime(start_date) - datetime.timedelta(days=period+5)), to_inttime(to_datetime(end_date) + datetime.timedelta(days=lags_eval[-1])) )
    
    ts = {}
    ts['momPrice'] = momentum(list_feats_fun[0][0],list_feats_fun[0][1], period)[to_datetime(start_date):to_datetime(end_date)]
    ts['momVolume'] = momentum(list_feats_fun[1][0],list_feats_fun[1][1], period)[to_datetime(start_date):to_datetime(end_date)]
        
    returns = get_returns(list_feats_fun[0][0])
    
    for l in lags_eval:
        ts["Lagged price t + "+str(l)] = returns.shift(-l)[to_datetime(start_date):to_datetime(end_date)]
        
    correlation = pd.concat(ts, axis=1).corr()
        
    return(pd.Series([correlation.iloc[1,0], correlation.iloc[2,0], correlation.iloc[3,0], correlation.iloc[4,0], 
                      correlation.iloc[1,2], correlation.iloc[1,3], correlation.iloc[1,4]], name = x,
                     index = ['momPrice_momVolume', 'momPrice_Pricet+5', 'momPrice_Pricet+10', 'momPrice_Pricet+15', 'momVolume_Pricet+5', 
                              'momVolume_Pricet+10' , 'momVolume_Pricet+15']))


test_periods = 50
days = (to_datetime(end_date) - to_datetime(start_date)).days

t = 0 
index = []
corr_bw = ['momPrice_momVolume', 'momPrice_Pricet+5', 'momPrice_Pricet+10', 'momPrice_Pricet+15', 'momVolume_Pricet+5', 
                              'momVolume_Pricet+10' , 'momVolume_Pricet+15']

while t<days:
    start_date_t = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t))
    end_date_t = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t+test_periods))
    for c in corr_bw:
        index.append((str(start_date_t)+" to "+ str(end_date_t), c))
    t += test_periods
        
df = pd.DataFrame(0, index= pd.MultiIndex.from_tuples(index, names=['Period', 'correl_bw']), columns = data.columns)


t = 0

while t < days:
    start_date_t = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t))
    end_date_t = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t+test_periods))
    for x in data.columns:
        corr = get_correl_indicator(x, np.mean, np.mean, start_date_t , end_date_t, lags_eval,  period)
        for c in corr_bw:
            df.loc[(str(start_date_t)+" to "+ str(end_date_t),c), x] = corr[c]

    t += test_periods
        

# =============================================================================
# Eval functions
# =============================================================================

def get_annual_return(index, n_years):
    return(index_price[-1]**(1/n_years)-1)

def get_downside_vol(returns, period):
    return(returns[returns<0].std()*np.sqrt(period))



def get_sharpe_ratio(index, periods=252):
    """
    Create the Sharpe ratio for the strategy, based on a 
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    returns = get_returns(pd.Series(index))
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


def get_max_drawdown(prices, window=252):
    roll_max = prices.rolling(window, min_periods=1).max()
    daily_drawdown = prices/roll_max - 1
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    daily_drawdown.plot()
    return(max_daily_drawdown.min())

def get_transcation_cost(countries, sizes, fee_table, curr_mkt_cp):
    fees = 0
    for i in range(len(countries)):
        fee = fee_table[(fee_table["Market"] == countries.iloc[i]) & (fee_table["Market cap"] < curr_mkt_cp.iloc[i])].iloc[-1,-1]
        fees += fee * sizes.iloc[i]
        #fees += fee * sizes
    return(fees)
        
def get_yearly_return(prices, year):
    return(prices[prices.index.year == year].iloc[-1]/prices[prices.index.year == year].iloc[0] - 1)
    

# =============================================================================
# Time Series Momentum: long rolling avg return higher than Ths* - Max single stock weight at MW*
# =============================================================================

from math import exp

Ths = 0.0042350702728018745
MW = 0.1
rebal = 19
period = 15

index_price = [1]
mapping = returns_0.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()
Allocation['Cash'] = Allocation.iloc[:,0]

mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]

#fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")

transaction_cost_values = []
t = 0
while t < len(mapping):
    if t % rebal == 0:
        if mapping.iloc[t,:].sum() <= 10 :
            Allocation.iloc[t,:-1] = mapping.iloc[t,:]*0.1
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        else:
            Allocation.iloc[t,:-1] = mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
            Allocation.iloc[t,-1] = 0
        perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
        
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
        index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
        transaction_cost_values.append(transaction_cost)
        t +=1
    else:
        Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
        Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
        Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
        t += 1

long_all = [1]
t = 0
Allocation_long = mapping.copy()
while t < len(mapping):
    if t % rebal == 0:
        Allocation_long.iloc[t,:] = 1/(Allocation_long.shape[1])
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:-1] * long_all[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*Allocation_long.iloc[t,:], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*abs(Allocation_long.iloc[t,:] - (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]), fees, mkt_cap)

        long_all.append(perf.sum() + Allocation_long.iloc[t,-1]* long_all[-1] - transaction_cost)
        t +=1
    else:
        Allocation_long.iloc[t,:] = (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]
        Allocation_long.iloc[t,:] = Allocation_long.iloc[t,:] / (Allocation_long.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:] * long_all[-1]
        long_all.append(perf.sum())
        t += 1

tc = pd.Series(transaction_cost_values, index = Allocation.index[[rebal*i for i in range(t//rebal + 1)]])

# =============================================================================
# Cross Sectional Momentum: long top Ths*%  - Min momentum at MW*
# =============================================================================

Ths = 0.00005
MW = 0.1
rebal = 75
period = 50

index_price = [1]
rolling = returns_0.rolling(period).apply(momentum_1).dropna()
mapping_ranking = rolling.apply(pd.Series.nlargest, axis=1, n=10).fillna(0).apply(lambda x: x !=0).astype('int').loc[:,rolling.columns]
mapping = mapping_ranking*mapping_positive
returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()
Allocation['Cash'] = Allocation.iloc[:,0]

#mkt_cap = get_stocks_data(mapping.columns, start_date, end_date, data_type='CUR_MKT_CAP').mean().loc[mapping.columns]

fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")

transaction_cost_values = []

t = 0
while t < len(mapping):
    if t % rebal == 0:
        if mapping.iloc[t,:].sum() <= 10 :
            Allocation.iloc[t,:-1] = mapping.iloc[t,:]*0.1
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        else:
            Allocation.iloc[t,:-1] = mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
            Allocation.iloc[t,-1] = 0
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
        index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
        transaction_cost_values.append(transaction_cost)
        t +=1
    else:
        Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
        Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
        Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum() + Allocation.iloc[t,-1] * index_price[-1])
        t += 1

long_all = [1]
t = 0
Allocation_long = mapping.copy()
while t < len(mapping):
    if t % rebal == 0:
        Allocation_long.iloc[t,:] = 1/(Allocation_long.shape[1])
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:-1] * long_all[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*Allocation_long.iloc[t,:], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*abs(Allocation_long.iloc[t,:] - (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]), fees, mkt_cap)

        long_all.append(perf.sum() + Allocation_long.iloc[t,-1]* long_all[-1] - transaction_cost)
        t +=1
    else:
        Allocation_long.iloc[t,:] = (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]
        Allocation_long.iloc[t,:] = Allocation_long.iloc[t,:] / (Allocation_long.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:] * long_all[-1]
        t += 1
        
tc = pd.Series(transaction_cost_values, index = Allocation.index[[rebal*i for i in range(t//rebal + 1)]])


# =============================================================================
# Cross Validation
# =============================================================================


def model(Ths, rebal, period):
    
    rebal = int(rebal)
    period = int(period)
    
    index_price = [1]
    mapping = returns_0.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    returns = returns_0.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
        
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 0
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1
    return(get_sharpe_ratio(index_price, 252))

from bayes_opt import BayesianOptimization

pbounds = {'Ths': (0.00001, 0.01), 'rebal': (10, 100), 'period': (5,75)}

### Optimization

def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=5,
        n_iter=10,
    )
    return(optimizer.max)

### Sensitivity of solution to params / Stability

optimizer = bayesian_cross_val (model, pbounds)


X = np.linspace(0.8, 1.2, 11)
Y = [[],[],[]]
for j, p in enumerate(optimizer['params'].keys()):
    for i in np.linspace(0.8, 1.2, 11):
        test =  list(optimizer['params'].values())
        test[j] = i * test[j]
        Y[j].append(model(*test))
    plt.plot(X,Y[j], label = 'sensitivity to '+p)
plt.legend()
plt.show()
    
        
# =============================================================================
# Time Series Momentum with market exposure adjustment
# =============================================================================

# momTadawul > 0.05 : 150%
# momTadawul in 0.005...0.05 : 100%
# momTadawul in 0...0.005 : 50%
# momTadawul <  0 : 0%

sector_classif  = pd.read_excel('C:/Users/jhouas/Desktop/KSA_sector_classif.xlsx')

Tadawul = get_stocks_data(['SASEIDX Index'], start_date, end_date, data_type='PX_LAST')

all_sectors = (data * mkt_cap.loc[data.columns]/mkt_cap.loc[data.columns].sum()).sum(axis=1)


def get_sector_index(sector, start_date, end_date):
    if sector == "All":
        data = get_stocks_data(universe, to_inttime(to_datetime(start_date) - datetime.timedelta(days=5)), to_inttime(to_datetime(end_date) + datetime.timedelta(days=5)), 'PX_LAST')
        all_sectors = (data * mkt_cap.loc[data.columns]/mkt_cap.loc[data.columns].sum()).sum(axis=1)
        return(all_sectors.loc[to_datetime(start_date):to_datetime(end_date)])
    else:
        if len(set(sector_classif[sector_classif['Sector']==sector]['Stock']) & set(data.columns)) < 5:
            return(all_sectors.loc[to_datetime(start_date):to_datetime(end_date)])
        else:
            x = x * mkt_cap.loc[x.columns]/mkt_cap.loc[x.columns].sum() # Market cap weighting
            return(x.sum(axis=1))


def Ths_fun(ths1,ths2,x):
    if x > ths1:
        return(1.5)
    elif x > ths2:  
        return(1)
    elif x > 0:
        return(0.5)
    else:
        return(0)
    
Ths =0.005638943418603367
Ths_min =0.0004977728700694834
Ths_max =0.009869632087707882

MW = 0.1
rebal = 7
period = 22



index_price = [1]
mapping = returns_0.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).rolling(period).mean().dropna()

for x in mkt_mom.index:
    if mkt_mom.loc[x] > Ths_max:
        mkt_mom.loc[x] = 1.5
    elif mkt_mom.loc[x] > Ths_min:  
        mkt_mom.loc[x] = 1
    elif mkt_mom.loc[x] > 0:
        mkt_mom.loc[x] = 0.5
    else:
        mkt_mom.loc[x] = 0
    
returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()
Allocation['Cash'] = Allocation.iloc[:,0]

#mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]

#fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")

transaction_cost_values = []
t = 0
while t < len(mapping):
    if t % rebal == 0:
        if mapping.iloc[t,:].sum() <= 10 :
            Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        else:
            Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
        
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
        index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
        transaction_cost_values.append(transaction_cost)
        t +=1
    else:
        Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
        Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
        Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
        t += 1

long_all = [1]
t = 0
Allocation_long = mapping.copy()
while t < len(mapping):
    if t % rebal == 0:
        Allocation_long.iloc[t,:] = 1/(Allocation_long.shape[1])
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:-1] * long_all[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*Allocation_long.iloc[t,:], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*abs(Allocation_long.iloc[t,:] - (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]), fees, mkt_cap)

        long_all.append(perf.sum() + Allocation_long.iloc[t,-1]* long_all[-1] - transaction_cost)
        t +=1
    else:
        Allocation_long.iloc[t,:] = (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]
        Allocation_long.iloc[t,:] = Allocation_long.iloc[t,:] / (Allocation_long.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:] * long_all[-1]
        long_all.append(perf.sum())
        t += 1

tc = pd.Series(transaction_cost_values, index = Allocation.index[[rebal*i for i in range(t//rebal+1)]])


# =============================================================================
# Time Series Momentum with sector exposure adjustment
# =============================================================================


Ths =0.005638943418603367
Ths_min =0.0004977728700694834
Ths_max =0.009869632087707882

MW = 0.1
rebal = 7
period = 22


index_price = [1]
mapping = returns_0.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
returns_sectors = returns_0.copy()
for x in returns_sectors.columns:
    returns_sectors.loc[:,x] = get_returns(get_sector_index(sector_classif[sector_classif['Stock']==x]['Sector'].values[0], start_date, end_date))

mom_sectors = returns_sectors.rolling(period).mean().dropna()
mapping_sectors = mom_sectors.applymap(lambda x : Ths_fun(Ths_max,Ths_min,x))

returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()
Allocation['Cash'] = Allocation.iloc[:,0]

#mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]

#fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")

transaction_cost_values = []
t = 0
while t < len(mapping):
    if t % rebal == 0:
        if mapping.iloc[t,:].sum() <= 10 :
            Allocation.iloc[t,:-1] = mapping_sectors.iloc[t,:] * mapping.iloc[t,:]*0.1
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        else:
            Allocation.iloc[t,:-1] = mapping_sectors.iloc[t,:] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
        
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
        index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
        transaction_cost_values.append(transaction_cost)
        t +=1
    else:
        Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
        Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
        Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
        t += 1

long_all = [1]
t = 0
Allocation_long = mapping.copy()
while t < len(mapping):
    if t % rebal == 0:
        Allocation_long.iloc[t,:] = 1/(Allocation_long.shape[1])
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:-1] * long_all[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*Allocation_long.iloc[t,:], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*abs(Allocation_long.iloc[t,:] - (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]), fees, mkt_cap)

        long_all.append(perf.sum() + Allocation_long.iloc[t,-1]* long_all[-1] - transaction_cost)
        t +=1
    else:
        Allocation_long.iloc[t,:] = (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]
        Allocation_long.iloc[t,:] = Allocation_long.iloc[t,:] / (Allocation_long.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:] * long_all[-1]
        long_all.append(perf.sum())
        t += 1

tc = pd.Series(transaction_cost_values, index = Allocation.index[[rebal*i for i in range(t//rebal)]])


# =============================================================================
# Optimization with market exposure
# =============================================================================



def model_1(Ths, Ths_max, Ths_min, rebal, period):
    
    rebal = int(rebal)
    period = int(period)
    
    index_price = [1]
    mapping = returns_0.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_0.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1        
        
    return(index_price[-1]**(1/10)-1)


from bayes_opt import BayesianOptimization

pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.01), 'Ths_min': (0.00001, 0.001), 'rebal': (5, 20), 'period': (10,50)}


### Optimization

def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    return(optimizer.max)

optimizer = bayesian_cross_val (model_1, pbounds)

opt_params = [0.005166017695020465 , 0.004534603536102631,0.00033647120331320457, 5, 30 ]
#opt_params = [0.005638943418603367, 0.009869632087707882, 0.0004977728700694834, 7, 22]
    
# X = np.linspace(0.9, 1.1, 5)
Y = [[],[]]
rebal_sens = [1,2,4,5,6,7,8,9]
period_sens = [24,26,28,30,32,34,36] #18,20,22,24,26]
period_sens = [1,2,3,4,5,6,7,8,9]
for i in rebal_sens:
    print(i)
    test =  opt_params.copy()
    test[3] = i
    Y[0].append(model_1(*test))
Z = []  
for i in period_sens:
    print(i)
    test =  opt_params.copy()
    test[4] = i
    Z.append(model_1(*test))
    
    
    
for j, p in enumerate(optimizer['params'].keys()):
    for i in X:
        test =  list(optimizer['params'].values())
        test[j] = i * test[j]
        Y[j].append(model_4(*test))
    plt.plot(X,Y[j], label = 'sensitivity to '+p)
plt.legend()
plt.show()

# =============================================================================
# Optimization with sector exposure
# =============================================================================



def model_2(Ths, Ths_max, Ths_min, rebal, period):
    
    rebal = int(rebal)
    period = int(period)
    
    index_price = [1]
    mapping = returns_0.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    returns_sectors = returns_0.copy()
    for x in returns_sectors.columns:
        returns_sectors.loc[:,x] = get_returns(get_sector_index(sector_classif[sector_classif['Stock']==x]['Sector'].values[0], start_date, end_date))
    
    mom_sectors = returns_sectors.rolling(period).mean().dropna()
    mapping_sectors = mom_sectors.applymap(lambda x : Ths_fun(Ths_max,Ths_min,x))
    
    returns = returns_0.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mapping_sectors.iloc[t,:] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mapping_sectors.iloc[t,:] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1
        
    return(get_sharpe_ratio(index_price, 252))


from bayes_opt import BayesianOptimization

pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.1), 'Ths_min': (0.00001, 0.001), 'rebal': (10, 100), 'period': (5,75)}

### Optimization

def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=5,
        n_iter=50,
    )
    return(optimizer.max)

optimizer = bayesian_cross_val (model_2, pbounds)

# =============================================================================
# Out-of-sample tester
# =============================================================================

def model_3(Ths, Ths_max, Ths_min, rebal, period, start_date, end_date):
    
    rebal = int(rebal)
    period = int(period)
    
    returns_period = returns_0.loc[to_datetime(start_date) + datetime.timedelta(days=1):to_datetime(end_date), :]
    
    index_price = [1]
    mapping = returns_period.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).rolling(period).mean().dropna()
    mapping = mapping.loc[mapping.index & mkt_mom.index, :]
    mkt_mom = mkt_mom.loc[mapping.index & mkt_mom.index]
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_period.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1        
        
    return(get_sharpe_ratio(index_price, 252))

    
def run_model_live(index_price, Ths, Ths_max, Ths_min, rebal, period, start_date, end_date):

    returns_period = returns_0.loc[to_datetime(start_date) - datetime.timedelta(days=period+10):to_datetime(end_date), :]
    
    mapping = returns_period.rolling(period).mean().dropna().apply(lambda x: x > Ths ).astype('int').loc[to_datetime(start_date):to_datetime(end_date), :]
    mkt_mom = get_returns(get_sector_index("All", to_inttime(to_datetime(start_date) - datetime.timedelta(days=period+10)), end_date)).rolling(period).mean().dropna().loc[to_datetime(start_date):to_datetime(end_date)]
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_period.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1  
    return(Allocation, transaction_cost_values)
    
    
train_period = 252
live_period = 100

pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.1), 'Ths_min': (0.00001, 0.001), 'rebal': (2, 25), 'period': (2,25)}

def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=2,
        n_iter=20,
    )
    return(optimizer.max)


t = 0
index_price = [1]
tc = []

Allocation = pd.DataFrame(columns = data.columns)

while t+live_period< len(returns_0):
    start_train = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t))
    end_train = to_inttime(to_datetime(start_date) + datetime.timedelta(days= t + train_period))
    end_live = to_inttime(to_datetime(start_date) + datetime.timedelta(days= t + train_period+live_period))
    
    mod = lambda Ths, Ths_max, Ths_min, rebal, period: model_3(Ths, Ths_max, Ths_min, rebal, period, start_train, end_train)
    
    optimizer = bayesian_cross_val (mod, pbounds)
    
    a, c = run_model_live(index_price, optimizer['params']['Ths'], optimizer['params']['Ths_max'], optimizer['params']['Ths_min'], int(optimizer['params']['rebal']), int(optimizer['params']['period']), end_train, end_live)
    
    Allocation = Allocation.append(a)
    
    tc = tc + c
    
    t += live_period
    

# =============================================================================
# TSM with exponential moving average
# =============================================================================
    
    

Ths = 0.005166017695020465 
Ths_min = 0.00033647120331320457
Ths_max = 0.004534603536102631

MW = 0.1
rebal = 5
period = 30


index_price = [1]
mapping = returns_0.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()

for x in mkt_mom.index:
    if mkt_mom.loc[x] > Ths_max:
        mkt_mom.loc[x] = 1.5
    elif mkt_mom.loc[x] > Ths_min:  
        mkt_mom.loc[x] = 1
    elif mkt_mom.loc[x] > 0:
        mkt_mom.loc[x] = 0.5
    else:
        mkt_mom.loc[x] = 0
    
returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()
Allocation['Cash'] = Allocation.iloc[:,0]

#mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]

#fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")

transaction_cost_values = []
t = 0
while t < len(mapping):
    if t % rebal == 0:
        if mapping.iloc[t,:].sum() <= 10 :
            Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        else:
            Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
            Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
        
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
        index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
        transaction_cost_values.append(transaction_cost)
        t +=1
    else:
        Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
        Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
        Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
        t += 1

long_all = [1]
t = 0
Allocation_long = mapping.copy()
while t < len(mapping):
    if t % rebal == 0:
        Allocation_long.iloc[t,:] = 1/(Allocation_long.shape[1])
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:-1] * long_all[-1]
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*Allocation_long.iloc[t,:], fees, mkt_cap)
        else:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), long_all[-1]*abs(Allocation_long.iloc[t,:] - (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]), fees, mkt_cap)

        long_all.append(perf.sum() + Allocation_long.iloc[t,-1]* long_all[-1] - transaction_cost)
        t +=1
    else:
        Allocation_long.iloc[t,:] = (returns.iloc[t-1,:] + 1) * Allocation_long.iloc[t-1,:]
        Allocation_long.iloc[t,:] = Allocation_long.iloc[t,:] / (Allocation_long.iloc[t,:].sum())
        perf = (returns.iloc[t,:] + 1) * Allocation_long.iloc[t,:] * long_all[-1]
        long_all.append(perf.sum())
        t += 1

tc = pd.Series(transaction_cost_values, index = Allocation.index[[rebal*i for i in range(t//rebal )]])


### optimization


def model_4(Ths, Ths_max, Ths_min, rebal, period):
    
    rebal = int(rebal)
    period = int(period)
    
    index_price = [1]
    mapping = returns_0.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_0.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1        
        
    return(get_sharpe_ratio(index_price, 252))


from bayes_opt import BayesianOptimization

pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.01), 'Ths_min': (0.0001, 0.001), 'rebal': (5, 20), 'period': (10,30)}

### Optimization

def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=1,
        n_iter=10,
    )
    return(optimizer.max)

optimizer = bayesian_cross_val (model_4, pbounds)

X = np.linspace(0.9, 1.1, 5)
Y = [[],[],[],[],[]]
for j, p in enumerate(optimizer['params'].keys()):
    for i in X:
        test =  list(optimizer['params'].values())
        test[j] = i * test[j]
        Y[j].append(model_4(*test))
    plt.plot(X,Y[j], label = 'sensitivity to '+p)
plt.legend()
plt.show()


# =============================================================================
# Out-of-sample tester for EWM
# =============================================================================


def model_5(Ths, Ths_max, Ths_min, rebal, period, start_date, end_date):
    
    rebal = int(rebal)
    period = int(period)
    
    returns_period = returns_0.loc[to_datetime(start_date) + datetime.timedelta(days=1):to_datetime(end_date), :]
    
    index_price = [1]
    mapping = returns_period.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    mkt_mom = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()
    mapping = mapping.loc[mapping.index & mkt_mom.index, :]
    mkt_mom = mkt_mom.loc[mapping.index & mkt_mom.index]
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_period.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1        
        
    return(get_sharpe_ratio(index_price, 252))

def run_model_live_ewm(index_price, Ths, Ths_max, Ths_min, rebal, period, start_date, end_date):

    returns_period = returns_0.loc[to_datetime(start_date) - datetime.timedelta(days=period+10):to_datetime(end_date), :]
    
    mapping = returns_period.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int').loc[to_datetime(start_date):to_datetime(end_date), :]
    mkt_mom = get_returns(get_sector_index("All", to_inttime(to_datetime(start_date) - datetime.timedelta(days=period+10)), end_date)).ewm(span=period, adjust=False).mean().dropna().loc[to_datetime(start_date):to_datetime(end_date)]
    
    for x in mkt_mom.index:
        if mkt_mom.loc[x] > Ths_max:
            mkt_mom.loc[x] = 1.5
        elif mkt_mom.loc[x] > Ths_min:  
            mkt_mom.loc[x] = 1
        elif mkt_mom.loc[x] > 0:
            mkt_mom.loc[x] = 0.5
        else:
            mkt_mom.loc[x] = 0
        
    returns = returns_period.loc[mapping.index, : ]
    Allocation = mapping.copy()
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    #mkt_cap = get_stocks_data(mapping.columns, '20160101', '20200101', data_type='CUR_MKT_CAP').mean().loc[mapping.columns]
    
    #fees = pd.read_excel("Z:/Public Folder/Duet Fund Information/1. Tools/Quant/fees.xlsx")
    
    transaction_cost_values = []
    t = 0
    while t < len(mapping):
        if t % rebal == 0:
            if mapping.iloc[t,:].sum() <= 10 :
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]*0.1
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            else:
                Allocation.iloc[t,:-1] = mkt_mom.iloc[t] * mapping.iloc[t,:]/(mapping.iloc[t,:].sum())
                Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
            
            if t == 0:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
            else:
                transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
            index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
            transaction_cost_values.append(transaction_cost)
            t +=1
        else:
            Allocation.iloc[t,:-1] = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
            Allocation.iloc[t,-1] = Allocation.iloc[t-1,-1]
            Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
            perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
            index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])
            t += 1  
    return(Allocation, transaction_cost_values)




train_period = 252
live_period = 100

#pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.01), 'Ths_min': (0.00001, 0.001), 'rebal': (2, 25), 'period': (2,25)}
pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.01), 'Ths_min': (0.00001, 0.001), 'rebal': (5, 20), 'period': (10,50)}


t = 0
index_price = [1]
tc = []

Allocation = pd.DataFrame(columns = data.columns)

while end_live < end_date:
    start_train = to_inttime(to_datetime(start_date) + datetime.timedelta(days=t))
    end_train = to_inttime(to_datetime(start_date) + datetime.timedelta(days= t + train_period))
    end_live = to_inttime(to_datetime(start_date) + datetime.timedelta(days= t + train_period+live_period))
    
    mod = lambda Ths, Ths_max, Ths_min, rebal, period: model_5(Ths, Ths_max, Ths_min, rebal, period, start_train, end_train)
    
    optimizer = bayesian_cross_val (mod, pbounds)
    
    a, c = run_model_live_ewm(index_price, optimizer['params']['Ths'], optimizer['params']['Ths_max'], optimizer['params']['Ths_min'], int(optimizer['params']['rebal']), int(optimizer['params']['period']), end_train, end_live)
    
    Allocation = Allocation.append(a)
    
    tc = tc + c
    
    t += live_period
    
    
# =============================================================================
# indicator dependant rebalancing momentum strategy
# =============================================================================


Ths = 0.01
Ths_min = 0.001
Ths_max = 0.01

MW = 0.1
period = 2

index_price = [1]
mapping = returns_0.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
weights = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()

for x in weights.index:
    if weights.loc[x] > Ths_max:
        weights.loc[x] = 1/len(mapping.columns)
    elif weights.loc[x] > Ths_min:  
        weights.loc[x] = 1/(0.65 * len(mapping.columns))
    elif weights.loc[x] > 0:
        weights.loc[x] = 1/(0.35 * len(mapping.columns))
    else:
        weights.loc[x] = 1/(0.2 * len(mapping.columns))
    
returns = returns_0.loc[mapping.index, : ]
Allocation = mapping.copy()

for t in range(len(mapping)):
    Allocation.iloc[t,:] = mapping.iloc[t,:] * weights.iloc[t]
    
Allocation['Cash'] = Allocation.iloc[:,0]

t = 0

transaction_cost_values = []

while t < len(mapping):
 
    if t == 0:
        transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
    else:
        transaction_cost = 0
        for i in range(len(mapping.columns[:-1])):
            if mapping.iloc[t,i] == mapping.iloc[t-1,i]:
                Allocation.iloc[t,i] =  Allocation.iloc[t-1,i ] *  (returns.iloc[t-1,i] + 1)
            else:
                transaction_cost += fees[(fees["Market"] == 'Saudi') & (fees["Market cap"] < mkt_cap.iloc[i])].iloc[-1,-1] * index_price[-1] * abs(Allocation.iloc[t,i] - (returns.iloc[t-1,i] + 1) * Allocation.iloc[t-1,i])          
        
    transaction_cost_values.append(transaction_cost)
    Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
        
    perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
    index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1] - transaction_cost)
    
    t += 1

tc = pd.Series(transaction_cost_values, index = Allocation.index)

# Optimization

def model_6(Ths, Ths_max, Ths_min, period):

    MW = 0.1
    period = int(period)
    
    index_price = [1]
    mapping = returns_0.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int')
    weights = get_returns(get_sector_index("All", start_date, end_date)).ewm(span=period, adjust=False).mean().dropna()
    
    for x in weights.index:
        if weights.loc[x] > Ths_max:
            weights.loc[x] = 1/len(mapping.columns)
        elif weights.loc[x] > Ths_min:  
            weights.loc[x] = 1/(0.65 * len(mapping.columns))
        elif weights.loc[x] > 0:
            weights.loc[x] = 1/(0.35 * len(mapping.columns))
        else:
            weights.loc[x] = 1/(0.2 * len(mapping.columns))
        
    returns = returns_0.loc[mapping.index, : ]
    Allocation = mapping.copy()
    
    for t in range(len(mapping)):
        Allocation.iloc[t,:] = mapping.iloc[t,:] * weights.iloc[t]
        
    Allocation['Cash'] = Allocation.iloc[:,0]
    
    t = 0
        
    while t < len(mapping):
     
        if t == 0:
            transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[1])], index=mapping.columns), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
        else:
            transaction_cost = 0
            for i in range(len(mapping.columns[:-1])):
                if mapping.iloc[t,i] == mapping.iloc[t-1,i]:
                    Allocation.iloc[t,i] =  Allocation.iloc[t-1,i ] *  (returns.iloc[t-1,i] + 1)
                else:
                    transaction_cost += fees[(fees["Market"] == 'Saudi') & (fees["Market cap"] < mkt_cap.iloc[i])].iloc[-1,-1] * index_price[-1] * abs(Allocation.iloc[t,i] - (returns.iloc[t-1,i] + 1) * Allocation.iloc[t-1,i])          
            
        Allocation.iloc[t,-1] = 1 - Allocation.iloc[t,:-1].sum()
            
        perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
        index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1] - transaction_cost)
        
        t += 1
    return(get_sharpe_ratio(index_price, 252))

pbounds = {'Ths': (0.00001, 0.01), 'Ths_max': (0.001, 0.01), 'Ths_min': (0.0001, 0.001),  'period': (2,30)}


def bayesian_cross_val (model, pbounds):
    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    return(optimizer.max)

optimizer = bayesian_cross_val (model_6, pbounds)
    

# =============================================================================
# Live strat - EWM 
# =============================================================================

universe = ['RJHI AB Equity', 'SABIC AB Equity', 'STC AB Equity',
       'RIBL AB Equity', 'BSFR AB Equity', 'SABB AB Equity',
       'ALINMA AB Equity', 'ALMARAI AB Equity', 'JOMAR AB Equity',
       'SECO AB Equity', 'MAADEN AB Equity', 'JARIR AB Equity',
       'SAFCO AB Equity', 'SAVOLA AB Equity', 'SIPCHEM AB Equity',
       'KAYAN AB Equity', 'APPC AB Equity', 'YANSAB AB Equity',
       'BJAZ AB Equity', 'SIIG AB Equity', 'NIC AB Equity',
       'MCDCO AB Equity', 'ALARKAN AB Equity', 'AOTHAIM AB Equity',
       'SACCO AB Equity', 'BUPA AB Equity', 'ZAINKSA AB Equity',
       'TAWUNIYA AB Equity', 'YACCO AB Equity', 'PETROCH AB Equity',
       'YNCCO AB Equity', 'TAIBA AB Equity', 'EMAAR AB Equity',
       'QACCO AB Equity', 'SPIMACO AB Equity', 'ALDREES AB Equity',
       'PETROR AB Equity', 'NADEC AB Equity', 'ADCO AB Equity',
       'ALCO AB Equity', 'SISCO AB Equity', 'EACCO AB Equity',
       'SCERCO AB Equity', 'SADAFCO AB Equity', 'CHEMICAL AB Equity',
       'BUDGET AB Equity', 'ARCCI AB Equity', 'JADCO AB Equity',
       'SARCO AB Equity', 'NGIC AB Equity', 'SAPTCO AB Equity',
       'SACO AB Equity', 'SAIC AB Equity', 'TACCO AB Equity',
       'CHEMANOL AB Equity', 'ALHOKAIR AB Equity', 'ALBABTAI AB Equity',
       'SRECO AB Equity', 'AADC AB Equity', 'ZIIC AB Equity',
       'KINGDOM AB Equity', 'ATTMCO AB Equity', 'MEDGULF AB Equity',
       'HB AB Equity', 'MALATH AB Equity', 'SFICO AB Equity',
       'WALAA AB Equity', 'AHFCO AB Equity', 'ZOUJAJ AB Equity',
       'AXA AB Equity', 'ASTRA AB Equity', 'GIZACO AB Equity',
       'NGCO AB Equity', 'SAUDIRE AB Equity', 'SCACO AB Equity',
       'BATIC AB Equity', 'ANAAM AB Equity', 'TAACO AB Equity',
       'TAPRCO AB Equity', 'NAMA AB Equity', 'BCI AB Equity',
       'SAAC AB Equity', 'SHIELD AB Equity', 'SPPC AB Equity',
       'ALKHLEEJ AB Equity', 'ALETIHAD AB Equity', 'GACO AB Equity',
       'REDSEA AB Equity', 'SIDC AB Equity', 'MESC AB Equity',
       'SPM AB Equity', 'SSP AB Equity', 'FIPCO AB Equity',
       'ASACO AB Equity', 'ALLIANZ AB Equity', 'AICC AB Equity',
       'SIECO AB Equity', 'NMMCC AB Equity', 'SAGR AB Equity',
       'TECO AB Equity', 'ALABDUL AB Equity', 'SALAMA AB Equity',
       'ACE AB Equity', 'ATC AB Equity', 'SAICO AB Equity',
       'GULFUNI AB Equity', 'ALALAMIY AB Equity', 'SABBT AB Equity']

Ths = 0.006
Ths_min = 0.00014
Ths_max = 0.005

MW = 0.1
rebal = 5
period = 10

index_price = [1]
Allocation = pd.DataFrame(columns = universe+["Cash"])
t = 0

today = date.today()
prev_day = today -  datetime.timedelta(days=1)

if t % rebal == 0:
    data_t = get_stocks_data(universe, to_inttime(today - datetime.timedelta(days=period + 2)), to_inttime(today), data_type='PX_LAST')
    returns_t = get_returns(data_t)
    
    mapping = returns_t.ewm(span=period, adjust=False).mean().dropna().apply(lambda x: x > Ths ).astype('int').iloc[-1,:]
    mkt_mom = get_returns(get_sector_index("All", to_inttime(today - datetime.timedelta(days=period + 2)), to_inttime(today))).ewm(span=period, adjust=False).mean().dropna().iloc[-1]
    
    if mkt_mom > Ths_max:
        mkt_mom = 1.5
    elif mkt_mom > Ths_min:
        mkt_mom = 1
    elif mkt_mom > 0:
        mkt_mom = 0.5
    else:
        mkt_mom = 0

    if mapping.sum() <= 10 :
        allo = mkt_mom * mapping*0.1
        Allocation = Allocation.append(pd.Series(list(allo.values) + [1 - allo.sum()], index=list(allo.index)+['Cash'], name=today))
    else:
        allo = mkt_mom * mapping/(mapping.sum())
        Allocation = Allocation.append(pd.Series(list(allo.values) + [1 - allo.sum()], index=list(allo.index)+['Cash'], name=today))

    perf = ((returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1]) * index_price[-1]
    
    if t == 0:
        transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[0])], index=mapping.index), index_price[-1]*Allocation.iloc[t,:-1], fees, mkt_cap)
    else:
        transaction_cost = get_transcation_cost(pd.Series(['Saudi' for j in range(mapping.shape[0])], index=mapping.index), index_price[-1]*abs(Allocation.iloc[t,:-1] - (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]), fees, mkt_cap)
    index_price.append(perf.sum() + Allocation.iloc[t,-1]* index_price[-1] - transaction_cost)
    transaction_cost_values.append(transaction_cost)
else:
    allo = (returns.iloc[t-1,:] + 1) * Allocation.iloc[t-1,:-1]
    Allocation = Allocation.append(pd.Series(list(allo.values) + [Allocation.iloc[t-1,-1]], index=list(allo.index)+['Cash'], name=today))
    Allocation.iloc[t,:] = Allocation.iloc[t,:] / (Allocation.iloc[t,:].sum())
    perf = (returns.iloc[t,:] + 1) * Allocation.iloc[t,:-1] * index_price[-1]
    index_price.append(perf.sum()+Allocation.iloc[t,-1]*index_price[-1])


       