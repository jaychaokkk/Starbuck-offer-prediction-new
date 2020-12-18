# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 星巴克毕业项目
#
# ### 简介
#
# 　　这个数据集是一些模拟 Starbucks rewards 移动 app 上用户行为的数据。每隔几天，星巴克会向 app 的用户发送一些推送。这个推送可能仅仅是一条饮品的广告或者是折扣券或 BOGO（买一送一）。一些顾客可能一连几周都收不到任何推送。 
#
# 　　顾客收到的推送可能是不同的，这就是这个数据集的挑战所在。
#
# 　　你的任务是将交易数据、人口统计数据和推送数据结合起来判断哪一类人群会受到某种推送的影响。这个数据集是从星巴克 app 的真实数据简化而来。因为下面的这个模拟器仅产生了一种饮品， 实际上星巴克的饮品有几十种。
#
# 　　每种推送都有有效期。例如，买一送一（BOGO）优惠券推送的有效期可能只有 5 天。你会发现数据集中即使是一些消息型的推送都有有效期，哪怕这些推送仅仅是饮品的广告，例如，如果一条消息型推送的有效期是 7 天，你可以认为是该顾客在这 7 天都可能受到这条推送的影响。
#
# 　　数据集中还包含 app 上支付的交易信息，交易信息包括购买时间和购买支付的金额。交易信息还包括该顾客收到的推送种类和数量以及看了该推送的时间。顾客做出了购买行为也会产生一条记录。 
#
# 　　同样需要记住有可能顾客购买了商品，但没有收到或者没有看推送。
#
# ### 示例
#
# 　　举个例子，一个顾客在周一收到了满 10 美元减 2 美元的优惠券推送。这个推送的有效期从收到日算起一共 10 天。如果该顾客在有效日期内的消费累计达到了 10 美元，该顾客就满足了该推送的要求。
#
# 　　然而，这个数据集里有一些地方需要注意。即，这个推送是自动生效的；也就是说，顾客收到推送后，哪怕没有看到，满足了条件，推送的优惠依然能够生效。比如，一个顾客收到了"满10美元减2美元优惠券"的推送，但是该用户在 10 天有效期内从来没有打开看到过它。该顾客在 10 天内累计消费了 15 美元。数据集也会记录他满足了推送的要求，然而，这个顾客并没被受到这个推送的影响，因为他并不知道它的存在。
#
# ### 清洗
#
# 　　清洗数据非常重要也非常需要技巧。
#
# 　　你也要考虑到某类人群即使没有收到推送，也会购买的情况。从商业角度出发，如果顾客无论是否收到推送都打算花 10 美元，你并不希望给他发送满 10 美元减 2 美元的优惠券推送。所以你可能需要分析某类人群在没有任何推送的情况下会购买什么。
#
# ### 最后一项建议
#
# 　　因为这是一个毕业项目，你可以使用任何你认为合适的方法来分析数据。例如，你可以搭建一个机器学习模型来根据人口统计数据和推送的种类来预测某人会花费多少钱。或者，你也可以搭建一个模型来预测该顾客是否会对推送做出反应。或者，你也可以完全不用搭建机器学习模型。你可以开发一套启发式算法来决定你会给每个顾客发出什么样的消息（比如75% 的35 岁女性用户会对推送 A 做出反应，对推送 B 则只有 40% 会做出反应，那么应该向她们发送推送 A）。
#
#
# # 数据集
#
# 一共有三个数据文件：
#
# * portfolio.json – 包括推送的 id 和每个推送的元数据（持续时间、种类等等）
# * profile.json – 每个顾客的人口统计数据
# * transcript.json – 交易、收到的推送、查看的推送和完成的推送的记录
#
# 以下是文件中每个变量的类型和解释 ：
#
# **portfolio.json**
# * id (string) – 推送的id
# * offer_type (string) – 推送的种类，例如 BOGO、打折（discount）、信息（informational）
# * difficulty (int) – 满足推送的要求所需的最少花费
# * reward (int) – 满足推送的要求后给与的优惠
# * duration (int) – 推送持续的时间，单位是天
# * channels (字符串列表)
#
# **profile.json**
# * age (int) – 顾客的年龄 
# * became_member_on (int) – 该顾客第一次注册app的时间
# * gender (str) – 顾客的性别（注意除了表示男性的 M 和表示女性的 F 之外，还有表示其他的 O）
# * id (str) – 顾客id
# * income (float) – 顾客的收入
#
# **transcript.json**
# * event (str) – 记录的描述（比如交易记录、推送已收到、推送已阅）
# * person (str) – 顾客id
# * time (int) – 单位是小时，测试开始时计时。该数据从时间点 t=0 开始
# * value - (dict of strings) – 推送的id 或者交易的数额
#

# +
import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import pickle
import os
import re
pd.set_option("display.max_columns", 100)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score , classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
InteractiveShell.ast_node_interactivity = "all"
# %matplotlib inline

# read in the json files
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

def pklsave(model, filename):
    """
    This function is to save the sklearn object
    INPUT :
        model : sklearn object
        filename : filepath to saved
    RETURN : none
    """
    pickle.dump(model, open(filename,'wb'))
    
def pklload(filename):
    """
    This function is to load the saved sklearn object
    INPUT : filename : filepath
    RETURN : loaded sklearn object
    """
    return pickle.load(open(filename, 'rb'))


# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
# -

# ## 概览portfolio

portfolio.info()


portfolio


# - 只有10个 offerid , 对应不同的推送活动 所以 ifficulty,duration, offer_type,reward可以合并起来 作为一个完整的推送活动名字 为 offername，后面处理的时候  offername 和 offerid 只留下一个就行
# - channels 列可以做成onehot编码
#     
#     
# - Only 10 data, so the information of 'offer_type', 'difficulty', 'duration', 'reward'  can all be seen as special content of one offer type, and can be added into one
# - The channels column needs to be one-hot encoded.
#

# +
# 函数: 对portfolio前处理，
# 思路：
# 1 onehot编码 , list_channels = ['web', 'email', 'mobile', 'social']
# 2 offername  把 ['offer_type', 'difficulty', 'duration', 'reward'] 这四个合并成一个 offername，后面处理的时候 offername和 offerid 留一个就行

def portfolio_prep(df):
    # df = portfolio
    # one hot encoded   channels 4 class
    df = df.copy()
    list_channels = ['web', 'email', 'mobile', 'social']

    for ch in list_channels:
        # ch = list_channels[0]
        df[ch] = df.channels.apply(lambda x: 1 if ch in x else 0)

    df = df.drop('channels', axis=1)

    df['offername'] = df.offer_type.astype(str) +'_'+ df.difficulty.astype(str) +'_'+ \
        df.duration.astype(str) +'_'+  df.reward.astype(str)
    
    # df = df.drop( ['offer_type', 'difficulty', 'duration', 'reward']  , axis=1)
   
    return df


# -

portfolio_clean = portfolio_prep( portfolio ) 
portfolio_clean

# ## 概览profile

profile.head()

profile.info()
# `became_member_on` data type is int64, need to parse to dates.

# - gender 和 income存在缺失数据   
# - became_member_on 可以先转化为datetime, 在转化为用户的  became_member_year 和 member_days 特征 
# - 特征工程的时候 age最好转化为年龄分组数据 以10为间隔
#     
# - There are some missing data for gender and income
# - became_member_on can be converted to datetime first, and then converted to the user's became_member_year and member_days features
# - In feature engineering, age is best converted into age grouping data with 10 intervals

# ### age 和 income

profile[['age', 'income']].describe()

profile[['age', 'income', 'became_member_on']].hist(figsize=(10,5));


# - age数据里 有太多118岁的 看上去不正常,总共2175人
# - age和income 基本符合正态分布，became_member人数 每年逐渐增多
#     
#     
# - to much 118 years old people (2175) in age columns, seems abnormal
# - age and income basically conform to the normal distribution, the number of become_member gradually increases every year

profile[profile.age == 118]['age'].value_counts()  # 2175


# ### 列缺失

profile.isnull().sum()  #  gender 2175, income 2175


profile[profile.age == 118][ ['gender', 'income'] ]


# - 118岁的用户 在income和gender上都缺失了信息 如果不影响整体特征分布 可以考虑删除。
#     
#     
# - Age at where 118 all missing values in income and gender columns. If it does not affect the overall feature distribution, consider delete these.

# ###  行缺失

# How much data is missing in each row of profiile  dataset?
row_null = profile.isnull().sum(axis=1)
row_null.value_counts()


# - 有2175行缺少了2个特征 占总数的14%
# - There are 2175 rows missing 2 features, 14% of the total

# +
#%%  比较 行缺失 vs 无缺失，在各列下的分布图  有无明显差别 

profile_wonull = profile[row_null ==0]
profile_winull = profile[row_null > 0]

for col in ['age', 'became_member_on','gender', 'income']:
    # col = 'became_member_on'
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
   
    if col == 'gender':
        # col = 'gender'
        profile_wonull[col].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(),rotation=0)
        try :
            profile_winull[col].value_counts().plot(kind='bar', ax=ax2)
        except:
            pass
    else:
        profile_wonull[col].hist(ax=ax1)
        profile_winull[col].hist(ax=ax2)
    
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_title('no missing value' )
    ax2.set_title('with missing value')
    f.suptitle('Feature: ' + col)
    f.tight_layout()
    f.subplots_adjust(top=0.85)

# -

# ### 重复

profile.id.duplicated().sum()  # 无重复

# ### income vs gender

profile.gender.value_counts()

# +
# 分析性别 和 income， 作图 性别count， 不同性别-income hist分布，不同性别-income box图
f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))

# gender distribution bar
profile.gender.value_counts().plot(kind='bar', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.set_title('Gender Distribution')
ax1.set_ylabel('Count')

# different gender's income distribution histogram
ax2.hist(profile[profile.gender == 'M']['income'], alpha=0.5, label='M')
ax2.hist(profile[profile.gender == 'F']['income'], alpha=0.5, label='F')
ax2.hist(profile[profile.gender == 'O']['income'], alpha=0.5, label='O')
ax2.set_xlabel('$ Income')
ax2.set_ylabel('Count')
ax2.set_title('Income Distribution')
ax2.legend()

# different gender's income statics box
sns.boxplot('gender', 'income', data=profile, ax=ax3, order=['M', 'F', 'O'])

plt.tight_layout()

# -

# - 男性人数较多 女性其次 还有212个没有性别的
# - 女性平均收入较高 其次是无性别的 
#   
#   
# - There are more men than women, and 212 people with no gender imfornation.
# - Women's average income is higher, followed by genderless

# - 对于缺失数据的行，在　became_member_on这个特征上和无缺失行　没有明显的分布规律差别,可以删除
# - For rows with missing data, there is no obvious difference in the distribution pattern of "became_member_on" with no missing rows. These rows can be deleted.

# ### 转换时间类型

# +
def profile_prep_dates(df):
    # df = profile
    df = df.copy()
    #convert to string
    df['became_member_on'] = pd.to_datetime( df.became_member_on.apply(lambda x: str(x)) )
    return df

profile_clean = profile_prep_dates( profile )
# -

# ## 概览 Transcript

transcript.head()


transcript.sample(20)

transcript.info()

# ### 缺失

transcript.isnull().sum()  # no null


# ###  event 和 value

transcript.event.value_counts()


transcript.value.sample(20)


# - value 列有两种情况 'offer id' 或者'offer_id' 字典，值为offerid，'amount'字典，值为交易量，因此这一列需要拆成2列，分别用'offer_id'和'amount'作为列名，值为对应的字典值，注意需要把'offer id'变为'offer_id'
#
# - event 列有4种活动，其中 offer received, offer viewed and offer completed 都是和 offer 有关 对应了value列的'offer id'，transaction 对应了value列的'amount'
#
#     
# - value column has two types, one is dict of 'offer id' and 'offer_id'!!!!!!!!!!! the other is dict of 'amount'
# - event column has two types, one is offer received, viewed and completed, the other is transaction
# - event transaction corresponds to value amount, and other event offer actions corresponds to value offer_id.
# - dict of 'offer id'and 'offer_id' can get a new column named 'offer id' in the row of offer event
# - dict of 'amount' can get a new column named 'amount' in the row of transaction

transcript.person.unique().shape


# ### 增加两列 offer_id  and  amount 

# +

def encode_offer_id(x):
    try :
        return x['offer id']
    except:
        return x['offer_id']

def transcript_prep(df):
    # df = transcript
    trans_df = df.copy()
    trans_df['offer_id'] = trans_df[~trans_df.event.isin(['transaction'])].value.apply(encode_offer_id)   # why can't lambda?
    trans_df['amount'] =  trans_df[trans_df.event.isin(['transaction'])].value.apply(lambda x: x['amount'])
    trans_df = trans_df.drop( 'value' , axis=1 )
    return trans_df


# -

transcript_try = transcript_prep(transcript)
transcript_try.sample(10)

# ## 三个合并  profile， portfolio， transcript  
#

print( portfolio_clean.columns) 
print( profile_clean.columns )
print( transcript.columns )


def merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df):
    # transcript_df= transcript ; profile_df = profile; portfolio_df = portfolio
    portfolio_clean = portfolio_prep( portfolio_df ) 
    profile_clean = profile_prep_dates(profile_df)
    transcript_clean = transcript_prep(transcript_df)
    
    dfmerge_tr_pr = pd.merge(transcript_clean, profile_clean, left_on=['person'], right_on = ['id'], how ='left')
    dfmerge_tr_pr = dfmerge_tr_pr.drop(['id'], axis=1)
    
    dfmerge_trpr_po =  pd.merge(dfmerge_tr_pr, portfolio_clean, left_on = 'offer_id', right_on ='id', how='left')
    dfmerge_trpr_po = dfmerge_trpr_po.drop(['id'], axis=1)
    return dfmerge_trpr_po


dfmerge_trprpo = merge_transcript_profile_portfolio(transcript, profile, portfolio)

dfmerge_trprpo.head()

# ### 缺失

dfmerge_trprpo.isnull().sum()

# ## 数据探索

# ### amount 平均，总数 ， 次数

# 所有用户用户的 交易情况 amount 平均值分布
dfmerge_trprpo.groupby('person')['amount'].mean().hist(bins=50)
plt.xlabel('$ Spending')
plt.ylabel('Count')
plt.title('Average Spending per customer')
dfmerge_trprpo.groupby('person')['amount'].mean().describe()

# +
# 所有用户的 交易情况 amount 总和分布

dfmerge_trprpo.groupby('person')['amount'].sum().hist(bins=50)
plt.xlabel('$ Spending')
plt.ylabel('Count')
plt.title('TOTAL Spending per customer')
dfmerge_trprpo.groupby('person')['amount'].sum().describe()

# +
# 所有用户的 交易情况 amount  count 分布

dfmerge_trprpo.groupby('person')['amount'].count().hist()
plt.xlabel('Number of transaction')
plt.ylabel('Count')
plt.title('The distribution of Number of transaction per customer')
dfmerge_trprpo.groupby(['person'])['amount'].count().describe()

# -

# - 用户单次交易的平均值是16元，大部分都在20元以下
# - 用户交易量的平均值是125元，大部分都在150元以下
# - 用户交易次数的平均值是8次，大部分都在11次以下
#     
#
# - The average value of a user's single transaction is 16 yuan, most of which are below 20 yuan
# - The average value of a user's total transaction is 125 yuan, most of which are below 150 yuan
# - The average value of a user's transactions count is 8, and most of them are below 11
#

# ### amount vs gender

# assign missing value in gender with 'U' value
dfmerge_temp = dfmerge_trprpo.copy()
dfmerge_temp.loc[dfmerge_temp.gender.isnull(), 'gender'] ='U'

# +
# 分析  不同性别-用户的 交易情况 amount 平均值 分布， 以及 不同性别-用户的  amount 计数，  不同性别-用户的  income box图

""" The number of transaction by GENDER """

#plot avg spending
f, (ax1, ax2, ax3 , ax4) = plt.subplots(1,4, figsize=(16,4))
dfmerge_temp.groupby('gender')['amount'].mean()[['M', 'F', 'O', 'U']].plot(kind='bar', ax=ax1);
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.set_ylabel("The average spending of transaction")
ax1.set_title('The avg spending of transaction')

# plot avg number of transaction

dfmerge_temp.groupby(['gender', 'amount'])['amount'].count().mean(level=0)[['M', 'F', 'O', 'U']].plot(kind='bar', ax=ax2);
ax2.set_xticklabels(ax1.get_xticklabels(), rotation=0);
ax2.set_ylabel("The average number of transaction");
ax2.set_title("The avg number of transaction");

# boxplot income distribution
dfmerge_temp.groupby('gender')['amount'].sum().plot(kind='bar', ax=ax3);
ax3.set_xticklabels(ax1.get_xticklabels(), rotation=0);
ax3.set_ylabel("The total spending of transaction");
ax3.set_title("The total spending of transaction");

#plot gender distribution
profile.fillna('U').gender.value_counts()[['M', 'F', 'O', 'U']].plot(kind='bar', ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
plt.title('Gender Distribution');
ax4.set_xlabel('gender')

plt.tight_layout()
# -

# - 女性的平均交易量最多, 其次是其他性别和男性
# - 男性的交易次数最多, 其次是其他性别和女性
# - 男性女性的总交易量比较接近,可能是因为男性虽然单次交易量少,但是人数多.其他性别和unknown的总交易量都很少
#     
#     
# - Women have the most average spending, followed by other genders and men
# - Men have the most average transaction count, followed by other genders and women
# - The total spending of males and females is relatively close, which may be due to the fact that although the males' single transaction spending is small, the number of males is large.The total spending of other genders and unknown is very small

# ### offer event count

# +
# 分析 查看不同offer 的 receive view 和 complete 数量，水平图

"""
Portfolio Distribution by event
"""
df_offerevent = dfmerge_trprpo[~dfmerge_trprpo.event.isin(['transaction'])]
plt.figure(figsize=(6,6))
sns.countplot(y='offername', hue='event', data= df_offerevent);
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=90)
plt.ylabel("Portfolio Name")
plt.title("Portfolio Distribution by Event")

# 发现 complete 竟然比 view还多 说明有的人 complete 和推荐无关 属于无效记录

# -

# -  对于bogo_5_7_5和discount_20_10_5这两种推送，complete 竟然比 view还多 说明有的人 complete 和推送无关, 属于无效记录
#     
#     
# - For push offers bogo_5_7_5 and discount_20_10_5, complete count is even more than view count, indicating that some complete event has nothing to do with push offers,these are invalid records
#

# +
# 分析 查看不同offer的比例  receive view 和 complete 数量 / receive数量，水平图

event_count = dfmerge_trprpo[~dfmerge_trprpo.event.isin(['transaction'])].groupby(['offername', 'event']).offer_id.count().unstack()
event_count['offer_received_frac'] = event_count['offer received']/ event_count['offer received']
event_count['offer_viewed_frac'] = event_count['offer viewed']/ event_count['offer received']
event_count['offer_completed_frac'] =  event_count['offer completed']/ event_count['offer received']

event_count_frac = event_count[['offer_received_frac','offer_viewed_frac', 'offer_completed_frac']]

event_count_stack = event_count_frac.stack().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x=0,y='offername', hue='event', data=event_count_stack)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(top=0.8)
plt.title("Before Cleaning : Offer Received, Viewed, and Completed (fraction)")
plt.ylabel('')
plt.xlabel('Fraction')

plt.tight_layout()
# -

# ### find invalid transaction 

# #### 函数 find_invalid 找到有效. 无效完成
#

# +
# 测试
df_merg = dfmerge_trprpo.copy()
portfolio_df = portfolio

# 
# 思路：
#  对买个用于进行迭代，看看用户有哪些offer活动 
# 然后对每个用户 每个offer活动迭代，看看这个offer是否有 revieve view 以及 complete
#  有view同时 ， complete 又在活动 receive 时间范围内，标记为 valid complete
# 对于其他complete 标记为 invalid complete

list_person = df_merg.person.unique()
list_validcomp = [];  list_invalidcomp = [] ; 

for person_id in tqdm( list_person ) :
    # person_id= list_person[0]
    # person_id = '88baa20c29a94178a43a7d68e5f039d4'
    df_sub_person = dfmerge_trprpo[dfmerge_trprpo.person == person_id]
    list_psn_ofr = df_sub_person.offer_id.unique()
    list_psn_ofr = list_psn_ofr[ ~pd.isna(list_psn_ofr) ] 
    
    for psn_ofr_id in list_psn_ofr:
#         psn_ofr_id  = list_psn_ofr[0]
#  psn_ofr_id = '0b1e1539f2cc45b7b9fa7c272da2e1d7'
        df_sub_psn_ofr  = df_sub_person[df_sub_person.offer_id == psn_ofr_id ]
        tj_rec =  'offer received'  in df_sub_psn_ofr.event.values
        tj_view =  'offer viewed'  in df_sub_psn_ofr.event.values
        tj_comp =  'offer completed'  in df_sub_psn_ofr.event.values
        # 对于每个收到的
        if tj_rec == True:
            starttime = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer received' ].time.values.min()
            duringtime  = portfolio_df[ portfolio_df.id ==  psn_ofr_id ].duration.values[0] *24
            endtime = starttime+ duringtime
            index_rec =  df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer received' ].time.idxmin()
            # 对于收到且完成了的
            if tj_comp ==True:
                comptime = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer completed' ].time.values.min()
                # 记录这个行号
                index_comp = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer completed' ].time.index.values
                
                # 对于收到完成  且在规定时间 且已经查看了的
                if comptime <= endtime and tj_view == True:
                    list_validcomp.extend( index_comp)
                else:
                    list_invalidcomp.extend( index_comp)
#             # 对于收到且没完成的
#             else :
#                     df_valid_1.loc[ 0 ,:]  =     [ person_id, psn_ofr_id, 'no_complete' , index_rec ]   

# -

# 输入 ； 合并后的数据 df_merg，以及合并前包含offer持续时间的offer数据  portfolio_df
# 输出 ：  df_merg 中所有有效offer完成 （单个用户 某个offer 看过的前提下 完成了）
# 和无效offer完成的 index （单个用户 某个offer 没看过就完成了）， 
# 分别为 list_validcomp 和 list_invalidcomp
def find_valid( df_merg , portfolio_df ):

    list_person = df_merg.person.unique()
    list_validcomp = [];  list_invalidcomp = [] ; list_validrec = []

    for person_id in tqdm( list_person ) :
        # person_id= list_person[0] ; person_id = '88baa20c29a94178a43a7d68e5f039d4'
        df_sub_person = dfmerge_trprpo[dfmerge_trprpo.person == person_id]
        list_psn_ofr = df_sub_person.offer_id.unique()
        list_psn_ofr = list_psn_ofr[ ~pd.isna(list_psn_ofr) ] 

        for psn_ofr_id in list_psn_ofr:
    #       psn_ofr_id  = list_psn_ofr[1] ; psn_ofr_id = '0b1e1539f2cc45b7b9fa7c272da2e1d7'
            df_sub_psn_ofr  = df_sub_person[df_sub_person.offer_id == psn_ofr_id ]
            tj_rec =  'offer received'  in df_sub_psn_ofr.event.values
            tj_view =  'offer viewed'  in df_sub_psn_ofr.event.values
            tj_comp =  'offer completed'  in df_sub_psn_ofr.event.values
            # 对于每个收到的
            if tj_rec == True:
                starttime = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer received' ].time.values.min()
                duringtime  = portfolio_df[ portfolio_df.id ==  psn_ofr_id ].duration.values[0] *24
                endtime = starttime+ duringtime
                # 最早收到offer的行号
                index_rec =  df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer received' ].time.idxmin()
                # 对于收到且完成了的
                if tj_comp ==True:
                    # 完成时间
                    comptime = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer completed' ].time.values.min()
                    # 完成offer的行号
                    index_comp = df_sub_psn_ofr[ df_sub_psn_ofr.event ==  'offer completed' ].index.values
                    # 对于在规定时间内完成的 且已经查看了的
                    if comptime <= endtime and tj_view == True:
                        # 记录 完成offer的行号
                        list_validcomp.extend( index_comp)
                        # 记录 完成时间之前的所有receive的行号
                        list_validrec.append( index_rec )
                    else:
                        list_invalidcomp.extend( index_comp )

    return list_validcomp, list_invalidcomp, list_validrec


# +

if  os.path.exists('sav/list_validcomp.pkl') and os.path.exists('sav/list_validcomp.pkl') and os.path.exists('sav/list_validrec.pkl'):
    list_validcomp =  pklload('sav/list_validcomp.pkl')
    list_invalidcomp =  pklload('sav/list_invalidcomp.pkl')
    list_validrec =  pklload('sav/list_validrec.pkl')

else:    
    list_validcomp, list_invalidcomp ,list_validrec =   find_valid( dfmerge_trprpo , portfolio )
#     list_validcomp
    pklsave(list_validcomp, 'sav/list_validcomp.pkl')
    pklsave(list_invalidcomp, 'sav/list_invalidcomp.pkl')
    pklsave(list_validrec, 'sav/list_validrec.pkl')
# -

# list_validcomp 和 list_validrec是不是长度一样？
len( list_validcomp ) 
len( list_validrec ) 
# list_validcomp比较多 说明有的人 有效完成次数不止一次

# 给 dfmerge_trprpo 增加一列  valid_complete,  valid为1 ，invalid 和 no_complete都为0
dfmerge_trprpo.loc[ : ,'valid'] = 0
dfmerge_trprpo.loc[ list_validcomp ,'valid'] =1
dfmerge_trprpo.loc[ : ,'invalid'] = 0
dfmerge_trprpo.loc[ list_invalidcomp ,'invalid'] =1
dfmerge_trprpo

# ### 去掉 invalid_complete以后  再对offer_event count进行作图

# +
#%% 去掉无效行以后 重新统计  不同offer- receive view 和 complete 比例，水平图

df_trprpo_valid = dfmerge_trprpo[ dfmerge_trprpo.invalid == 0 ]

event_count = df_trprpo_valid[~df_trprpo_valid.event.isin(['transaction'])].groupby(['offername', 'event']).offer_id.count().unstack()
event_count['offer_received_frac'] = event_count['offer received']/ event_count['offer received']
event_count['offer_viewed_frac'] = event_count['offer viewed']/ event_count['offer received']
event_count['offer_completed_frac'] =  event_count['offer completed']/ event_count['offer received']

event_count_frac = event_count[['offer_received_frac','offer_viewed_frac', 'offer_completed_frac']]

event_count_stack = event_count_frac.stack().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x=0,y='offername', hue='event', data=event_count_stack)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(top=0.8)
plt.title("Before Cleaning : Offer Received, Viewed, and Completed (fraction)")
plt.ylabel('')
plt.xlabel('Fraction')

plt.tight_layout()


# -

# - 对于所有推送offer 发现 complete 比 view 少，说明只剩下了有效记录
#     
#     
# - For all push offers, it is found that complete is less than view, indicating that only valid records are left

# # 函数集合

# ## 清理信息， 合并

# +
# 函数 portfolio_prep 
# 输入 没有清洗以前的 portfolio
# 输出  处理后的 portfolio
# 对 channels onehot, 
# 把 offer_type difficulty duration reward 合并为一个offername，
def portfolio_prep(df):
    # df = portfolio
    # one hot encoded   channels 4 class
    df = df.copy()
    list_channels = ['web', 'email', 'mobile', 'social']

    for ch in list_channels:
        # ch = list_channels[0]
        df[ch] = df.channels.apply(lambda x: 1 if ch in x else 0)

    df = df.drop('channels', axis=1)

    df['offername'] = df.offer_type.astype(str) +'_'+ df.difficulty.astype(str) +'_'+ \
        df.duration.astype(str) +'_'+  df.reward.astype(str)
    
#     df = df.drop( ['offer_type', 'difficulty', 'duration', 'reward']  , axis=1)
    return df


# 输入 原始 profile
# 输出  处理后的 profile
# 1 对 became_member_on 转化为时间类型，然后提取 year ，转化onehot
# 2 用户年龄分组  10为一组
def profile_prep(df):
    # df = profile
    df = df.copy()
    #convert to string
    df['became_member_on'] = pd.to_datetime( df.became_member_on.apply(lambda x: str(x)) )
    df[ 'member_year'] = df['became_member_on'].apply(lambda x: 'mbyr_'+ str(x.year) )
    df = pd.concat( [ df , pd.get_dummies(df['member_year'])] ,axis= 1 ) 
    
    min_age_limit = np.int(np.floor(np.min(df['age'])/10)*10)
    max_age_limit = np.int(np.ceil(np.max(df['age'])/10)*10)

    df['agerange'] =pd.cut(df['age'], (range(min_age_limit,max_age_limit + 10, 10)),right=False)

    df['agerange'] = df['agerange'].astype('str')

    df = pd.concat( [ df , pd.get_dummies(df['agerange'])] ,axis= 1 ) 
    df =df.drop(columns=['agerange'])
    return df

def encode_offer_id(x):
    try :
        return x['offer id']
    except:
        return x['offer_id']

# 输入 没有处理以前的  transcript
# 输出 处理以后的 transcript， value列转化为另外两列 offer_id  amount
# 思路
# 1  把 event列中 transaction以外的 对应的字典，要么是 x['offer id']  或者x['offer_id']  放到 offer_id这一列里
# 2  把 event列中 transaction 对应的字典，x['amount'] 放到 amount 这一列里
# 3 删除原来的value
def transcript_prep(df):
    # df = transcript
    trans_df = df.copy()
    trans_df['offer_id'] = trans_df[~trans_df.event.isin(['transaction'])].value.apply(encode_offer_id)   # why can't lambda?
    trans_df['amount'] =  trans_df[trans_df.event.isin(['transaction'])].value.apply(lambda x: x['amount'])
    trans_df = trans_df.drop( 'value' , axis=1 )
    return trans_df

# 输入 3个原始df
# 输出 清洗后的3个df ，根据用户id和offerid  合并起来
def merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df):
    # transcript_df= transcript ; profile_df = profile; portfolio_df = portfolio
    portfolio_clean = portfolio_prep( portfolio_df ) 
    profile_clean = profile_prep(profile_df)
    transcript_clean = transcript_prep(transcript_df)
    
    dfmerge_tr_pr = pd.merge(transcript_clean, profile_clean, left_on=['person'], right_on = ['id'], how ='left')
    dfmerge_tr_pr = dfmerge_tr_pr.drop(['id'], axis=1)
    
    dfmerge_trpr_po =  pd.merge(dfmerge_tr_pr, portfolio_clean, left_on = 'offer_id', right_on ='id', how='left')
    dfmerge_trpr_po = dfmerge_trpr_po.drop(['id'], axis=1)
    return dfmerge_trpr_po




# -

# ##  特征提取

# +
#%% 函数  根据amount提取消费信息
# 输入 某个用户的 id 和 用户子df
# 输出  这个用户名 命名的ser 消费的 关于平均 计数 总值
def addfeat_spending(df, profile_id):
    # profile_id = dfmerge_trprpo.person.unique()[0] ; 
    # df = dfmerge_trprpo[ dfmerge_trprpo.person ==  profile_id]  ; 
    avg_spending = df.amount.mean()
    transaction_count = df.amount.count()
    sum_spending = df.amount.sum()

    spending_series = pd.Series([avg_spending, transaction_count, sum_spending], \
                                index=["avg_spending", "transaction_count", 'sum_spending'], name=profile_id)
    return spending_series

# 输入 某个用户的 id 和 用户子df
# 输出 这个用户 有效完成和无效完成  对于两种 offer_type ：bogo  discount ，的次数
def addfeat_invalid( df , profile_id):
    # profile_id = dfmerge_trprpo.person.unique()[0] ; 
    # df = dfmerge_trprpo[ dfmerge_trprpo.person ==  profile_id]  ; 
    #  df  = subset_df.copy()
    valid_bogo_count = len(df[ (df.valid == 1) &  (df.offer_type == 'bogo')])
    invalid_bogo_count = len(df[ (df.invalid == 1 ) &  (df.offer_type == 'bogo')])
    valid_dscut_count = len(df[ (df.valid == 1) &  (df.offer_type == 'discount')])
    invalid_dscut_count = len(df[ ( df.invalid == 1)  &  (df.offer_type == 'discount') ])
    valid_series = pd.Series([valid_bogo_count , invalid_bogo_count , valid_dscut_count ,invalid_dscut_count], 
                             index=[ 'valid_bogo_count','invalid_bogo_count',
                                    'valid_dscut_count' , 'invalid_dscut_count'], 
                             name=profile_id)
    return valid_series


def load_file(filepath):
    """Load file csv"""
    df_clean = pd.read_csv(filepath)
    df_clean = df_clean.set_index(df_clean.columns[0])
    df_clean = profile_parse_dates(df_clean)
    return df_clean

# 函数，特征提取
# 输入   合并后的信息 df_merge。 以及 处理以后的用户信息 df_prof_clean
# 输出  给用户信息增加更多特征
# 思路 1 对 df_merge里 每个用户遍历，得到用户子df
# 2 用之前的函数 建立用户特征 series
# 3 Ser转置为一行 和原来的 df_prof_clean 通过id 连接到一起
# 4 更新后的用户 profile_updated 存起来
def feature_extraction( df_merge, df_prof_clean):

    # df_merge = dfmerge_trprpo ; profile_df = profile
    try:
        profile_updated = load_file('data/profile_updated.csv')
        print("The profile_updated.csv file is available at local folder.")
    except:
        
        list_addfeat = ["avg_spending", "transaction_count", 'sum_spending',
                       'valid_bogo_count','invalid_bogo_count','valid_dscut_count' , 'invalid_dscut_count']
        allfeat_df = pd.DataFrame(index=list_addfeat)

        ar_personid = df_merge.person.unique()
        for profile_id in tqdm( ar_personid ):
            # profile_id = ar_personid[0]
            subset_df = df_merge[df_merge.person == profile_id]
            
            # 特征提取 消费信息 和无效完成信息
            spending_series = addfeat_spending( subset_df, profile_id )
            allfeat_df.loc[ spending_series.index , profile_id ] = spending_series.values
            invalid_series  = addfeat_invalid(  subset_df, profile_id ) 
            allfeat_df.loc[ invalid_series.index , profile_id] = invalid_series.values     
            
        # df concatenation
        profile_updated = pd.concat([df_prof_clean.set_index('id'),allfeat_df.T ], axis=1, sort=False)

    return profile_updated



# -

# # 特征工程

# ## 增加特征

# +

# 合并
dfmerge_trprpo = merge_transcript_profile_portfolio(transcript, profile, portfolio)
# 找到并排除 invalid
dfmerge_trprpo.loc[ : ,'valid'] = 0
dfmerge_trprpo.loc[ list_validcomp ,'valid'] =1
dfmerge_trprpo.loc[ : ,'invalid'] = 0
dfmerge_trprpo.loc[ list_invalidcomp ,'invalid'] =1
dfmerge_trprpo.loc[ list_invalidcomp ,'invalid'] =1

dfmerge_trprpo




# +

if  os.path.exists('sav/profile_updated.pkl') :
    profile_updated =  pklload('sav/profile_updated.pkl')

else:    
    # 利用 valid 数据做特征提取
    df_prof_clean  = profile_prep(profile)
    profile_updated = feature_extraction( dfmerge_trprpo, df_prof_clean )    
    # saving
    profile_updated.to_csv('data/profile_updated.csv')

# -

profile_updated

# ## 缺失

# +
#%% 看看每列缺失多少   Assess missing data in columns

col_null = profile_updated.isnull().sum()

col_null_frac = col_null / profile_updated.shape[0]
plt.figure(figsize= (10,5))
col_null_frac.plot(kind='bar')

# +
#%%  找到并删除 超过0.3比例的空缺列

# cols to drop that have more than 30% missing values
cols_to_drop = col_null_frac[col_null_frac > 0.3].index.tolist()
cols_to_drop

# drop columns in cols_to_drop
profile_up_nomiss = profile_updated.drop(cols_to_drop, axis=1)

# +
#%% 看看各行 缺失 多少

row_null = profile_up_nomiss.isnull().sum(axis=1)
row_null.hist()


# +
# 观察有缺失行和没缺失行 关心的预测列 hist分布
  
def dist_compare_cont(attribute, data):
    """
    input : 
        attribute: feature / attribute
        data : dataframe
    return : None, only plot the histogram
    """
    # data = profile_updated_  ;  attribute = 'avg_spending'
    row_null = data.isnull().sum(axis=1)
    f, (ax1,ax2) = plt.subplots(1,2, sharex=True, figsize=(8,3))
    ax1 = data[row_null==0][attribute].hist(ax=ax1, bins=20)
    ax2 = data[row_null > 0][attribute].hist( ax=ax2, bins=20)
    ax1.set_title('No missing value')
    ax2.set_title('With missing value')
    f.suptitle('Feature: ' + attribute)
    f.tight_layout()
    f.subplots_adjust(top=0.8)


cols_to_compare = ['became_member_on', 'avg_spending', 'transaction_count', 'sum_spending']

for col in cols_to_compare:
    dist_compare_cont(col, profile_up_nomiss)
# -

# - 发现 No missing 和 With missing 两者相比 没有明显差别 可以考虑删除
#     
#     
# - It is found that data with missing and without missing have no obvious difference, the missing rows can be deleted

#%% 删除这部分缺失的行 
profile_up_nomiss = profile_up_nomiss.drop( profile_up_nomiss[row_null > 0].index  , axis=0)
profile_up_nomiss.isnull().sum().sum()

# ## one hot: gender, member_year

# +
#%% 再次 one hot 编码，主要是 gender 
profile_new = pd.concat( [ profile_up_nomiss , pd.get_dummies(profile_up_nomiss['gender'])] ,axis= 1 ) 
profile_new= profile_new.drop(columns = ['became_member_on','member_year','age','gender'] , axis=1)

profile_new.hist( figsize=(20,20))

# -

profile_new['[110, 120)'].value_counts()
profile_new= profile_new.drop(columns = ['[110, 120)'] , axis=1)


# ## save结果  profile_new

#%% 保存数据
pklsave(profile_new, 'sav/profile_new.sav')


# ##  Data analysis 

# +
#%% ！！！数据分析 ，热图 EDA 指定列，分别针对<100 >100

cols = [ 'income', 'avg_spending', 'transaction_count', 'sum_spending',
       'valid_bogo_count',   'invalid_bogo_count',   'valid_dscut_count', 'invalid_dscut_count' ,
        'mbyr_2013', 'mbyr_2014', 'mbyr_2015', 'mbyr_2016','mbyr_2017', 'mbyr_2018']
f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(profile_new[cols].corr(), ax=ax, annot=True)
ax.set_title("Profile Main - Features Correlation")

# -

# - 除了 income 和 avg_spending 以外，这些特征之间没有很强的关联性
#
#    

# # 构造数据集 预测用户收到offer后是否会完成　

# ## 整理所有offer received的信息

list_validcomp =  pklload('sav/list_validcomp.pkl')
list_invalidcomp =  pklload('sav/list_invalidcomp.pkl')
list_validrec =  pklload('sav/list_validrec.pkl')
profile_updated =  pklload('sav/profile_updated.pkl')
profile_new  = pklload('sav/profile_new.sav')


dfmerge_trprpo = merge_transcript_profile_portfolio(transcript, profile, portfolio)
df_rec_all = dfmerge_trprpo.loc[ dfmerge_trprpo.event == 'offer received',['person','offer_id']]
# 删除 offer_id 和 person 重复的部分
df_rec_new = df_rec_all.drop(  df_rec_all[df_rec_all.duplicated()].index  )
# 确认 重复的部分里 没有 list_validrec
np.intersect1d( df_rec_all[df_rec_all.duplicated()].index , list_validrec )


# 添加 validrec 特征
df_rec_new.loc[ :, 'validrec' ] = 0
df_rec_new.loc[ list_validrec, 'validrec' ] =1


# 根据 profile_new 添加 user的特征
df_rec_new = pd.merge(df_rec_new, profile_new, left_on=['person'], right_index=True, how ='inner')
# 根据 portfolio_clean 添加 offer 的特征
portfolio_clean = portfolio_prep( portfolio ) 
df_rec_new = pd.merge(df_rec_new, portfolio_clean, left_on=['offer_id'], right_on=['id'], how ='inner')


# ## 删除不必要特征

# +
# 删除不必要的 offer 特征
df_rec_new = df_rec_new.drop( columns=['offer_id','id'] )

# 由于 imformation 的 offer是没有完成结果的 不在预测范围内 也删除
index_infor = df_rec_new.loc[ df_rec_new.offername.str.contains('infor'),: ].index
df_rec_new = df_rec_new.drop( index_infor )
df_rec_new.offername.value_counts()

# offername 变为 onehot 特征
df_rec_new = pd.concat( [ df_rec_new , pd.get_dummies(df_rec_new['offername'])] ,axis= 1 ) 
df_rec_new = df_rec_new.drop( columns= 'offername' )
df_rec_new

# offer_type 变为 onehot 特征
df_rec_new = pd.concat( [ df_rec_new , pd.get_dummies(df_rec_new['offer_type'], prefix='tp')] ,axis= 1 ) 
df_rec_new = df_rec_new.drop( columns= 'offer_type' )
df_rec_new
# -

# 删除不必要的 user特征
df_rec_new = df_rec_new.drop( columns= 'person' )
df_rec_new.columns
len(df_rec_new.columns) 

# 继续删除更多 user 特征
df_rec_clean = df_rec_new.drop( columns= [ 'avg_spending', 'transaction_count',
       'sum_spending', 'valid_bogo_count', 'invalid_bogo_count',
       'valid_dscut_count', 'invalid_dscut_count']  )
df_rec_clean.columns
len(df_rec_clean.columns) 

# 删除 offer 特征
df_rec_clean = df_rec_clean.drop( columns= [ 'bogo_10_5_10', 'bogo_10_7_10', 'bogo_5_5_5', 'bogo_5_7_5',
       'discount_10_10_2', 'discount_10_7_2', 'discount_20_10_5',
       'discount_7_7_3']  )
df_rec_clean.columns
len(df_rec_clean.columns) 


# ## 数据清洗函数

def process_datafeature( df_merge, df_offer, df_profile, list_validrec):
    # df_merge= dfmerge_trprpo ;  df_offer = portfolio_clean; df_profile = profile_updated
    dfmerge_trprpo = df_merge.copy()
    portfolio_clean = df_offer.copy()
    profile_updated = df_profile.copy()
    
    # 添加validrec特征
    df_rec_all = dfmerge_trprpo.loc[ dfmerge_trprpo.event == 'offer received',['person','offer_id']]
    df_rec_new = df_rec_all.drop(  df_rec_all[df_rec_all.duplicated()].index  )
    df_rec_new.loc[ :, 'validrec' ] = 0
    df_rec_new.loc[ list_validrec, 'validrec' ] =1
    # 合并 用户和offer信息
    df_rec_new = pd.merge(df_rec_new, profile_updated, left_on=['person'], right_index = True, how ='inner')
    df_rec_new = pd.merge(df_rec_new, portfolio_clean, left_on=['offer_id'], right_on=['id'], how ='inner')
    
    # 某些列进行 onehot ，删除不需要的列
    index_infor = df_rec_new.loc[ df_rec_new.offername.str.contains('infor'),: ].index
    df_rec_new = df_rec_new.drop( index_infor )
    df_rec_new = pd.concat( [ df_rec_new , pd.get_dummies(df_rec_new['gender'])] ,axis= 1 ) 
    df_rec_new = pd.concat( [ df_rec_new , pd.get_dummies(df_rec_new['offername'])] ,axis= 1 ) 
    df_rec_new = pd.concat( [ df_rec_new , pd.get_dummies(df_rec_new['offer_type'], prefix='tp')] ,axis= 1 ) 

    df_rec_new= df_rec_new.drop(columns = ['became_member_on','member_year','age','gender'] , axis=1)
    df_rec_new= df_rec_new.drop(columns = ['[110, 120)'] , axis=1)

    df_rec_new = df_rec_new.drop( columns=['offer_id','id'] )
    df_rec_new = df_rec_new.drop( columns= 'offer_type' )
    df_rec_new = df_rec_new.drop( columns= 'offername' )
    df_rec_new = df_rec_new.drop( columns= 'person' )
    df_rec_new = df_rec_new.drop( columns= [  'valid_bogo_count', 'invalid_bogo_count',
           'valid_dscut_count', 'invalid_dscut_count']  )
    df_rec_new = df_rec_new.drop( columns= [ 'bogo_10_5_10', 'bogo_10_7_10', 'bogo_5_5_5', 'bogo_5_7_5',
           'discount_10_10_2', 'discount_10_7_2', 'discount_20_10_5',
           'discount_7_7_3']  )    
    
    # 删除空列和空行
    col_null = df_rec_new.isnull().sum()
    col_null_frac = col_null / df_rec_new.shape[0]
    cols_to_drop = col_null_frac[col_null_frac > 0.3].index.tolist()
    df_rec_nomiss = df_rec_new.drop(cols_to_drop, axis=1)
    row_null = df_rec_nomiss.isnull().sum(axis=1)
    df_rec_nomiss = df_rec_nomiss.drop( df_rec_nomiss[row_null > 0].index  , axis=0)
       
    return df_rec_nomiss


df_rec_new  = process_datafeature( dfmerge_trprpo, portfolio_clean, profile_updated, list_validrec)

df_rec_new.columns
len( df_rec_new.columns )

pklsave(df_rec_new, 'sav/df_rec_new.pkl')


# ## 分割数据集

# +

df_rec_new  = pklload('sav/df_rec_new.pkl')

random_state = 42

label_name = 'validrec'
variables = df_rec_new.drop(columns=[label_name])
label = df_rec_new.filter([label_name])

(X_train, X_test, y_train, y_test) = train_test_split(variables.values,
                                        label.values,
                                        test_size=0.2,
                                        random_state=random_state)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.ravel()
y_test = y_test.ravel()

X_train.shape


# -

# # 预测模型

# ### 函数 评估模型
#

def evaluate_model_performance(clf, X,y):
    """ Prints a model's accuracy and F1-score
    
    INPUT:
        clf: Model object
        X: Training data matrix
        y: Expected model output vector
    
    OUTPUT:
        clf_accuracy: Model accuracy
        clf_f1_score: Model F1-score"""
    
    # clf = lr_random.best_estimator_
    # clf = pipe_lr_clf
    #  class_name 就是单纯的 把很长的模型名字 sklearn.linear_model._logistic.LogisticRegression 
    # 给简化为 'LogisticRegression'而已
    
    y_pred = clf.predict(X)
    fbt_score = fbeta_score( y , y_pred, beta=2)
    print(  'F-beta score: {0:.3f} \n'.format(fbt_score ))
    print( classification_report(y, y_pred))
    return


# ##   线性分类模型

# +
model_path = os.path.join('sav/pipe_lrsvc.joblib')

if os.path.exists(model_path):
    pipe_lrsvc = load(model_path)
else:
    # 建立模型
    pipe_lrsvc = Pipeline([    ('scaler', MinMaxScaler()), ('clf', LinearSVC()) ])
    pipe_lrsvc.fit(X_train, y_train)
    dump(pipe_lrsvc, model_path)
# -


evaluate_model_performance( pipe_lrsvc ,X_test, y_test)

# ### 线性分类模型调参

pipe_lrsvc.get_params()


# +
model_path = os.path.join('sav/cv_lrsvc.joblib')

if os.path.exists(model_path):
    cv_lrsvc = load(model_path)
else:
    
    # 设置参数矩阵
    parameters = {
        'clf__dual': [True, False],
        'clf__tol': [1e-4, 1e-5],
        'clf__C':[0.8, 1, 1.2],
        'clf__max_iter': [ 1e3, 3e3, 1e4]  }

    # 评价指标
    scorer = make_scorer( fbeta_score, beta=2 )
    cv_lrsvc = GridSearchCV( pipe_lrsvc , param_grid=parameters,   scoring= scorer )

    # 再训练
    cv_lrsvc.fit( X_train, y_train )
    dump(cv_lrsvc, model_path)
    
# 查看最佳参数
print( cv_lrsvc.best_params_)



# -

# 测试模型结果
evaluate_model_performance(cv_lrsvc, X_test, y_test)

# 调参后 F-beta score 还是0.746 没有变化 

# ##  K近邻算法

# +
model_path = os.path.join('sav/pipe_kmn.joblib')

if os.path.exists(model_path):
    pipe_kmn = load(model_path)
else:
    pipe_kmn = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', KNeighborsClassifier())
    ])
    pipe_kmn.get_params()
    pipe_kmn.fit(X_train, y_train)
    dump(pipe_kmn, model_path)


# -


evaluate_model_performance(pipe_kmn,X_test, y_test )

# ### K近邻算法调参

# +
model_path = os.path.join('sav/cv_kmn.joblib')

if os.path.exists(model_path):
    cv_kmn = load(model_path)
else:
    
    # 设置参数矩阵
    parameters = {
        'clf__leaf_size': [10, 30, 100],
        'clf__n_neighbors': [5, 20] ,
        'clf__weights':[ 'uniform', 'distance']
    }
    
    scorer = make_scorer(fbeta_score,beta=2)
    cv_kmn = GridSearchCV(pipe_kmn, param_grid=parameters , scoring= scorer )

    # 再训练
    cv_kmn.fit(X_train, y_train)
    dump(cv_kmn, model_path)



# +
# 查看最佳参数
cv_kmn.best_params_

# 测试模型结果
evaluate_model_performance(cv_kmn,X_test, y_test )
# -

# K临近算法 调参后 F-beta score 还是0.741 没有变化

# ## 随机森林模型

# +
model_path = os.path.join('sav/pipe_rf.joblib')

if os.path.exists(model_path):
    pipe_rf = load(model_path)
else:
    pipe_rf = Pipeline([('sc', StandardScaler()),
                      ('rf_clf', RandomForestClassifier(random_state=random_state))
                      ])
    pipe_rf.fit(X_train, y_train)
    dump(pipe_rf, model_path)

# -

evaluate_model_performance(pipe_rf,X_test, y_test )

pipe_rf.get_params()

# ### 随机森林调参

# +
model_path = os.path.join('sav/cv_rf.joblib')

if os.path.exists(model_path):
    cv_rf = load(model_path)
else:
    
    # 设置参数矩阵
    parameters = {
         'rf_clf__n_estimators': [ 50, 100, 200],
         'rf_clf__max_depth': [5,10,15],
        'rf_clf__min_samples_split': [10,50,100],
         'rf_clf__min_samples_leaf': [  2, 3, 5] }
    scorer = make_scorer(fbeta_score,beta=2)
    cv_rf = GridSearchCV(pipe_rf, param_grid= parameters , scoring=scorer )

    # 再训练
    cv_rf.fit(X_train, y_train)
    dump(cv_rf, model_path)

# -

evaluate_model_performance(cv_rf,X_test, y_test )

cv_rf.best_params_


# - 随机森林模型调参后 f_beta分数从 0.784 增加到0.813

# ## Gradient Boosting classifier模型

# +
model_path = os.path.join('sav/pipe_gb.joblib')

if os.path.exists(model_path):
    pipe_gb = load(model_path)
else:
    pipe_gb = Pipeline([('sc', StandardScaler()),
                      ('gb_clf', GradientBoostingClassifier(random_state=random_state))
                      ])
    pipe_gb.fit(X_train, y_train)
    dump(pipe_gb, model_path)
    
# -

evaluate_model_performance(pipe_gb,X_test, y_test )
pipe_gb.get_params()

# ### Gradient Boosting模型调参

# +
model_path = os.path.join('sav/cv_gb.joblib')

if os.path.exists(model_path):
    cv_gb = load(model_path)
else:
    
    # 设置参数矩阵
    parameters = {
        'gb_clf__learning_rate': [0.1, 0.05, 0.01],
        'gb_clf__n_estimators': [50, 100, 200] ,
        'gb_clf__max_depth':  [5,10,15] ,
        'gb_clf__min_samples_split': [10,50,100],
  	  	 'gb_clf__min_samples_leaf': [ 1,  3, 5] }
        
#     scorer = make_scorer(fbeta_score,beta=2)
    scorer = make_scorer(fbeta_score,beta=2)
    cv_gb = GridSearchCV(pipe_gb, param_grid= parameters , scoring= scorer )

    # 再训练
    cv_gb.fit(X_train, y_train)
    dump(cv_gb, model_path)

# -

evaluate_model_performance(cv_gb,X_test, y_test )
cv_gb.best_params_

# - GBM 模型调参后 f_beta分数从 0.817 增加到0.841

# ## 特征重要性

# +
variable_names = df_rec_new.drop(columns=[label_name]).columns

relative_importance = cv_gb.best_estimator_.steps[1][1].feature_importances_
relative_importance = relative_importance / np.sum(relative_importance)

feature_importance =\
    pd.DataFrame(list(zip(variable_names,
                          relative_importance)),
                 columns=['feature', 'relativeimportance'])

feature_importance = feature_importance.sort_values('relativeimportance',
                                                    ascending=False)

feature_importance = feature_importance.reset_index(drop=True)

palette = sns.color_palette("Blues_r", feature_importance.shape[0])

plt.figure(figsize=(8, 8))
sns.barplot(x='relativeimportance',
            y='feature',
            data=feature_importance,
            palette=palette)
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.title('Random Forest Estimated Feature Importance')
# -

feature_importance.head(n=10)

# - 决定优惠推送对用户是否有效的最关键因素是 用户总消费金额，其次是社交媒体广告，和活动奖励

# ## 结果汇总

# +
df_model_perfm = pd.DataFrame( columns= ['initial','best_params'])

df_model_perfm.loc['LinearSVC',:] = [ 0.746 ,0.746]
df_model_perfm.loc['KNeighbors',:] = [0.741, 0.741 ]
df_model_perfm.loc['RandomForest',:] = [0.784, 0.813 ]
df_model_perfm.loc['GradientBoosting',:] = [ 0.817 , 0.841]
df_model_perfm
# -

# # 结论

# - 我建立了一个判断某类活动信息是否对用户有效果，即用户在受到活动信息后，是否会采取交易行为的模型。   
# - 我解决这个问题分为四个步骤。
#   - 首先，我把优惠活动信息，客户资料和客户交易数据结合到一起，根据客户交易数据提取了更多的客户特征。
#   - 然后，我根据客户交易信息和优惠活动信息，找出哪些推送活动在有效期内交易（有效完成），哪些推送后没有交易或者在期限外交易（无效完成）。
#   - 接着，我比较了线性分类，K临近算法，随机森林和梯度提升模型的性能。结果表明，梯度提升模型具有最佳的F_beta分数,在测试集上为0.841。
#   - 梯度提升模型最优参数 'clf__learning_rate': 0.01, 'clf__max_depth': 5, 'clf__min_samples_leaf': 3, 'clf__min_samples_split': 10, 'clf__n_estimators': 50
# - 遇到的问题和挑战：
#   - 最初按照回归问题建立的模型 用来预测每个用户有效完成推送活动的概率，但是样本较少导致预测结果不好。后来改为按照分类模型预测用户是否会有效完成，模型表现提升很多
#   - 比较麻烦的是根据每个活动时间限制和用户在该时间内 是否有"接收-浏览-完成"这一系列行为，判断是否为有效完成。
# - 可以改进的方面：
#   - 上述分类模型 只是判断了特定活动推送对于特定客户是否有效，但是由于缺乏足够信息，没有考察推送次数这一特征和用户有效完成的关系。还可以建立一个回归模型，预测不同推送活动对于不同用户的完成率，从而找出哪些用户在受到信息后可以多次有效消费。


