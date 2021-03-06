# Starbuck_offer_prediction
* 这个数据集是一些模拟 Starbucks rewards 移动 app 上用户行为的数据。每隔几天，星巴克会向 app 的用户发送一些推送。这个推送可能仅仅是一条饮品的广告或者是折扣券或 BOGO（买一送一）。一些顾客可能一连几周都收不到任何推送。我的任务是将交易数据、人口统计数据和推送数据结合起来，预测某个用户会不会受到app推送活动的影响。
* 用到的库 pandas，numpy，math，json, seaborn, matplotlib, tqdm, datetime, pickle, sklearn,warnings
* 文件 Starbucks_Capstone_notebook.ipynb

* 步骤：
  * 1. 读入3个数据
    * portfolio.json – 包括推送的 id 和每个推送的元数据（持续时间、种类等等）

|    |   reward | channels                             |   difficulty |   duration | offer_type    | id                               |
|---:|---------:|:-------------------------------------|-------------:|-----------:|:--------------|:---------------------------------|
|  0 |       10 | ['email', 'mobile', 'social']        |           10 |          7 | bogo          | ae264e3637204a6fb9bb56bc8210ddfd |
|  1 |       10 | ['web', 'email', 'mobile', 'social'] |           10 |          5 | bogo          | 4d5c57ea9a6940dd891ad53e9dbe8da0 |
|  2 |        0 | ['web', 'email', 'mobile']           |            0 |          4 | informational | 3f207df678b143eea3cee63160fa8bed |
|  3 |        5 | ['web', 'email', 'mobile']           |            5 |          7 | bogo          | 9b98b8c7a33c4b65b9aebfe6a799e6d9 |
|  4 |        5 | ['web', 'email']                     |           20 |         10 | discount      | 0b1e1539f2cc45b7b9fa7c272da2e1d7 |

    * profile.json – 每个顾客的人口统计数据，
    
|    | gender   |   age | id                               |   became_member_on |   income |
|---:|:---------|------:|:---------------------------------|-------------------:|---------:|
|  0 |          |   118 | 68be06ca386d4c31939f3a4f0e3dd783 |           20170212 |      nan |
|  1 | F        |    55 | 0610b486422d4921ae7d2bf64640c50b |           20170715 |   112000 |
|  2 |          |   118 | 38fe809add3b4fcf9315a9694bb96ff5 |           20180712 |      nan |
|  3 | F        |    75 | 78afa995795e4d85b5d9ceeca43f5fef |           20170509 |   100000 |
|  4 |          |   118 | a03223e636434f42ac4c3df47e8bac43 |           20170804 |      nan |
    
    * transcript.json – 交易、收到的推送、查看的推送和完成的推送的记录
    
|    | person                           | event          | value                                            |   time |
|---:|:---------------------------------|:---------------|:-------------------------------------------------|-------:|
|  0 | 78afa995795e4d85b5d9ceeca43f5fef | offer received | {'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'} |      0 |
|  1 | a03223e636434f42ac4c3df47e8bac43 | offer received | {'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'} |      0 |
|  2 | e2127556f4f64592b11af22de27a7932 | offer received | {'offer id': '2906b810c7d4411798c6938adc9daaa5'} |      0 |
|  3 | 8ec6ce2a7e7949b1bf142def7d0e0586 | offer received | {'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'} |      0 |
|  4 | 68617ca6246f4fbc85e91a2a49552598 | offer received | {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'} |      0 |

  * 2. 对类型数据进行onehot编码，把时间数据转化为datetime，分析异常值和缺失值，删除对数据分布没有影响的异常和缺失值，把dict类数据拆成多列，把相似的字符进行统一
  
|    |   reward |   difficulty |   duration | offer_type    | id                               |   web |   email |   mobile |   social | offername           |
|---:|---------:|-------------:|-----------:|:--------------|:---------------------------------|------:|--------:|---------:|---------:|:--------------------|
|  0 |       10 |           10 |          7 | bogo          | ae264e3637204a6fb9bb56bc8210ddfd |     0 |       1 |        1 |        1 | bogo_10_7_10        |
|  1 |       10 |           10 |          5 | bogo          | 4d5c57ea9a6940dd891ad53e9dbe8da0 |     1 |       1 |        1 |        1 | bogo_10_5_10        |
|  2 |        0 |            0 |          4 | informational | 3f207df678b143eea3cee63160fa8bed |     1 |       1 |        1 |        0 | informational_0_4_0 |
|  3 |        5 |            5 |          7 | bogo          | 9b98b8c7a33c4b65b9aebfe6a799e6d9 |     1 |       1 |        1 |        0 | bogo_5_7_5          |
|  4 |        5 |           20 |         10 | discount      | 0b1e1539f2cc45b7b9fa7c272da2e1d7 |     1 |       1 |        0 |        0 |

|    | gender   |   age | id                               | became_member_on    |   income |
|---:|:---------|------:|:---------------------------------|:--------------------|---------:|
|  0 |          |   118 | 68be06ca386d4c31939f3a4f0e3dd783 | 2017-02-12 00:00:00 |      nan |
|  1 | F        |    55 | 0610b486422d4921ae7d2bf64640c50b | 2017-07-15 00:00:00 |   112000 |
|  2 |          |   118 | 38fe809add3b4fcf9315a9694bb96ff5 | 2018-07-12 00:00:00 |      nan |
|  3 | F        |    75 | 78afa995795e4d85b5d9ceeca43f5fef | 2017-05-09 00:00:00 |   100000 |
|  4 |          |   118 | a03223e636434f42ac4c3df47e8bac43 | 2017-08-04 00:00:00 |      nan |

|    | person                           | event          |   time | offer_id                         |   amount |
|---:|:---------------------------------|:---------------|-------:|:---------------------------------|---------:|
|  0 | 78afa995795e4d85b5d9ceeca43f5fef | offer received |      0 | 9b98b8c7a33c4b65b9aebfe6a799e6d9 |      nan |
|  1 | a03223e636434f42ac4c3df47e8bac43 | offer received |      0 | 0b1e1539f2cc45b7b9fa7c272da2e1d7 |      nan |
|  2 | e2127556f4f64592b11af22de27a7932 | offer received |      0 | 2906b810c7d4411798c6938adc9daaa5 |      nan |
|  3 | 8ec6ce2a7e7949b1bf142def7d0e0586 | offer received |      0 | fafdcd668e3743c1bb461111dcafc2a4 |      nan |
|  4 | 68617ca6246f4fbc85e91a2a49552598 | offer received |      0 | 4d5c57ea9a6940dd891ad53e9dbe8da0 |      nan |

  * 3. 将3个处理好的数据集 通过用户id 和 推送offer id，进行整合，得到每个用户的用户信息+交易信息+推送活动信息
  * 4. 增加用户特征，比如总交易量、平均交易量、交易次数、不同推送活动的完成情况、无效完成情况等等
  * 5. 根据每个用户的推送活动id 和活动信息，判断该推送有效时间以内是否成功交易，如果成功交易，则对应的offer recieve推送记为 validrec =1，如果没有交易或者在推送时间以外交易，则 validrec =0。
  
  |    | person                           | event          |   time | offer_id                         |   amount | gender   |   age | became_member_on    |   income | member_year   |   mbyr_2013 |   mbyr_2014 |   mbyr_2015 |   mbyr_2016 |   mbyr_2017 |   mbyr_2018 |   [10, 20) |   [100, 110) |   [110, 120) |   [20, 30) |   [30, 40) |   [40, 50) |   [50, 60) |   [60, 70) |   [70, 80) |   [80, 90) |   [90, 100) |   reward |   difficulty |   duration | offer_type   |   web |   email |   mobile |   social | offername        |   valid |   invalid |
|---:|:---------------------------------|:---------------|-------:|:---------------------------------|---------:|:---------|------:|:--------------------|---------:|:--------------|------------:|------------:|------------:|------------:|------------:|------------:|-----------:|-------------:|-------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|---------:|-------------:|-----------:|:-------------|------:|--------:|---------:|---------:|:-----------------|--------:|----------:|
|  0 | 78afa995795e4d85b5d9ceeca43f5fef | offer received |      0 | 9b98b8c7a33c4b65b9aebfe6a799e6d9 |      nan | F        |    75 | 2017-05-09 00:00:00 |   100000 | mbyr_2017     |           0 |           0 |           0 |           0 |           1 |           0 |          0 |            0 |            0 |          0 |          0 |          0 |          0 |          0 |          1 |          0 |           0 |        5 |            5 |          7 | bogo         |     1 |       1 |        1 |        0 | bogo_5_7_5       |       0 |         0 |
|  1 | a03223e636434f42ac4c3df47e8bac43 | offer received |      0 | 0b1e1539f2cc45b7b9fa7c272da2e1d7 |      nan |          |   118 | 2017-08-04 00:00:00 |      nan | mbyr_2017     |           0 |           0 |           0 |           0 |           1 |           0 |          0 |            0 |            1 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |           0 |        5 |           20 |         10 | discount     |     1 |       1 |        0 |        0 | discount_20_10_5 |       0 |         0 |
|  2 | e2127556f4f64592b11af22de27a7932 | offer received |      0 | 2906b810c7d4411798c6938adc9daaa5 |      nan | M        |    68 | 2018-04-26 00:00:00 |    70000 | mbyr_2018     |           0 |           0 |           0 |           0 |           0 |           1 |          0 |            0 |            0 |          0 |          0 |          0 |          0 |          1 |          0 |          0 |           0 |        2 |           10 |          7 | discount     |     1 |       1 |        1 |        0 | discount_10_7_2  |       0 |         0 |
|  3 | 8ec6ce2a7e7949b1bf142def7d0e0586 | offer received |      0 | fafdcd668e3743c1bb461111dcafc2a4 |      nan |          |   118 | 2017-09-25 00:00:00 |      nan | mbyr_2017     |           0 |           0 |           0 |           0 |           1 |           0 |          0 |            0 |            1 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |           0 |        2 |           10 |         10 | discount     |     1 |       1 |        1 |        1 | discount_10_10_2 |       0 |         0 |
|  4 | 68617ca6246f4fbc85e91a2a49552598 | offer received |      0 | 4d5c57ea9a6940dd891ad53e9dbe8da0 |      nan |          |   118 | 2017-10-02 00:00:00 |      nan | mbyr_2017     |           0 |           0 |           0 |           0 |           1 |           0 |          0 |            0 |            1 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |           0 |       10 |           10 |          5 | bogo         |     1 |       1 |        1 |        1 | bogo_10_5_10     |       0 |         0 |

  * 6. 进行数据分析，观察用户特征 如性别，年龄和交易行为之间的关系
  * 7. 将不同的推送offer特征，和不同的用户特征整合到一起，作为数据集，预测这些offer对于特定用户的validrec，即用户是否会完成推送offer的交易活动
  
  |    |   validrec |   income |   mbyr_2013 |   mbyr_2014 |   mbyr_2015 |   mbyr_2016 |   mbyr_2017 |   mbyr_2018 |   [10, 20) |   [100, 110) |   [20, 30) |   [30, 40) |   [40, 50) |   [50, 60) |   [60, 70) |   [70, 80) |   [80, 90) |   [90, 100) |   avg_spending |   transaction_count |   sum_spending |   reward |   difficulty |   duration |   web |   email |   mobile |   social |   F |   M |   O |   tp_bogo |   tp_discount |
|---:|-----------:|---------:|------------:|------------:|------------:|------------:|------------:|------------:|-----------:|-------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|---------------:|--------------------:|---------------:|---------:|-------------:|-----------:|------:|--------:|---------:|---------:|----:|----:|----:|----------:|--------------:|
|  0 |          1 |   100000 |           0 |           0 |           0 |           0 |           1 |           0 |          0 |            0 |          0 |          0 |          0 |          0 |          0 |          1 |          0 |           0 |       22.7529  |                   7 |         159.27 |        5 |            5 |          7 |     1 |       1 |        1 |        0 |   1 |   0 |   0 |         1 |             0 |
|  1 |          1 |    70000 |           0 |           0 |           0 |           0 |           0 |           1 |          0 |            0 |          0 |          0 |          0 |          0 |          1 |          0 |          0 |           0 |       19.2433  |                   3 |          57.73 |        5 |            5 |          7 |     1 |       1 |        1 |        0 |   0 |   1 |   0 |         1 |             0 |
|  3 |          0 |    53000 |           0 |           0 |           0 |           0 |           0 |           1 |          0 |            0 |          0 |          0 |          0 |          0 |          1 |          0 |          0 |           0 |       12.1433  |                   3 |          36.43 |        5 |            5 |          7 |     1 |       1 |        1 |        0 |   0 |   1 |   0 |         1 |             0 |
|  4 |          0 |    88000 |           0 |           0 |           0 |           0 |           0 |           1 |          0 |            0 |          0 |          0 |          0 |          1 |          0 |          0 |          0 |           0 |       25.2067  |                   3 |          75.62 |        5 |            5 |          7 |     1 |       1 |        1 |        0 |   1 |   0 |   0 |         1 |             0 |
|  6 |          0 |    41000 |           0 |           0 |           1 |           0 |           0 |           0 |          0 |            0 |          0 |          0 |          0 |          1 |          0 |          0 |          0 |           0 |        4.00615 |                  13 |          52.08 |        5 |            5 |          7 |     1 |       1 |        1 |        0 |   0 |   1 |   0 |         1 |             0 |

  * 8. 评估了线性分类、K临近算法、随机森林和梯度提升模型的性能。
  
* 数据清洗和分析结果：
  * 对于 profile数据，发现118岁的用户有2175名，占总数据的14%，年龄值异常，并且在income和gender上都缺失了信息，在通过特征工程拓展特征以后，缺失和无缺失的分布基本一致，可以删除缺失行。
  * 对于 transcript数据，value 列有两种情况 'offer id' 或者'offer_id' 字典，值为offerid，'amount'字典，值为交易量，因此这一列需要拆成2列，分别用'offer_id'和'amount'作为列名，值为对应的字典值，注意需要把'offer id'变为'offer_id'
  * 对于bogo_5_7_5和discount_20_10_5这两种推送，complete 竟然比 view还多 说明有的人 complete 和推送无关, 属于无效记录。排除无效记录后，对所有推送均有 complete 比 view 少，情况合理。
  * age和income 基本符合正态分布，became_member人数 每年逐渐增多，女性平均收入较高 其次是无性别的 
  * 用户单次交易的平均值是16元，大部分都在20元以下；用户交易量的平均值是125元，大部分都在150元以下；用户交易次数的平均值是8次，大部分都在11次以下。
  * 女性的平均交易量最多, 其次是其他性别和男性。男性的交易次数最多, 其次是其他性别和女性。男性女性的总交易量比较接近,可能是因为男性虽然单次交易量少,但是人数多.其他性别和unknown的总交易量都很少。
  * 除了 income 和 avg_spending 以外，这些特征之间没有很强的关联性。
  
* 预测模型结果：
  * 结果表明，梯度提升模型具有最佳的F_beta分数,在测试集上为0.841。
  * 梯度提升模型参数选择 'clf__learning_rate': [0.1, 0.05, 0.01],  'clf__n_estimators': [50, 100, 200] , 'clf__max_depth':  [5,10,15] , 'clf__min_samples_split': [10,50,100], 'clf__min_samples_leaf': [ 1,  3, 5] 
  * 梯度提升模型 最优参数 'clf__learning_rate': 0.01, 'clf__max_depth': 5, 'clf__min_samples_leaf': 3, 'clf__min_samples_split': 10, 'clf__n_estimators': 50
  * 决定优惠推送对用户是否有效的最关键因素是 用户总消费金额，其次是社交媒体广告，和活动奖励
  
* 遇到的问题和挑战：
最初按照回归问题建立的模型 用来预测每个用户有效完成推送活动的概率，但是样本较少导致预测结果不好。后来改为按照分类模型预测用户是否会有效完成，模型表现提升很多
比较麻烦的是根据每个活动时间限制和用户在该时间内 是否有"接收-浏览-完成"这一系列行为，判断是否为有效完成。

* 可以改进的方面：
上述分类模型 只是判断了特定活动推送对于特定客户是否有效，但是由于缺乏足够信息，没有考察推送次数这一特征和用户有效完成的关系。还可以建立一个回归模型，预测不同推送活动对于不同用户的完成率，从而找出哪些用户在受到信息后可以多次有效消费。


