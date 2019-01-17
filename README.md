# market_impact
以Kissell Research Group(JP Morgan) 的I-star市场冲击模型为理论基础，详见 http://www.kissellresearch.com/krg-i-star-market-impact-model
对沪深两市指数和个股进行模型参数拟合，使用通联1分钟级行情数据，通过分钟粒度价格变化模拟买卖盘不均衡量Q
使用基于nnls 优化改进的sklearn线性模型进行参数a1-a3的拟合， b1与a4使用改进的grid search，通过先验知识确定范围， 使用迭代方法进行参数空间的最优求解，结合适当剪纸优化
Source Root: mi_models/mi_models
模型入口：api/mi_model.py, 调用例子：
        db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                 "mincached": 0, "maxcached": 1}
        file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
        mi = MIModel(db_config)
        #模型训练，分别对60,90,120分钟进行组合计算因子，训练完成的拟合模型在目录*/data/models下，因子文件在目录 */data/features/ 下, MI 模型参数结果在file_name中
        mi.train(sec_code='000001', exchange='XSHG', start_date='20150103', end_date='20181018', file_name=file_name)
        ret = mi.load_model(file_name)
        #预测输出(临时冲击、永久冲击、市场冲击、瞬间冲击)
        ret = mi.predict(sec_code='000001', exchange='XSHG', quantity=1000, begin_time='20180719 10:00:00',
                              end_time='20180910 10:30:00', file_name=file_name)


The module apply the market impact model I-star from Kissell Research Group(http://www.kissellresearch.com/krg-i-star-market-impact-model) or JP Morgan.
We train the model with A-stock and index history  1-min market data, through which we simulate the market imbalance factor Q.
Based on the optimaization problem nnls, we modify the linear regression model in sklearn, and train the parameters a1-a3 with this updated model.
Moreover, we apply grid search to train b1 and a4 through some pre-knowledge for the range.

Entry for this module: api/mi_model.py. Sample calls are showed as followings:
            db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                 "mincached": 0, "maxcached": 1}
        file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
        mi = MIModel(db_config)
        #train the model, grouped by 60, 90,and 120 mins.Trained models are stored under */data/models, features under */data/features/, and trained params are kept in file_name
        mi.train(sec_code='000001', exchange='XSHG', start_date='20150103', end_date='20181018', file_name=file_name)
        ret = mi.load_model(file_name)
        #predict outputs tuple (temporary impact, permanent impact, total market impact, and instant impact )
        ret = mi.predict(sec_code='000001', exchange='XSHG', quantity=1000, begin_time='20180719 10:00:00',
                              end_time='20180910 10:30:00', file_name=file_name)
