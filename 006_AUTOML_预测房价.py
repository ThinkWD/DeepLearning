import pandas
from autogluon.tabular import TabularDataset, TabularPredictor

# 训练 (最佳成绩: 0.11851)
train_data = TabularDataset('./dataset/HousePricesAdvanced/train.csv')
predictor = TabularPredictor(label='SalePrice')
predictor.fit(train_data.drop(columns=['Id']), presets='best_quality', time_limit=900)

# 预测
test_data = TabularDataset('./dataset/HousePricesAdvanced/test.csv')
preds = predictor.predict(test_data.drop(columns=['Id']))
test_data['SalePrice'] = pandas.Series(preds)
submission = pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)
