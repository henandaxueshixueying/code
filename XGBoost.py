import os

import pandas as pd
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib
# import shap
import joblib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.font_manager.findSystemFonts()
import os
os.chdir(r'C:\Users\86151\Desktop\ed\202402\-Nevergiveup-Shen')
plt.rcParams.update({'font.size': 27})


data = pd.read_csv(r'C:\Users\86151\Desktop\ed\202402\-Nevergiveup-Shen\data_3000.csv', encoding='gbk')
data.columns = data.columns.str.replace('in:', '').str.replace('out:', '')

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-3], data.iloc[:, -3:], random_state=256)
# 特征变量分布
for col in data.columns[-3:]:
    # 去掉网格线
    plt.figure(figsize=(6, 5))
    plt.gca().grid(False)
    tips = sns.load_dataset("tips")
    plt.gca().yaxis.set_major_formatter('{x:.0%}')
    n, bins, patches = plt.hist(data[col], weights=np.ones_like(data[col]) / len(data[col]),
                                label=col, color='mediumblue', rwidth=0.98)
    plt.gca().yaxis.set_major_formatter('{x:.0%}')
    plt.axvline(data[col].mean(), color='r', linestyle='--', label='Average')
    plt.text(data[col].mean(), max(n), round(data[col].mean(), 2), horizontalalignment='center')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Relative frequency')
    plt.savefig(f'{col}_dist.plot.png')
    plt.show()


# 皮尔逊相关系数
# 定义起始颜色和结束颜色
from matplotlib.colors import LinearSegmentedColormap
cm_colors = np.array([[198, 91, 63], [222, 160, 145], [237, 204, 197], [215, 230, 234], [119, 165, 179], [64, 128, 148]]) / 255

# 创建颜色映射
cmap_name = 'custom_colormap'
cm = LinearSegmentedColormap.from_list(cmap_name, cm_colors)

plt.figure(figsize=(10, 8))
sns.heatmap(data.iloc[:, :-3].corr(), annot=True, cmap=cm, linewidths=.5, fmt='.2f')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

model = MultiOutputRegressor(xgboost.XGBRegressor())

# 定义参数网格
param_grid = {
    "estimator__n_estimators": [50, 100, 150],
    "estimator__max_depth": [3, 4, 5],
    "estimator__learning_rate": [0.01, 0.1, 1],
}

# 定义网格搜索对象
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    n_jobs=-1,
)

# 训练模型
grid_search.fit(x_train, y_train)

# 定义模型
model = MultiOutputRegressor(xgboost.XGBRegressor(**grid_search.best_params_))

# 训练模型
model.fit(x_train, y_train, eval_set=[(x_train, y_train.iloc[:, 0])])

# 预测测试集
y_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

# MAE,MSE,MAPE,R2
train_eval = [mean_absolute_error(np.array(y_train).reshape(-1), y_train_pred.reshape(-1)),
              mean_squared_error(np.array(y_train).reshape(-1), y_train_pred.reshape(-1)),
              mean_absolute_percentage_error(np.array(y_train).reshape(-1), y_train_pred.reshape(-1)),
              r2_score(np.array(y_train).reshape(-1), y_train_pred.reshape(-1))]
test_eval = [mean_absolute_error(np.array(y_test).reshape(-1), y_pred.reshape(-1)),
             mean_squared_error(np.array(y_test).reshape(-1), y_pred.reshape(-1)),
             mean_absolute_percentage_error(np.array(y_test).reshape(-1), y_pred.reshape(-1)),
             r2_score(np.array(y_test).reshape(-1), y_pred.reshape(-1))]
model_eval = pd.DataFrame({'train_eval': train_eval, 'test_eval': test_eval}, index=['MAE', 'MSE', 'MAPE', 'R2'])
model_eval.to_excel(r'model_eval.xlsx')
# 学习率曲线
xgb = xgboost.XGBRegressor(**grid_search.best_params_)
xgb.fit(x_train, y_train.iloc[:, 0], eval_set=[(x_train, y_train.iloc[:, 0]), (x_test, y_test.iloc[:, 0])])

plt.figure(figsize=(8,7))
plt.plot(xgb.evals_result()['validation_0']['rmse'], label='train')
plt.plot(xgb.evals_result()['validation_1']['rmse'], label='test')
plt.xlabel('iterations')
plt.ylabel('rmse loss')
plt.legend()

plt.savefig('learning curve.png', dpi=300)
plt.show()

# 预测与实际效果图
for i in range(3):
    plt.figure(figsize=(8,7))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], s=8, label='Data', facecolors='none', edgecolors='lightcoral')
    x = np.linspace(y_test.iloc[:, i].min(), y_test.iloc[:, i].max(), 100)
    y = x
    plt.plot(x, y, label='Fit', color='r')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title('Validation:R={}'.format(round(r2_score(y_test.iloc[:, i], y_pred[:, i]), 5)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{y_test.columns[i]}_预测效果.png')
    plt.show()

import json
# 最优参数保存
with open('best_params.txt', 'w') as f:
    f.write(json.dumps(grid_search.best_params_))
#
# shap.initjs()
# # 计算每个子模型的SHAP值并绘制总结图
# for (i, estimator), title in zip(enumerate(model.estimators_), y_test.columns):
#     explainer = shap.Explainer(estimator)
#     shap_values = explainer(x_test)
#     shap.summary_plot(shap_values, x_test, show=False)
#     plt.title(title, loc='center')
#     plt.tight_layout()
#     plt.savefig(f'{title} summary_plot.png')
#     plt.show()
#
#     # plt.savefig(f'C:\\Users\\Zz\\Desktop\\project\\4.24 xgb多任务\\model{i}_sumary.png')
#     # plt.clf()
#
#     feature_names = x_test.columns
#     feature_importances = np.abs(shap_values.values).mean(axis=0)
#     top_features = feature_names[np.argsort(feature_importances)][-2:]
#     # print(top_features[-1],top_features[0])
#
#     # 绘制dependence_plot
#     shap.dependence_plot(top_features[-1], shap_values.values, x_test, interaction_index=top_features[0], show=False)
#     plt.title(title, loc='center')
#     plt.savefig(f'{title} dependence_plot.png')
#     plt.show()
    # plt.savefig(f'C:\\Users\\Zz\\Desktop\\project\\4.24 xgb多任务\\model{i}_dependence.png')
    # plt.clf()
    # 绘制force_plot
    # shap.force_plot(explainer.expected_value, shap_values.values[0,:], x_test.iloc[0,:])

    # 绘制decision_plot
    # shap.decision_plot(explainer.expected_value, shap_values.values[0,:], x_test.iloc[0,:], show=True)
    # plt.savefig(f'C:\\Users\\Zz\\Desktop\\project\\4.24 xgb多任务\\model{i}_decision.png')
    # plt.clf()



# joblib.dump(model, r'xgb_model.pkl')
