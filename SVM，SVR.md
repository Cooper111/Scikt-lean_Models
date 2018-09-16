## 回归问题
### 数据格式
- 输入数据（20640,16）
 
全为数值，输入由转换流水线得，可见Boston_House系列代码

- 输入标签
> housing = pd.read_csv("housing.csv")
>housing_labels = housing["median_house_value"].copy()

#### 算法
<pre>
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)

</pre>

### 评估 (一)
<pre>
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
#111094.6308539982

</pre>

### 评估 (二)
<pre>
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
</pre>