### 数据格式

- 特征值-标签
- 回归问题

<pre>
40920	8.326976	0.953952	largeDoses
14488	7.153469	1.673904	smallDoses
26052	1.441871	0.805124	didntLike
</pre>
>x [n, 3]

>y [n,]

#### 算法
<pre>
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = np.array([[22587.0]])  # Cyprus' GDP per capita
print(model.predict(X_new)) 
# outputs [[ 5.76666667]]
# =======================用同一集训练和验证，差==============================
</pre>
- 还有数据格式(20640, 16)，label（20640，）的

### 评估
<pre>
#使用mean_squared_error,测量整个训练集上的RMSE
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#使用mean_absolute_error,测量整个训练集上的MAE
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
</pre>
### 评估 (二)
<pre>
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_score(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_score(tree_rmse_scores)
</pre>


### 数据格式

- 二进制图像
- 分类问题

>eg:手写数字识别
<pre>
00000000000000111000000000000000
00000000000001111000000000000000
00000000000111111110000000000000
00000000000111111111000000000000
00000000011111111111100000000000
00000000011111101111100000000000
00000000111111000011110000000000
00000000111111000011111000000000
00000001111110000000111100000000
00000001111100000000111100000000
00000011111100000000111100000000
00000011111100000000111100000000
00000011110000000000111100000000
00000001111000000000011110000000
00000001111000000000001111000000
00000001111100000000001111000000
00000011111100000000001111000000
00000011111000000000001111000000
00000011111000000000001111000000
00000011111000000000001110000000
00000001111100000000001111000000
00000000111100000000001111000000
00000000111100000000001111000000
00000000111100000000011111000000
00000000111100000000111110000000
00000000111100000001111100000000
00000000011100000011111100000000
00000000001111100111111100000000
00000000000111111111111000000000
00000000000111111111100000000000
00000000000111111111000000000000
00000000000011111100000000000000
</pre>
> x [n, 32*32]
> y [n,]

#### 算法
<pre>
#knn
from sklearn.neighbors import KNeighborsClassifier
#begin time
start = time.clock()

#progressing
knn_clf=KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)

print( score.mean() )
#end time
elapsed = (time.clock() - start)
print("Time used:",int(elapsed), "s")
#k=3
#0.942300738697
#0.946100822903 weights='distance'
#0.950799888775 p=3
#k=5
#0.939899237556
#0.94259888029
#k=7
#0.935395994386 
#0.938997377902
#k=9
#0.933897851978
</pre>


<pre>#最后用全量数据训练，提交kaggle。代码模版
clf=knn_clf

start = time.clock()
clf.fit(X_train,y_train)
elapsed = (time.clock() - start)
print("Training Time used:",int(elapsed/60) , "min")

result=clf.predict(X_test)
result = np.c_[range(1,len(result)+1), result.astype(int)]
df_result = pd.DataFrame(result, columns=['ImageId', 'Label'])

df_result.to_csv('./results.knn.csv', index=False)
#end time
elapsed = (time.clock() - start)
print("Test Time used:",int(elapsed/60) , "min")
</pre>