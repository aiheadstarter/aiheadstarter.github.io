---
layout: single
title:  "회귀방정식 실습예제 입니다."
---

```python

# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import statsmodels.formula.api as smf

# PyDataset을 사용하여 Prestige 데이터를 로드합니다.
from pydataset import data
prestige = data('Prestige')

# formula를 사용하여 선형 회귀 모델을 정의하고 피팅합니다.
model = smf.ols('income ~ education', data=prestige).fit()

# 결과 요약을 출력합니다.
print(model.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 income   R-squared:                       0.334
    Model:                            OLS   Adj. R-squared:                  0.327
    Method:                 Least Squares   F-statistic:                     50.06
    Date:                Thu, 15 Feb 2024   Prob (F-statistic):           2.08e-10
    Time:                        18:41:07   Log-Likelihood:                -975.61
    No. Observations:                 102   AIC:                             1955.
    Df Residuals:                     100   BIC:                             1960.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept  -2853.5856   1407.039     -2.028      0.045   -5645.111     -62.060
    education    898.8128    127.035      7.075      0.000     646.778    1150.847
    ==============================================================================
    Omnibus:                       61.134   Durbin-Watson:                   1.546
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              298.196
    Skew:                           1.950   Prob(JB):                     1.77e-65
    Kurtosis:                      10.413   Cond. No.                         45.5
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python

```


```python
import pandas as pd
my_series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```


```python
my_series
```




    a    1
    b    2
    c    3
    d    4
    dtype: int64




```python
import numpy as np
my_array = np.array([1, 2, 3, 4])
```


```python
my_array
```




    array([1, 2, 3, 4])




```python
import numpy as np
one_dimensional_array = np.array([1, 2, 3, 4, 5])
```


```python
one_dimensional_array
```




    array([1, 2, 3, 4, 5])




```python
two_dimensional_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```


```python
two_dimensional_array
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
three_dimensional_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
```


```python
three_dimensional_array
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
import numpy as np
two_dimensional_array = np.array([[1, 2, 3, 4, 5]])
```


```python
two_dimensional_array
```




    array([[1, 2, 3, 4, 5]])




```python
import numpy as np
one_dimensional_array = np.array([1, 2, 3, 4, 5])

# 1차원 배열을 5행 1열의 2차원 배열로 변환
two_dimensional_array = one_dimensional_array.reshape(5, 1)

two_dimensional_array

```




    array([[1],
           [2],
           [3],
           [4],
           [5]])




```python
import pandas as pd
import statsmodels.api as sm
from pydataset import data

# PyDataset에서 Prestige 데이터셋 로드
prestige = data('Prestige')

# education의 평균보다 큰 데이터만 필터링
subset_prestige = prestige[prestige['education'] > prestige['education'].mean()]

# 독립 변수에 상수항 추가
X = subset_prestige[['education']]
X = sm.add_constant(X)

# 종속 변수 선택
y = subset_prestige['income']

# 선형 회귀 모델 적합
model = sm.OLS(y, X).fit()

# 결과 요약 출력
print(model.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 income   R-squared:                       0.246
    Model:                            OLS   Adj. R-squared:                  0.230
    Method:                 Least Squares   F-statistic:                     15.30
    Date:                Thu, 15 Feb 2024   Prob (F-statistic):           0.000293
    Time:                        20:10:23   Log-Likelihood:                -479.86
    No. Observations:                  49   AIC:                             963.7
    Df Residuals:                      47   BIC:                             967.5
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       -1.03e+04   4921.464     -2.093      0.042   -2.02e+04    -398.175
    education   1455.0876    371.947      3.912      0.000     706.826    2203.349
    ==============================================================================
    Omnibus:                       35.840   Durbin-Watson:                   1.602
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               94.977
    Skew:                           2.036   Prob(JB):                     2.38e-21
    Kurtosis:                       8.472   Cond. No.                         104.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
import statsmodels.api as sm

# statsmodels에서 내장 데이터 세트 불러오기
data = sm.datasets.get_rdataset("mtcars", "datasets").data

# 데이터의 처음 몇 줄을 출력
print(data.head())

```

                        mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  \
    rownames                                                                     
    Mazda RX4          21.0    6  160.0  110  3.90  2.620  16.46   0   1     4   
    Mazda RX4 Wag      21.0    6  160.0  110  3.90  2.875  17.02   0   1     4   
    Datsun 710         22.8    4  108.0   93  3.85  2.320  18.61   1   1     4   
    Hornet 4 Drive     21.4    6  258.0  110  3.08  3.215  19.44   1   0     3   
    Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3   
    
                       carb  
    rownames                 
    Mazda RX4             4  
    Mazda RX4 Wag         4  
    Datsun 710            1  
    Hornet 4 Drive        1  
    Hornet Sportabout     2  



```python
import statsmodels.api as sm
import pandas as pd

# statsmodels를 사용하여 mtcars 데이터 불러오기
mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data

# pandas를 사용하여 데이터를 CSV 파일로 저장
mtcars.to_csv('mtcars.csv', index=False)  # 'index=False'는 행 이름(인덱스)를 저장하지 않음을 의미
```


```python
# 특정 컬럼만 선택하여 새로운 데이터프레임 생성
selected_columns = ["mpg", "hp", "wt", "disp", "drat"]
new_mtcars_data = data[selected_columns]

# 새로운 데이터프레임 확인
new_mtcars_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>hp</th>
      <th>wt</th>
      <th>disp</th>
      <th>drat</th>
    </tr>
    <tr>
      <th>rownames</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mazda RX4</th>
      <td>21.0</td>
      <td>110</td>
      <td>2.620</td>
      <td>160.0</td>
      <td>3.90</td>
    </tr>
    <tr>
      <th>Mazda RX4 Wag</th>
      <td>21.0</td>
      <td>110</td>
      <td>2.875</td>
      <td>160.0</td>
      <td>3.90</td>
    </tr>
    <tr>
      <th>Datsun 710</th>
      <td>22.8</td>
      <td>93</td>
      <td>2.320</td>
      <td>108.0</td>
      <td>3.85</td>
    </tr>
    <tr>
      <th>Hornet 4 Drive</th>
      <td>21.4</td>
      <td>110</td>
      <td>3.215</td>
      <td>258.0</td>
      <td>3.08</td>
    </tr>
    <tr>
      <th>Hornet Sportabout</th>
      <td>18.7</td>
      <td>175</td>
      <td>3.440</td>
      <td>360.0</td>
      <td>3.15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 새로운 데이터프레임의 통계 요약
new_mtcars_summary = new_mtcars_data.describe()
new_mtcars_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>hp</th>
      <th>wt</th>
      <th>disp</th>
      <th>drat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20.090625</td>
      <td>146.687500</td>
      <td>3.217250</td>
      <td>230.721875</td>
      <td>3.596563</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.026948</td>
      <td>68.562868</td>
      <td>0.978457</td>
      <td>123.938694</td>
      <td>0.534679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.400000</td>
      <td>52.000000</td>
      <td>1.513000</td>
      <td>71.100000</td>
      <td>2.760000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.425000</td>
      <td>96.500000</td>
      <td>2.581250</td>
      <td>120.825000</td>
      <td>3.080000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19.200000</td>
      <td>123.000000</td>
      <td>3.325000</td>
      <td>196.300000</td>
      <td>3.695000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.800000</td>
      <td>180.000000</td>
      <td>3.610000</td>
      <td>326.000000</td>
      <td>3.920000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>335.000000</td>
      <td>5.424000</td>
      <td>472.000000</td>
      <td>4.930000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 새로운 데이터프레임의 분포 시각화
plt.figure(figsize=(15, 10))

# 각 변수에 대한 히스토그램 생성
for i, column in enumerate(new_mtcars_data.columns, start=1):
    plt.subplot(2, 3, i)  # 2행 3열의 서브플롯 구성
    sns.histplot(new_mtcars_data[column], kde=True, bins=15)  # 커널 밀도 추정(KDE) 포함
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 새로운 데이터프레임을 이용한 산점도 행렬 생성
sns.pairplot(new_mtcars_data, kind='reg', plot_kws={'line_kws':{'color':'salmon'}, 'scatter_kws':{'color':'royalblue', 's':60}})
plt.suptitle("Car Performance", y=1.02)  # 제목 설정과 위치 조정
plt.show()  # 플롯 표시

```


    
![png](output_21_0.png)
    



```python
import statsmodels.api as sm

# new_mtcars_data에서 'mpg'를 종속 변수로, 나머지를 독립 변수로 설정
X = new_mtcars_data.drop('mpg', axis=1)  # 독립 변수
y = new_mtcars_data['mpg']  # 종속 변수

# OLS 회귀 모델을 위한 상수항 추가
X = sm.add_constant(X)

# 선형 회귀 모델 생성 및 적합
new_mtcars_lm = sm.OLS(y, X).fit()

# 모델 요약 결과 출력
summary = new_mtcars_lm.summary()
summary

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.838</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.814</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   34.82</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Feb 2024</td> <th>  Prob (F-statistic):</th> <td>2.70e-10</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:54:35</td>     <th>  Log-Likelihood:    </th> <td> -73.292</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    32</td>      <th>  AIC:               </th> <td>   156.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    27</td>      <th>  BIC:               </th> <td>   163.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   29.1487</td> <td>    6.294</td> <td>    4.631</td> <td> 0.000</td> <td>   16.235</td> <td>   42.062</td>
</tr>
<tr>
  <th>hp</th>    <td>   -0.0348</td> <td>    0.012</td> <td>   -2.999</td> <td> 0.006</td> <td>   -0.059</td> <td>   -0.011</td>
</tr>
<tr>
  <th>wt</th>    <td>   -3.4797</td> <td>    1.078</td> <td>   -3.227</td> <td> 0.003</td> <td>   -5.692</td> <td>   -1.267</td>
</tr>
<tr>
  <th>disp</th>  <td>    0.0038</td> <td>    0.011</td> <td>    0.353</td> <td> 0.727</td> <td>   -0.018</td> <td>    0.026</td>
</tr>
<tr>
  <th>drat</th>  <td>    1.7680</td> <td>    1.320</td> <td>    1.340</td> <td> 0.192</td> <td>   -0.940</td> <td>    4.476</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.267</td> <th>  Durbin-Watson:     </th> <td>   1.736</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.072</td> <th>  Jarque-Bera (JB):  </th> <td>   4.327</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.899</td> <th>  Prob(JB):          </th> <td>   0.115</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.102</td> <th>  Cond. No.          </th> <td>4.26e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.26e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python

```
