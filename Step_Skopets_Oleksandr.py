#!/usr/bin/env python
# coding: utf-8

# ### 1.Скачайте этот набор данных IKEA.

# * Импортируем все библиотеки необходимые для работы

# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


# In[37]:


dataset = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv', index_col=[0])
dataset


# In[38]:


dataset.item_id.value_counts()


# * Удаляем дубликаты, которые есть в базе

# In[39]:


df = dataset.drop_duplicates()
df


# #### Дубликатов в датасете нету

# ### 2. Выполните исследовательский анализ для набора данных, включая описательную статистику и визуализации. Опишите результаты.

# * Посмотрим разширеную информацию о нашем датасете

# In[40]:


df.info()


# In[41]:


dataset.sellable_online.value_counts()


# In[42]:


dataset.old_price.value_counts()


# #### В датасете 3694 уникальных товаров, пустые значения есть только в колонках с параметрами товаров

# * Удалим некоторые колонки, которые нам не помогут в дальнейшем анализе. А именно "item_id", "old_price", "sellable_online","link":
#     * item_id - мы будем использовать индекс;
#     * old_price - этот параметр у виде объекта и так же для многих товаров нету значения, даже переведя в числовое значение, мы не сможем использовать эти данные для всего датасету, данные отсуствуют в 3040 товарах из 3694;
#     * sellable_online - на мой взгляд, нету дополнительных характеристик, по которым мы можем понять на сколько этот показатель нам необходим;
#     * link - ссылка на товар.

# In[43]:


df = df.drop(['item_id','old_price','sellable_online','link'], axis=1)
df


# In[44]:


df.describe()


# * Построим гистограммы для некоторых параметров

# In[45]:


for col in ['price', 'depth', 'height', 'width']:
    print(col)
    plt.hist(df[col])
    plt.show()


# #### По гистограммам можно предположить, что эти величины имеют логнормальное распределение.

# * Построим гистограмму для категорий

# In[46]:


sns.countplot(x=df['category']).set_xticklabels(df['category'].unique(), rotation=90)


# In[47]:


plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='price', y='category')
plt.show()


# In[48]:


plt.subplots(figsize=(18,6))
sns.violinplot(data=df, y='price', x='category')
plt.xticks(rotation=90)
plt.show()


# #### Большенство категории имеют выбросы со стороны максимальных цен.

# * Посмотрим есть ли влияние цвета товару на цену

# In[49]:


plt.subplots(figsize=(8,6))
sns.violinplot(data=df, y='price', x='other_colors')
plt.show()


# In[50]:


sns.pairplot(df[['price','depth','height','width','other_colors','category']], hue='other_colors')


# #### Похоже, цены на товары, которые имеют несколько цветов и цены на одноцветные товары - имеют одинаковое распределение.

# In[53]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, center=0, annot=True)
plt.show()


# #### Прослеживается некоторая корелляция между ценой и габаритами изделия

# * Посмотрим подробную информацию о колонке "designer"

# In[54]:


df['designer'].value_counts()


# #### Для анализу данные в таком формате нам не подойдут

# ### 3. Основываясь на EDA и вашем здравом смысле, выберите две гипотезы, которые вы хотите проверить / проанализировать. Для каждой гипотезы перечислите нулевую гипотезу и другие возможные альтернативные гипотезы, разработайте тесты, чтобы различать их, и выполните их. Опишите результаты.

# ###  Гипотеза №1
# По гистограмме цен можно было предположить, что цена имеет логнормальное распределение. Если случайная величина имеет логнормальное распределение, то её логарифм имеет нормальное распределение. Построим гистограмму для логарифма цен и проверим гипотезу H_0 о том, что цены имеют логнормальное распределение.

# In[55]:


plt.hist(np.log10(df.price))


# In[59]:


from scipy import stats


# * Используем критерий D’Agostino’s K^2 Test для проверки гипотезы о нормальности распределения.

# In[62]:


h0_1_p_value = stats.normaltest(np.log10(df.price)).pvalue
print(f'p-value: {h0_1_p_value}')


# #### Так как значение p-value очень маленькое, то с большой долей уверенности критерий D’Agostino’s K^2 Test нашу нулевую гипотезу _отвергает_, хотя по гистограммам этого не скажешь. Возможно гистограммы имею очень мало корзин. Увеличим, количество корзин до 60

# In[63]:


plt.hist(np.log10(df.price), bins=60)


# #### Наблюдаем выбросы и справа и слева от купола, данная гипотеза о нормальном распределении цен не нашла подтверждение.

# ### Гипотеза №2
# Проверим гипотезу о том что цены на товары которые имеют несколько цветов статистически не отличаются от цен на одноцветные товаров.

# In[64]:


one_color = df[df['other_colors'] == 'No']['price'].apply(np.log10)
mult_color = df[df['other_colors'] == 'Yes']['price'].apply(np.log10)
one_color.name, mult_color.name = 'one color', 'multiplie colors'

one_color.hist(alpha=0.5, color='red', weights=[1./len(one_color)]*len(one_color), bins=20)
mult_color.hist(alpha=0.5, color='blue', weights=[1./len(mult_color)]*len(mult_color), bins=20)
plt.axvline(one_color.mean(), color='red', alpha=0.8, linestyle='dashed')
plt.axvline(mult_color.mean(), color='blue', alpha=0.8, linestyle='dashed')
plt.axvline(one_color.median(), color='red', alpha=0.8, linestyle='dotted')
plt.axvline(mult_color.median(), color='blue', alpha=0.8, linestyle='dotted')
plt.legend([one_color.name, mult_color.name])


# * Мы уже знаем что цена не имеет нормального распределения. Проверим критерий Манна-Уитни (он работает с не нормальным распределением и непарными выборками).

# In[66]:


stats.mannwhitneyu(one_color, mult_color).pvalue


#  * p-value - меньше 0.01, гипотеза _отвергнута_. Разница цен на многоцветные и одноцветные товары статистически значима. 
# Так как распределение цен не существенно отличается от нормального вычислим p-значение t-теста Стьюдента.

# In[67]:


stats.ttest_ind(one_color, mult_color, equal_var=False).pvalue


# #### Данный тест также подтверждает несостоятельность нашей гипотезы.

# ### 4. Обучите модель предсказывать цену на мебель.

# #### 4.1 Укажите, какие столбцы не следует включать в модель ипочему. 

# * При первичной обратотки мы удалили несколько колонок, а именно _"item_id", "old_price", "sellable_online","link"_.

# #### 4.2 Создайте конвейер перекрестной проверки для обучения и оценки модели, включая (при необходимости) такие шаги как,  вменение пропущенных значений и нормализация.

# На этапе EDA мы обнаружили, что некоторые поля имеют пустые значения или некоректно заполенные ("мусор").
# 
# _Габариты_
# 
# Габариты имеют пустые значения. Есть несколько стратегий заполнения пустых значений габаритов.
# 
# 1. Можно просто удалить все строки у которых имеются пустые значения.
# 2. Воспользоваться SimpleImputer библиотеки sklearn и заполнить пустые значения каким-то (например, медианным) значением. 
# 3. Придумать свой вариант (см. дальше)
# 
# _Дизайнеры_
# 
# Колонка `designer` содержит "мусор" (строки начинающиеся с цифр), который надо будет удалить и как-то заполнить пустые значения. При более детальном рассмотрении можно увидеть, что некоторые товары разрабатывала группа дизайнеров, например 
# ```
# Francis Cayouette/K Hagberg/M Hagberg
# K Hagberg/M Hagberg/Francis Cayouette
# ```
# и также можно заметить, что в данном случае строки разные, а состав дизайнеров один и тот же. Здравый смысл подсказывает, что необходимо привести все строки к одному виду, например, можно отсортировать имена дизайнеров по алфавиту.
# 
# Также очень много (около 30%) товаров имеет в списке дизайнеров строку `IKEA of Sweden`. Учитывая что все эти товары изготовлены IKEA, то, возможно, это значение также является мусором.
# 

# In[68]:


'''
Процедура приведения колонки designer к "нормальному виду"
value - строка, состав дизайнеров
removeIKEA - удалять или нет IKEA of Sweden
emptyValue - чем заменять неверные значения, по умолчанию np.nan, можно поробовать менять на пустую строку или "IKEA of Sweden"
'''
def cleanDesigners(value, removeIKEA=False, emptyValue=np.nan):
    #если это не строка возвращаем само значение
    if not isinstance(value, str):
        return value
    #если строка начинается на цифру, возвращаем пустое значение
    if len(value)>0 and value[0].isdigit():
        return emptyValue
    #разбиваем строку по / 
    designers = value.split("/")
    if removeIKEA:
        #пытаемся удалить "IKEA of Sweden"
        try:
            designers.remove("IKEA of Sweden")
        except:
            pass
    if len(designers) > 0:
        #возвращаем строку отсортированную по именам дизайнеров
        return '/'.join(sorted(designers))
    else:
        #или пустую строку если список пустой
        return emptyValue


# * Добавим новую колонку `designer_clean` - поле designer без мусора, с отсортированным списком дизайнеров. Пустые значения запролним "IKEA of Sweden"
# 
# * Если колонка `designer` содержала 381 уникальное значение, то `designer_clean` - 199
# 

# In[69]:


df["designer_clean"] = df["designer"].apply(cleanDesigners, args=(False, "IKEA of Sweden")) 
df["designer_clean"].value_counts()


# * Для заполнения отсутствующих габаритов воспользуемся возможностями SimpleImputer. Так же нам необходимо перевести категорийные переменные в цифровой вид. Так как категории товаров это неупорядоченные данные, то воспользуемся OneHotEncoder и построим соответствующий Pipeline
# 
# * Так как мы хотим прогнозировать цену - вещественное число, то воспользуемся каким-нибудь из регрессоров, например деревом решений DecisionTreeRegressor()

# In[70]:


import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# df1 = df
X = df[['depth','width','height','category','designer_clean','other_colors']]
#X = df1[['depth','width','height','category']]
Y = df['price']
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

numeric_transf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('impute', SimpleImputer(strategy='median'))
])

categorical_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

col_prepr = ColumnTransformer(transformers=[
    ('numeric', numeric_transf, ['depth','width','height']),
    ('categorical', categorical_transf, ['category','designer_clean','other_colors'])
])

dtr = Pipeline(steps=[
    ('col_prep', col_prepr),
    # ('dtr', RandomForestRegressor(max_depth = 100))
    ('dtr', DecisionTreeRegressor(max_depth = 10, random_state=42))
])

dtr.fit(X_train, Y_train)
dtr_predict = dtr.predict(X_test)
print('R^2 : {:.5f}'.format(dtr.score(X_test, Y_test)))
print('MAE : {:.5f}'.format(sk.metrics.mean_absolute_error(dtr_predict, Y_test)))
print('MSE : {:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(dtr_predict, Y_test))))


# Наша первая модель показала следующий результат:
# ```
# R^2 : 0.74147
# MAE : 396.26201
# MSE : 705.91036
# ```
# 
# Для улучшения прогноза можно предпринять следующие шаги:
# 
# 1. Подобрать другую стратегию заполнения отсутствующих габаритов.
# 2. Использовать другой Encoder для категорий и дизайнеров.
# 3. Подобрать другой регрессор.
# 4. Подобрать наиболее оптимальные параметры регрессора.
# 
# Рассмотрим эти шаги.
# 
# 1. В первоначальной модели мы заполняли отсутствующие габариты медианным значением по всей выборке. Но это не совсем корректно, так как мы имеем 17 различных категорий, от фурнитуры до шкафов, которые существенно отличаются по размерам. По всей видимости более корректным будет заполнить отсутствующие значения габаритов средним значением по категории.
# 
# 2. Возможно есть смысл закодировать категории и дизайнеров средней (медиана) ценой по категории.
# 
# Добавим в датафрейм новые поля:

# In[72]:


df["other_colors_1"] = df["other_colors"].map(dict(Yes=1, No=0))

#рассчитаем среднее значение каждого габарита в разрезе категорий
median_d = df.groupby(['category'])['depth'].median()
median_h = df.groupby(['category'])['height'].median()
median_w = df.groupby(['category'])['width'].median()

median_price = df.groupby(['category'])['price'].median()
median_dsgn = df.groupby(['designer_clean'])['price'].median()

df = df.set_index(['category'])
df['depth_1'] = df['depth'].fillna(median_d)
df['height_1'] = df['height'].fillna(median_h)
df['width_1'] = df['width'].fillna(median_w)
df['category_median_price'] = median_price
df = df.reset_index()

df = df.set_index(['designer_clean'])
df['designer_median_price'] = median_dsgn
df = df.reset_index()

df.head()


#  * Для упрощения подбора параметров выборок и модели создадим процедуру, которая будет принимать на вход выборки и расчитывать оценки для разных регрессоров

# In[73]:


'''
'''
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def getBestRegressor(X, Y):
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    models = [
        sk.linear_model.LinearRegression(),
        sk.linear_model.LassoCV(),
        sk.linear_model.RidgeCV(),
        sk.svm.SVR(kernel='linear'),
        sk.neighbors.KNeighborsRegressor(n_neighbors=16),
        sk.tree.DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor()
    ]

    TestModels = pd.DataFrame()
    res = {}
    tmp = {}
    #для каждой модели из списка
    for model in models:
        #получаем имя модели
        m = str(model)
        tmp['Model'] = m[:m.index('(')]    
        #обучаем модель
        model.fit(X_train, Y_train) 
        #вычисляем R^2 - коэффициент детерминации
        tmp['R^2'] = '{:.5f}'.format(model.score(X_test, Y_test))
        #вычисляем MAE - средний модуль отклонения 
        tmp['MAE'] = '{:.5f}'.format(sk.metrics.mean_absolute_error(model.predict(X_test), Y_test))
        #вычисляем RMSE - корень из среднего квадрата отклонения
        tmp['RMSE'] = '{:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(model.predict(X_test), Y_test)))

        #записываем данные и итоговый DataFrame
        TestModels = TestModels.append([tmp])
    #делаем индекс по названию модели
    TestModels.set_index('Model', inplace=True)
    res['model'] = TestModels
    res['X_train'] = X_train
    res['Y_train'] = Y_train
    res['X_test'] = X_test
    res['Y_test'] = Y_test
    return res


# #### Тест №1.
# 
# Возьмем габариты расчитанные по медиане в разрезе категорий, категории и дизайнеры закодированны медианой цены.
# Проверим какой регрессор покажет лучший результат и как изменится оценка для модели на основе дерева решений.

# In[75]:


X1 = df[['width_1','depth_1','height_1','category_median_price','designer_median_price','other_colors_1']]
Y1 = df['price']

test1 = getBestRegressor(X1, Y1)
test1['model'].sort_values(by='R^2', ascending=False)


# #### Наилучший результат показал RandomForestRegressor. `R^2 = 0.83172`
# 
# #### Дерево решений для данных выборок так же показывает лучший результат `R^2 = 0.78065`

# #### Тест №2.
# 
# Удалим все строки с незаполненными габаритами, категории и дизайнеры закодированны медианой цены.

# In[76]:


df2 = df.dropna(subset=['width','height','depth'])
X2 = df2[['width','depth','height','category_median_price','designer_median_price','other_colors_1']]
Y2 = df2['price']

test2 = getBestRegressor(X2, Y2)
test2['model'].sort_values(by='R^2', ascending=False)


# In[77]:


Y1.shape


# Оценки `R^2` моделей улучшились. RandomForestRegressor - по прежнему показывает наилучший результат. 
# 
# Стоит обратить внимание на то, что хотя оценка `R^2` улучшилась, но при этом среднее абсолютное отклонение `MAE` стало больше. Скорее всего это результат того, что в первом случае мы заполнили большое количество пустых значения средним по категории. Если в первом тесте у нас было 3694 значения, то во втором тесте, после удаления пустых габаритов - всего 1899 значений. Т.е. практически 48.5% строк так или иначе заполнены средним значением.

# #### Тест №3.
# 
# Попробуем удалить "IKEA of Sweden" из списка дизайнеров в случае если список дизайнеров содержит другие фамилии. Например, у нас есть
# ```
# designer                              count
# Ehlén Johansson                         161
# Ehlén Johansson/IKEA of Sweden          145
# ```
# после преобразования будем иметь
# ```
# designer                              count
# Ehlén Johansson                         306
# ```

# In[78]:


df["designer_clean_2"] = df["designer"].apply(cleanDesigners, args=(True, "IKEA of Sweden"))

median_dsgn2 = df.groupby(['designer_clean_2'])['price'].median()

df = df.set_index(['designer_clean_2'])
df['designer_median_price2'] = median_dsgn2
df = df.reset_index()

df["designer_clean_2"].value_counts()


# In[79]:


df3 = df.dropna(subset=['width','depth','height'])
X3 = df3[['width','depth','height','category_median_price','designer_median_price2','other_colors_1']]
Y3 = df3['price']

test3 = getBestRegressor(X3, Y3)
test3['model'].sort_values(by='R^2', ascending=False)


# Результат практически не изменился. 
# 
# Наилучшие результаты показал RandomForestRegressor на выборках из _Тест №3_. 
# 
# Воспользуемся функцией GridSearchCV. Проведем кросс-валидацию и подберем оптимальные параметры для RandomForestRegressor

# In[81]:


from sklearn.model_selection import GridSearchCV, cross_val_score

X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42)
forest_grid = GridSearchCV(RandomForestRegressor(), {'n_estimators': [100,500], 'max_depth': [10,100,None], 'max_features': ['auto','sqrt','log2']}, cv=5, n_jobs=-1, verbose=3)
forest_grid.fit(X_train, Y_train)

print('Best Estimator :',forest_grid.best_estimator_)
print('Best Score     :',forest_grid.best_score_)
print('')
print('R^2            : {:.5f}'.format(sk.metrics.r2_score(Y_test, forest_grid.predict(X_test))))
print('MAE            : {:.5f}'.format(sk.metrics.mean_absolute_error(forest_grid.predict(X_test), Y_test)))
print('RMSE           : {:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(forest_grid.predict(X_test), Y_test))))
print('')
print('Feature importance:')
print('--------------------------------')
for feat, importance in zip(X_train.columns, forest_grid.best_estimator_.feature_importances_):
    print('{:.5f}    {f}'.format(importance, f=feat))

sns.set_style('whitegrid')
sns.barplot(y=X_train.columns, x=forest_grid.best_estimator_.feature_importances_)


# In[82]:


sns.regplot(x=Y_test, y=forest_grid.predict(X_test))


# #### 4.3 Предложите методы повышения производительности модели. Опишите результаты.

# Для улучшения качества модели необходимо улучшать качество данных.
# 
# 1. Можно попробовать извлечь недостающие габариты из описания товара. 
# 
# 2. Можно попробывать извлечь недостающие данные из старой цены товара.
# 
# 3. Можно попробывать извлечь количество товару продано онлайн. И найти влияние онлайн продаж на общие продажи товару.
# 
# 4. Можно попробывать поработать с "дизайнерами" товару и посмотреть их влияние на цену товару.
# 
# Как видно из таблицы и графика важности переменных (feature importance) "дизайнеры" вносят довольно большой вклад. Можно поэкспериментировать с кодированием дизайнеров. В случае коллективной работы, надо разделить группу дизайнеров на отдельные имена и затем применить подход используемый в OneHotEncoder. Т.е. если у нас есть такие строки: три дизайнера `Carina Bengs`, `Ebba Strandmark` и `IKEA of Sweden`, но в разных составах
# ```
# 1.   Carina Bengs/Ebba Strandmark
# 2.   Ebba Strandmark
# 3.   Ebba Strandmark/IKEA of Sweden
# ```
# три дизайнера - `Carina Bengs`, `Ebba Strandmark` и `IKEA of Sweden`, но в разных составах, то после применения энкодера мы должны получить
# ```
#      CB   ES   IK
# 1.    1    1    0
# 2.    0    1    0   
# 3.    0    1    1
# ```
# 

# #### _В данном проекте мы оценили и изучили предоставленный набор данных, проверили гипотезы о распределениях параметров. Построили модель предсказания цен на мебель. В процессе обучения модели протестировали несколько стратегий заполнения недостающих данных, подобрали наилучший регрессор и его оптимальные параметры._

# In[ ]:




