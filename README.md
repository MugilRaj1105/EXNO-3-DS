## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
     import pandas as pd 
     df= pd.read_csv("/content/Encoding Data.csv")
     df
```

<img width="377" height="438" alt="image" src="https://github.com/user-attachments/assets/06d10789-1f2b-4b5b-8c73-246f74852a18" />

  
```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])
```

<img width="167" height="228" alt="image" src="https://github.com/user-attachments/assets/73d7a3bd-905c-4137-91e0-7851069ee937" />


```
df['bo2']= e1.fit_transform(df[["ord_2"]])
df
```
<img width="393" height="448" alt="image" src="https://github.com/user-attachments/assets/23d44dd8-2467-40a5-a466-65a08c2dd1d8" />


```
le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc
```

<img width="397" height="441" alt="image" src="https://github.com/user-attachments/assets/c07af8c2-ae07-419c-af77-a0b3bbd0795d" />

```

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2
 ```
<img width="520" height="452" alt="image" src="https://github.com/user-attachments/assets/ae55b89f-fc2a-4c03-874f-83284b468812" />


```

pd.get_dummies(df2,columns=["nom_0"])


```
<img width="798" height="451" alt="image" src="https://github.com/user-attachments/assets/77fc4da5-7eea-4488-8af5-fb9dbb1b70af" />


```
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df
```
<img width="608" height="461" alt="image" src="https://github.com/user-attachments/assets/a0891b9d-b9fd-4f69-8241-f65e67f62e1a" />

```
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df
```
<img width="838" height="443" alt="image" src="https://github.com/user-attachments/assets/95957e02-7c55-44fa-b19d-cf017caad707" />


```
dfb= pd.concat([df,nd],axis=1)
dfb
```
<img width="838" height="443" alt="image" src="https://github.com/user-attachments/assets/95957e02-7c55-44fa-b19d-cf017caad707" />


```
from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC
```
<img width="710" height="461" alt="image" src="https://github.com/user-attachments/assets/d5b6cafc-7fa3-4783-8210-53f7d350d64e" />

```
import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1031" height="501" alt="image" src="https://github.com/user-attachments/assets/8b182307-78b2-42f5-9d13-89e06b15e981" />

```
df.skew()
```
<img width="370" height="246" alt="image" src="https://github.com/user-attachments/assets/0ab78fee-2c00-4add-ba96-6da17ba358fe" />


```
np.log(df["Highly Positive Skew"])
```
<img width="327" height="556" alt="image" src="https://github.com/user-attachments/assets/3211ee84-2234-469f-b66d-01b189dbe015" />


```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="342" height="568" alt="image" src="https://github.com/user-attachments/assets/db3ed89a-0a56-4fc9-a8fb-5c3658b61a03" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="282" height="553" alt="image" src="https://github.com/user-attachments/assets/4f3b6c7e-807b-40c7-afe5-7b8fd6d64444" />

```
np.square(df["Highly Positive Skew"])
```
<img width="330" height="547" alt="image" src="https://github.com/user-attachments/assets/f3b65ab9-1271-4473-b3c5-45a540cd02d2" />

```
df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1322" height="522" alt="image" src="https://github.com/user-attachments/assets/71da052f-08fc-431c-bf56-13fef4e604d8" />

```
df.skew()
```
<img width="391" height="288" alt="image" src="https://github.com/user-attachments/assets/81972c36-5b7e-4e16-a448-d336bc097721" />


```
df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="492" height="313" alt="image" src="https://github.com/user-attachments/assets/1f0d978e-cb3d-49f7-be70-ce0a8af2d0fc" />


```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])
 df
```
<img width="1743" height="531" alt="image" src="https://github.com/user-attachments/assets/1966dd1a-47c6-47b7-b096-bcdd33cce0ec" />


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
<img width="832" height="590" alt="image" src="https://github.com/user-attachments/assets/55cf90e3-1422-4fad-8a1b-f509444698b2" />

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
<img width="574" height="432" alt="image" src="https://github.com/user-attachments/assets/e39c3a88-fdd2-4571-849e-9d1b706e8cea" />

```

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```
<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/ffb3a371-a340-42b5-827a-0870f9cdba4f" />


```

 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()

```
<img width="739" height="550" alt="image" src="https://github.com/user-attachments/assets/67ef64c8-cc9c-45df-a613-b98db7aa6e55" />

```
dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

```
<img width="708" height="523" alt="image" src="https://github.com/user-attachments/assets/4ec9de4e-10f4-4131-a759-d89afd8179c1" />

```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```
<img width="706" height="527" alt="image" src="https://github.com/user-attachments/assets/eb0c56de-d408-4137-911f-f9a6ba9b8fc6" />

# RESULT:
           Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully


       
