import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("Thyroid_Diff.csv")

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

quantitative_data  = x.columns[x.dtypes == "int64"]
scaler = StandardScaler()
quantitative_preprocessed = pd.DataFrame(scaler.fit_transform(x[quantitative_data]), columns=quantitative_data)

qualitative_data = x.columns[x.dtypes == "object"].values
encoder = OneHotEncoder(sparse_output=False)
qualitative_preprocessed = pd.DataFrame(
    encoder.fit_transform(x[qualitative_data]), 
    columns=encoder.get_feature_names_out(qualitative_data)
)

x_preprocessed = pd.concat([quantitative_preprocessed, qualitative_preprocessed], axis=1)

y = y.map({"No": 0, "Yes": 1, })

x_train,x_test,y_train,y_test = train_test_split(x_preprocessed, y, test_size=0.33,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],eval_metric='logloss')

lgb.plot_importance(model)
plt.savefig('importance.png')
lgb.plot_metric(model)
plt.savefig('metric.png')

cm = metrics.confusion_matrix(y_test, model.predict(x_test))
# Display confusion matrix
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
disp.plot(cmap="Blues_r")
plt.savefig('confusion_matrix.png')

print(metrics.classification_report(y_test,model.predict(x_test)))


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))

