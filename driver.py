from ser_iare import *
from main import *
from ser_plots import *
#from tf_mfcc import *

classes=6

df=make_df()
print(df.head())
print('\n',df.columns,df.shape)
x=get_mfcc(df)
lengths=[x[i].shape for i in range(len(x))]
print(lengths)
#x=min_size_equalize(x)
#check(x)

#dump_features(x)
df,v=load_featurmes()

if classes ==4:
    y=df['Emotion_4']
x_train,y_train,x_val,y_val,le=split(x,y=df['Emotion'])
print(dict(zip(le.transform(le.classes_),le.classes_)))

print(x_train.shape,x_train[0].shape,end=" ")

#history,model=model(x_train,y_train,x_val,y_val,outshape=classes)

model_1=keras.models.load_model('/Users/kashyapbrahmandam/Desktop/Kaggle Files/content/Saved Models/Best-Model.h5')
loss,acc=get_loss_acc(model,x_val,y_val)

plot_stats(history)
scores_plot(model,x_val,y_val,acc,le)