from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_stats(history_logs):
    plt.plot(history_logs.history['accuracy'])
    plt.plot(history_logs.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history_logs.history['loss'])
    plt.plot(history_logs.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def report_res_and_plot_matrix(y_test, y_pred, plot_classes):
    # report metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    # print(f"Classes: {plot_classes}")

    # plot matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()

    tick_marks = np.arange(len(plot_classes))
    plt.xticks(ticks=tick_marks, labels=plot_classes, rotation=90)
    plt.yticks(ticks=tick_marks, labels=plot_classes, rotation=90)

    group_counts = [f'{value:0.0f}' for value in cnf_matrix.flatten()]
    group_percentages = [f'{100 * value:0.1f} %' for value in
                         cnf_matrix.flatten() / np.sum(cnf_matrix)]
    labels = [f'{v1}\n({v2})' for v1, v2 in
              zip(group_counts, group_percentages)]
    n = int(np.sqrt(len(labels)))
    labels = np.asarray(labels).reshape(n, n)
    sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def scores_plot(model,x_val,y_val,acc,label_encoder):
    y_pred = model.predict(x_val).argmax(axis=1)
    print('Optimized model, accuracy: {:5.2f}%'.format(100 * acc))
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    tparams = report_res_and_plot_matrix(y_val, y_pred, label_encoder.classes_)