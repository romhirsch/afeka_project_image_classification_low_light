import pandas as pd
import matplotlib.pyplot as plt
path = r"C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\DP\EfficientNetV2B0_FineTurning_0_10.xlsx"
df = pd.read_excel(path)
fig, ax = plt.subplots(6, sharex=True)
for i, d in enumerate(['Test', 'ExDark_test', 'level1', 'level2', 'level3', 'level4']):
    ax[i].plot(df[df['Dataset']==d]['Model'],df[df['Dataset']==d]
    [['Accuracy', 'Precision','Recall', 'F1-score']],
               label=['Accuracy', 'Precision','Recall', 'F1-score'])
    ax[i].set_ylabel(d)
lines, labels = ax[0].get_legend_handles_labels()
fig.legend(lines, labels)