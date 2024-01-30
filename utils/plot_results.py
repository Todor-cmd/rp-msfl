import os
import numpy as np
import seaborn as sns;
import pandas as pd;
import matplotlib.pyplot as plt

cases = ['no-attack', 'attack-case-1', 'attack-case-2', 'fmes-median', 'fmes-krum-2', 'fmes-m-krum', 'fmes-trimmed-mean', 'fmes-bulyan' ,'fmes-dnc-2']

# for case in cases:
#     directory = 'results/alexnet-fashion/' + case + '/'

#     # List all files in the directory
#     all_files = os.listdir(directory)

#     # Filter only CSV files
#     csv_files = [file for file in all_files if file.endswith('.csv')]

#     # Sort the CSV files alphabetically (to ensure consistent order)
#     csv_files.sort()

#     # Assign the first three CSV files to df, df1, and df2 respectively
#     df = pd.read_csv(os.path.join(directory, csv_files[0]))
#     df1 = pd.read_csv(os.path.join(directory, csv_files[1]))
#     df2 = pd.read_csv(os.path.join(directory, csv_files[2]))

#     df_te_acc = df['Final Test Accuracy'].iloc[-1]
#     df1_te_acc = df1['Final Test Accuracy'].iloc[-1]
#     df2_te_acc = df2['Final Test Accuracy'].iloc[-1]

    
#     print(case, ': ',df_te_acc, '| ', df1_te_acc, '| ', df2_te_acc)

#     numbers = [df_te_acc, df1_te_acc, df2_te_acc]

    
directory = 'results/alexnet-fashion/' + cases[0] + '/'
all_files = os.listdir(directory)
csv_files1 = [file for file in all_files if file.endswith('.csv')]
csv_files1.sort()
    
directory2 = 'results/alexnet-fashion/' + cases[1] + '/'
all_files2 = os.listdir(directory2)
csv_files2 = [file for file in all_files2 if file.endswith('.csv')]
csv_files2.sort()

directory3 = 'results/alexnet-fashion/' + cases[2] + '/'
all_files3 = os.listdir(directory3)
csv_files3 = [file for file in all_files3 if file.endswith('.csv')]
csv_files3.sort()


df1 = pd.read_csv(os.path.join(directory, csv_files1[1]))
df2 = pd.read_csv(os.path.join(directory2, csv_files2[1]))
df3 = pd.read_csv(os.path.join(directory3, csv_files3[0]))

# Calculate rolling mean with a window of 30
df1['Rolling_Mean'] = df1['Validation Accuracy'].rolling(window=30).mean()
df2['Rolling_Mean'] = df2['Validation Accuracy'].rolling(window=30).mean()
df3['Rolling_Mean'] = df3['Validation Accuracy'].rolling(window=30).mean()

# Plot the accuracy using Seaborn
sns.set_theme()
sns.lineplot(x=df1.index, y='Rolling_Mean', data=df1, label='No Attack')
sns.lineplot(x=df2.index, y='Rolling_Mean', data=df2, label='Attack Case 1')
sns.lineplot(x=df3.index, y='Rolling_Mean', data=df3, label='Attack Case 2')

# Label the axes
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Fashion-MNIST with AlexNet - Rolling Mean (Window=30)')

# Show the plot
plt.show()

# # Calculate rolling mean with a window of 30
# df1= df1['Validation Accuracy']
# df2= df2['Validation Accuracy']
# df3= df3['Validation Accuracy']

# # Plot the accuracy using Seaborn
# sns.set_theme()
# sns.lineplot(x=df1.index, y = df1, data=df1, label='No Defense')
# sns.lineplot(x=df2.index, y = df2, data=df2, label='Attack Case 1')
# sns.lineplot(x=df3.index, y = df3, data=df3, label='Attack Case 2')



# # Label the axes
# plt.xlabel('Epoch')
# plt.ylabel('Validation Accuracy')
# plt.title('Fashion-MNIST with VGG11')

# # Show the plot
# plt.show()



