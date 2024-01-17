import seaborn as sns;
import pandas as pd;
import matplotlib.pyplot as plt

df = pd.read_csv('results/alexnet-cifar10/attack-case-1/170124-1806-cifar10-attack case 1-Fedmes-1500e-2att-min-max-alexnet.csv')
df1 = pd.read_csv('results/alexnet-cifar10/attack-case-2/170124-1820-cifar10-attack case 2-Fedmes-1500e-2att-min-max-alexnet.csv')
df2 = pd.read_csv('results/alexnet-cifar10/no-attack/170124-1424-cifar10-multi-cross-Fedmes-1500e-0att-min-max-alexnet.csv')



# Plot the accuracy using Seaborn
sns.lineplot(x=df.index, y='Accuracy', data=df, label='Attack Case 1')
sns.lineplot(x=df1.index, y='Accuracy', data=df1, label='Attack Case 2')
sns.lineplot(x=df2.index, y='Accuracy', data=df2, label='No Attack')

# Label the axes
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Cifar10 with AlexNet')

# Show the plot
plt.show()



