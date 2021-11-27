from packages import *

folder_dir = 'data/train_logs'

plt.figure(figsize=(40, 20))

for folder in os.listdir(folder_dir):

    data = pd.read_csv(f'{folder_dir}/{folder}/record.csv')

    plt.plot(data['epoch'], data['train_loss'], linestyle='-', label=folder)
    plt.scatter(data['epoch'], data['train_loss'], s=20)

plt.legend(loc = 0, prop = {'size':30})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()