from packages import *

folder_dir = 'data/train_logs'

plt.figure(figsize=(40, 20))

for folder in os.listdir(folder_dir):

    data = pd.read_csv(f'{folder_dir}/{folder}/record.csv')

    plt.plot(data['epoch'], data['test_f1'], linestyle='dashed', label=folder)
    plt.scatter(data['epoch'], data['test_f1'], s=20)

plt.legend()
plt.show()