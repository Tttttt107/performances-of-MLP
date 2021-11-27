from packages import *


class dataset(torch.utils.data.Dataset):
    
    def __init__(self, filename, dtype='train'):
        
        self.data = pd.read_csv(f'data/processed/{filename}.csv').to_numpy()
        
        self.rows = list(range(self.data.shape[0]))
        #split training dataset and testing dataset
        #training data (90%), testing data (10%)
        self.rows = self.rows[ : int(len(self.rows)*0.9)] if dtype =='train' else self.rows[int(len(self.rows)*0.9) : ]
        
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, index): 
        
        data = np.float32(self.data[index])
        data, target = data[:-1], data[-1]

        #this step is to encode the label into numerical value
        #encode genuine to 0 and posed to 1
        empty = np.zeros((2, data.shape[0]))
        empty[0] = data 
        empty[1] = np.repeat(target, len(data))
        empty = np.float32(empty)
        
        return torch.from_numpy(empty)
    

def process_data():
    
    data = pd.read_csv('data/raw/anger.csv')
    target = data['Label'] == 'Genuine'
    target = target.astype(int)

    #drop the irrelevant columns in the dataset
    data.drop(columns=['index', 'Video', 'Label'], inplace=True)

    #normalize all values in the dataset
    data = MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(data=data)
    #the column "label" as the target in MLP model
    data['label'] = target 
    
    data = data.sample(frac=1)
    
    data.to_csv('data/processed/data.csv', index=False)
    
    return None


if __name__ == '__main__':
    process_data()
    