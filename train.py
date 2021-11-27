from packages import * 

from data import dataset

from model import MLP

# parameters
DROPOUT = 0.5
START_LR = 1e-2
BATCH_SIZE = 1024
EPOCH_SAMPLES = 2 ** 15
MAX_EPOCH = 100

LINE_WIDTH = 80
#to store the records of training MLP models
LOG_PATH = 'data/train_logs'


def train(net_obj, scheduler_head):
    
    # initialize dataset 
    train_dataset = dataset('data', dtype='train')
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=EPOCH_SAMPLES)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                                                   pin_memory=True, drop_last=False)
    
    test_dataset = dataset('data', dtype='test')
    
    # mkdir for records 
    time = datetime.now().strftime('%Y_%b_%d_%p_%I_%M_%S')
    #name of the file: MLP_number of parameters (size)
    log_path = f'{LOG_PATH}/MLP_{net_obj.compute_params()}'
    os.mkdir(log_path)
    os.mkdir(f'{log_path}/test')

    #each file contains epoch, train_loss, test_loss, train accuracy, test accuracy, train_f1 and test_f1
    with open(f'{log_path}/record.csv', 'a') as fileout: 
        fileout.write(','.join( ['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'train_f1', 'test_f1'] ) + '\n')

    #print the size of each MLP model (number of parameters)
    print(f'net parameters: {net_obj.compute_params():_}')
    
    model = torch.nn.DataParallel(net_obj)
    #define a loss function
    loss_function = torch.nn.CrossEntropyLoss()

    #define the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=START_LR, total_steps=MAX_EPOCH*len(train_dataloader))
    
    for epoch_current in range(MAX_EPOCH):
        #the loop to calculate the information recorded in the file
        
        train_loss = [] 
        test_loss = [] 
        train_f1 = [] 
        train_accuracy = [] 
        test_input = [] 
        test_target = [] 
        test_prediction = [] 
        
        print('-' * LINE_WIDTH)
        model.train() 
        for input_data in train_dataloader: 
            # B * dimension 
            # dimension -> dataset -> 2(features/target) * 6(featres size)
            # B * 2(features/target) * 6(featres size)
            
            # input_data.shape -> B * 6 
            # target_data.shape -> B, 
            input_data, target_data = input_data[:,0,:], input_data[:,1,0].long()
            prediction = model(input_data)
            
            loss = loss_function(prediction, target_data)
            
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            probability = torch.nn.Softmax(dim=1)(prediction).detach()
            _, prediction = torch.max(probability, 1)

            prediction = prediction.numpy().astype(float)
            target_data = target_data.detach().numpy().astype(float)
            
            train_f1.append(f1_score(target_data, prediction))
            train_accuracy.append(accuracy_score(target_data, prediction))
            train_loss.append(loss.detach().numpy())

        model.eval()        
        with torch.no_grad(): 
            for index in range(len(test_dataset)):
                
                input_data = test_dataset[index]
                # input_data.shape -> B * 6
                # target_data -> B, 
                input_data, target_data = input_data[0:1,:], input_data[1:2,0].long()
                prediction = model(input_data)

                loss = loss_function(prediction, target_data)
                test_loss.append(loss.numpy().astype(float))
                
                test_input.append(input_data.numpy().flatten().astype(float))
                test_target.append(target_data.numpy().flatten().astype(float))

                probability = torch.nn.Softmax(dim=1)(prediction)
                _, prediction = torch.max(probability, 1)
                test_prediction.append(prediction.numpy().flatten().astype(float))
        
        train_loss = sum(train_loss)/len(train_loss)
        train_f1 = sum(train_f1) / len(train_f1)
        train_accuracy = sum(train_accuracy) / len(train_accuracy)
        test_loss = sum(test_loss)/len(test_loss)
        test_f1 = f1_score(test_target, test_prediction)
        test_accuracy = accuracy_score(test_target, test_prediction)
        records = [train_loss, test_loss, train_accuracy, test_accuracy, train_f1, test_f1]
        records = [[round(float(item), 5) for item in records]]
        
        print_out = pd.DataFrame(data=records, columns=['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'train_f1', 'test_f1'])
        print(f'{scheduler_head}[{epoch_current}/{MAX_EPOCH}]', print_out.to_string(), sep='\n')
        
        with open(f'{log_path}/record.csv', 'a') as fileout: 
            fileout.write(','.join([str(epoch_current)] + [str(round(item, 2)) for item in records[0]]) + '\n')
            
        with open(f'{log_path}/test/test_{epoch_current}.csv', 'a') as fileout: 
            
            for data, target, prediction in zip(test_input, test_target, test_prediction):
                package = list(data) + list(target) + list(prediction)
                fileout.write(','.join([str(item) for item in package]) + '\n') 
        
    return None 


def scheduler():
    
    # build MLP models
    # same number of layers but different size
    '''
    net_configs = [((6, 10), (10, 8), (8, 2)), #two hidden layers
                   ((6, 12), (12, 18), (18, 2)), #two hidden layers
                   ((6, 14), (14, 24), (24, 2))] #two hidden layers
    '''

    net_configs = [((6, 12), (12, 2)), #one hidden layer
                  ((6, 10), (10, 12), (12, 2)),  #two hidden layers
                  ((6, 12), (12, 24), (24, 16), (16, 2)), #three hidden layers
                  ((6, 12), (12, 16), (16, 32), (32, 24), (24, 2)), #four hidden layers
                  ((6, 12), (12, 24), (24, 36), (36, 48), (48, 24), (24, 2)), #five hidden layers
                  ((6, 12), (12, 24), (24, 36), (36, 48), (48, 64), (64, 6), (6, 2))] #six hidden layers

    #train each mlp model
    for index, config in enumerate(net_configs, 1): 
        net = MLP(configs=config, dropout=DROPOUT)
        train(net, f'[{index}/{len(net_configs)}]')
        
    return None 


if __name__ == '__main__':
    scheduler()