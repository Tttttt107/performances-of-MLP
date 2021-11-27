from packages import * 

class Layer(torch.nn.Sequential):
#define a MLP model structure
    
    def __init__(self, *, in_features, out_features, dropout):
        super().__init__()
        self.add_module('Linear', torch.nn.Linear(in_features=in_features, out_features=out_features))
        #add dropout
        self.add_module('Dropout', torch.nn.Dropout(dropout)) if isinstance(dropout, float) else None
        #use GELU as the activation function
        self.add_module('Activation', torch.nn.GELU())
        
        
class MLP(torch.nn.Sequential): 
    
    def __init__(self, configs, dropout):
        super().__init__()
        for index, layout in enumerate(configs): 
            feature_in, feature_out = layout
            self.add_module(f'Layer {index}', Layer(in_features=feature_in, out_features=feature_out, dropout=dropout))

    #this step is to calculate the size of each MLP model (the size of MLP model is the number of parameters)
    def compute_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
    
