params = {
    #structure:300,512
    'structure':'ssd_300',
    #base net:vgg_16
    'feature extractor':'vgg_16',
    #dataset
    'dataset':{
        'name':'pascal',
        'year':'2007',
        'type':'trainval'
    }
    #type:train,eval
    'task':'train',
    #training param
    'train params':{
        #initial learning rate
        'learning rate':'0.001',
        #decay factor
        'decay factor':'0.05',
        #decay epoch
        'decay epoch':'10000',
        #display step,show average loss per % step
        'display step':'500',
        
    }
}
