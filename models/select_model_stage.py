# ----------------------------
# select for model
# ----------------------------
def define_Model(opt,stage0=False,stage1=False,stage2=False):
    model = opt['model']
    #model = 'stage'
    if model == 'stage': 
        from models.model_stage import ModelStage as M 
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt,stage0,stage1,stage2)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m

