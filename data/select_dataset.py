
def define_Dataset(phase,dataset_opt):

    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['train']:
        from data.dataset_train import DatasetTSMS as D

    elif dataset_type in ['test']:
        from data.dataset_test import DatasetTSMS as D

    elif dataset_type in ['val']:
        from data.dataset_val import DatasetTSMS as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(phase,dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
