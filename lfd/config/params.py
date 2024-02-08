
def get_params(target=None, set_aside=None, mode='binaryclass'):
    '''
    This function provides a parameter dictionary for the pipe.learn() function.
    
    Arguments
    ---------
    target : String
            Target variable used in target encoding, biselection, and modelling.
    set_aside : List(String)
            List of variables not to be included in any transformer or as feature for modelling.
    mode : String, in ('linear', 'binaryclass', 'multiclass') 
            What kind of model to build.
    '''
    params = dict(
        set_aside = set_aside,
        data = dict(
            add_noise = dict(seed=0),
            test_split = dict(test_size=0.2, stratify_col=None, seed=0),
        ),
        transform = dict(
            uniselector = dict(min_occ=0.01, max_occ=0.99),
            imputer = dict(default_cat='MISSING', default_cont='median'),
            encoder = dict(min_occ=0.01, method='onehot' if target is None else 'target', target=target),
            biselector = dict(threshold=0.8, target=target),
        ),
        model = dict(
            target=target, mode=mode, seed_train=0,
            base0 = dict(algorithm='xgboost', name='Xgboost', hyper_params=dict(
                n_estimators=100, max_depth=6
            )),
            calibrate = dict(algorithm='isotonic', hyper_params=dict(method='quantile')) \
                if mode=='linear' else dict(algorithm='regression', hyper_params=dict()),
        ),
    )
    return params