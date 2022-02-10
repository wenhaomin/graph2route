# -*- coding: utf-8 -*-
import os
import torch
from pprint import pprint
os.environ['MKL_SERVICE_FORCE_INTEL']='1'
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

def run(params):
    pprint(params)
    model = params['model']
    if model == 'graph2route_pd':
        import algorithm.graph2route_pd.train as graph2route_pd
        graph2route_pd.main(params)
    if model == 'graph2route_logistics':
        import algorithm.graph2route_logistics.train as graph2route_logistics
        graph2route_logistics.main(params)

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    from my_utils.utils import multi_thread_work, dict_merge

    params = vars(get_params())

    args_lst = []
    for model in ['graph2route_pd']:
        for max_num in [25]:
            if model in ['graph2route_pd','graph2route_logistics']:
                for hs in [8]:
                    for cou_embed in [10]:
                        for gcn_num_layers in [2]:
                            gcnru_params = {'model': model, 'hidden_size': hs,
                                            'gcn_num_layers': gcn_num_layers,  'max_num':max_num, 'courier_embed_dim':cou_embed}
                            gcnru_params = dict_merge([params, gcnru_params])
                            args_lst.append(gcnru_params)

    print(args_lst)
    for p in args_lst:
        run(p)
        print('Finished')








