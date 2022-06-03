# -*- coding: utf-8 -*-

def run(params):
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

    params = vars(get_params())
    params['model'] = 'graph2route_logistics'
    #graph2route_logistics: For pick-up route prediction in logistics
    #graph2route_pd: For pick-up then delivery route prediction in food delivery
    run(params)














