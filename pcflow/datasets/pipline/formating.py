from pcflow.utils.data_utils.data_collate import DataContainer as DC


class Collection(object):
    def __init__(self,
                 meta_keys=[],
                 list_keys=[],
                 stack_keys=[]):
        self.meta_keys = meta_keys
        self.list_keys = list_keys
        self.stack_keys = stack_keys
    
    def __call__(self, data_dict):
        results = {}
        for mk in self.meta_keys:
            results[mk] = DC(data_dict[mk], cpu_only=True)
        for lk in self.list_keys:
            results[lk] = DC(data_dict[lk], cpu_only=False, stack=False)
        for sk in self.stack_keys:
            results[sk] = DC(data_dict[sk], cpu_only=False, stack=True)
        return results