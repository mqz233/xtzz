import json

from jsonpath import jsonpath


class Read_json(object):
    dict = {}

    def __init__(self, params):
        self.params = params
        self.frame = []
        self.svv = []
        self.master = []
        self.radar = []
        self.posx = []
        self.posy = []
        self.posz = []
        self.v = []
        self.northv = []
        self.upv = []
        self.eastv = []
        self.psi = []
        self.gv = []
        self.iv = []
        self.comm = []

    def get_json_data(self):
        with open('push_00002.json', 'rb') as f:
            params = json.load(f)
            # print(params)
            dict = params
        f.close()
        return dict

    def data_analysis(self):
        self.frame = jsonpath(self.params, '$..frame')[0]
        self.svv = jsonpath(self.params, '$...svv')[0]
        self.master = jsonpath(self.params, '$...master')[0]
        self.radar = jsonpath(self.params, '$...radar')[0]
        self.posx = jsonpath(self.params, '$...posx')[0]
        self.posy = jsonpath(self.params, '$...posy')[0]
        self.posz = jsonpath(self.params, '$...posz')[0]
        self.v = jsonpath(self.params, '$...v')[0]
        self.northv = jsonpath(self.params, '$...northv')[0]
        self.upv = jsonpath(self.params, '$...upv')[0]
        self.eastv = jsonpath(self.params, '$...eastv')[0]
        self.psi = jsonpath(self.params, '$...psi')[0]
        self.gv = jsonpath(self.params, '$...gv')[0]
        self.iv = jsonpath(self.params, '$...iv')[0]
        self.comm = jsonpath(self.params, '$...comm')[0]
        return self.frame, self.svv, self.master, self.radar, self.posx, self.posy, self.posz, self.v, self.northv, self.upv, self.eastv, self.psi, self.gv, self.iv, self.comm
