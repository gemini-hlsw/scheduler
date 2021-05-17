import csv
from os import getcwd

from openpyxl import load_workbook
from datetime import datetime

class Resource:
    def __init__(self,path):
        self.path = getcwd()+path
        self.FPU = {}
        self.FPUr = {}
        self.GRAT = {}
        self.fpu_to_barcode = {}
        self.instruments = {}
        self.mode = {}
        self.LGS = {}
        self.IFU = {'gs': {}, 'gn': {}}

    def _load_fpu(self, name, site):
        barcodes = {}
        ifu = {}
        with open(f'{self.path}/GMOS{site.upper()}_{name}201789.txt') as f:
            reader =  csv.reader(f, delimiter=',') 
            for row in reader:
                barcodes[row[0].strip()] = [i.rstrip() for i in row[2:]]
                ifu[row[0].strip()] = row[1].strip()
        return ifu, barcodes
                
    def _load_grat(self, site):
        out_dict = {}
        with open(f'{self.path}/GMOS{site.upper()}_GRAT201789.txt') as f:
            reader =  csv.reader(f, delimiter=',') 
            for row in reader:
                out_dict[row[0].strip()] = [i.rstrip() for i in row[1:]]
        return out_dict

    def _load_f2b(self, site):
        out_dict = {}
        with open(f'{self.path}/gmos{site}_fpu_barcode.txt') as f:
            
            for row in f:
                fpu, barcode = row.split()
                out_dict[fpu] = barcode
        return out_dict
    
    def _to_bool(self, b):
        return True if b == 'Yes' else False

    def _excel_reader(self,sites):
        workbook = load_workbook(filename=f'{self.path}/2018B-2019A Telescope Schedules.xlsx')
        for site in sites:
            sheet = workbook[site]
            self.instruments[site.lower()] = {row[0].value.strftime("%Y-%m-%d"): 
                                              [c.value for c in row[3:] ] for row in sheet.iter_rows(min_row=2)}

            self.mode[site.lower()] = {row[0].value.strftime("%Y-%m-%d"): row[1].value for row in sheet.iter_rows(min_row=2)}
            self.LGS[site.lower()] = {row[0].value.strftime("%Y-%m-%d"): 
                                      self._to_bool(row[2].value) for row in sheet.iter_rows(min_row=2)}
           
        if not self.instruments or not self.mode or not self.LGS:
            raise Exception("Problems on reading spreadsheet...") 

    def connect(self, sites):
        """
        Allows the mock to load all the data locally, emulating a connection to the API.

        """
        for site in sites:

            _site = 'gs' if site == 's' else 'gn'
            self.IFU[_site]['FPU'], self.FPU[_site] = self._load_fpu('FPU',site)
            self.IFU[_site]['FPUr'], self.FPUr[_site] = self._load_fpu('FPUr',site)
            self.GRAT[_site] = self._load_grat(site)
            self.fpu_to_barcode[_site] = self._load_f2b(site)

        if not self.FPU or not self.FPUr or not self.GRAT:
            raise Exception("Problems on reading files...") 
        
        self._excel_reader(['GS','GN'])

    def night_info(self, info, site, date):

        site = 'gs' if site == site.GS else 'gn'

        if info == 'fpu':
            return self.FPU[site][date] if date in self.FPU[site] else None

        elif info == 'fpur':
            return self.FPUr[site][date] if date in self.FPUr[site] else None

        elif info == 'grat':
            return self.GRAT[site][date] if date in self.GRAT[site] else None

        elif info == 'instr':
            return self.instruments[site][date] if date in self.instruments[site] else None
        
        elif info == 'LGS':
            return self.LGS[site][date] if date in self.LGS[site] else None
        
        elif info == 'mode':
            return self.mode[site][date] if date in self.mode[site] else None
        
        elif info == 'fpu-ifu':
            return self.IFU[site]['FPU'][date] if date in self.IFU[site]['FPU'] else None
        elif info == 'fpur-ifu':
            return self.IFU[site]['FPUr'][date] if date in self.IFU[site]['FPUr'] else None
        else:
            print(f'No information about {info} is stored')
            return None
        
        

    