import pandas as pd
from icecream import ic

class fs:
	def __init__(self):
		self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		self.t_ranges = [[0,0.2],[0.2,0.3],[0.3,0.7]]
		self.f18_inbending_total_lumi = 5.511802214933477e+40
		self.f18_inbending_total_lumi_inv_nb = 5.511802214933477e+7
		self.Ebeam = 10.6 #GeV, CLAS12 beam energy
		self.m_p = 0.938 #GeV, proton mass
		self.alpha = 1/137 #, fine structure constant



		