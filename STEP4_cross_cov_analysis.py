from helpers.cross_cov_analysis import * 
from helper.skeletonise import *


fname = 'dictionary_name.pkl'
fname0 = 'base_fname' # calcium extraction filename *.xlsx
with open(fname, 'rb') as dict_file:
	dataset = pickle.load(dict_file) # need to contain neuron names, ratios, velocity


Neurons = []
Correlations = []
Shift = []
CorrType = []
Windows = []

skeleton, fps = load_skeletons(path, fname0, npoints)
try:
    skeleton = correct_head_tail_neural_data(skeleton, os.path.join(path, fname0.replace('-', '_')), height)
except:
    print('no skeletons')
length = calculateLength(skeleton)
if 'PLM' in dataset['neurons']:
    ii = dataset['neurons'].index('PLM')
else:
    ii = 0
ratio = dataset['ratio'][ii]
validIdxs = mark_reliable_frames(length, ratio, val_min=-1, val_max=-1, blank=15, fps=8)
validIdxs = np.where(validIdxs)[0]
for inrn, neuron in enumerate(dataset['neurons']):
    vel = dataset['velocity'][inrn]
    ratio = dataset['ratio'][inrn]
    ratio = ratio - np.nanmin(ratio)
    for it, type in enumerate(['ct', 'cn']):
        maxPos, maxVal, Idxs = windowed_cross_corr_max_pos(ratio[validIdxs], vel[validIdxs, it], window, step)
        Correlations.extend(maxVal)
        Shift.extend(maxPos)
        Windows.extend(Idxs)
        CorrType.extend([type] * len(maxVal))
        Neurons.extend([neuron] * len(maxVal))
        