# Simulate Data
import numpy as np
import networkx as nx
from kuramoto import Kuramoto

# Network generation
n_nodes = 50 
p = 1 # p=1 -> all-to-all connectivity
graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=p) 
graph = nx.to_numpy_array(graph_nx)
# Kuramoto model simulation
natural_freq_mean = 20
natural_freq_var = .1
dt = .01
T = 10
K_start = 0 
K_stop = 3 
coupling_vals = np.linspace(K_start, K_stop, 200)
runs = []
for coupling in coupling_vals:     
	model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=n_nodes) 
	model.natfreqs = np.random.normal(natural_freq_mean, natural_freq_var, size=n_nodes)  # reset natural frequencies (20rad/sec 3 Hz)
	act_mat = model.run(adj_mat=graph)   
	runs.append(act_mat)
runs_array = np.array(runs)
coherence = []
for i, coupling in enumerate(coupling_vals):
    coherence.append(
        [model.phase_coherence(vec)
         for vec in runs_array[i, ::].T] 
    )
coherence_array = np.array(coherence)

# Window the data and window both the zero-step and one-step-ahead label time-series
from gtda.time_series import SlidingWindow

window_size = 25 
stride = 1
coherence_threshold = .8
onset_threshold = .5
SW = SlidingWindow(window_size, stride)
# The for-loop steps through each coupling strength value
for i in np.arange(len(coherence_array)):
	data = runs_array[i,::]
	data = np.sin(data.T)
	coherence = coherence_array[i, :]
	labels = coherence
	labels2 = np.where(coherence > coherence_threshold, 1, 0) # sync regime
	labels1 = np.where(coherence > onset_threshold, 1, 0) # onset regime
	labels = labels1 + labels2
	data_sw, labels_sw = SW.fit_transform_resample(data, labels)
	# We want to predict the label of the next window given the current window
	labels_sw_one_step = np.roll(labels_sw, -1)
	labels_sw_one_step[-1] = labels_sw_one_step[-2] # fill last val to its previous value
	if i == 0:
		yr_one_step = labels_sw_one_step
		X_sw = data_sw
	else:
		yr_one_step = np.concatenate((yr_one_step, labels_sw_one_step), axis=0)
		X_sw= np.concatenate((X_sw, data_sw), axis=0)

# Create topological feature vector
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints

PD = PearsonDissimilarity()
X_pd = PD.fit_transform(X_sw)
VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2]) 
Ampl = Amplitude()
X_a = []
PerEnt = PersistenceEntropy()
X_pe = []
NumPts = NumberOfPoints()
X_np = []
for i in np.arange(len(X_pd)):
	X_vr = VR.fit_transform([X_pd[i]])  
	X_a.append(Ampl.fit_transform(X_vr)) 
	X_pe.append(PerEnt.fit_transform(X_vr))
	X_np.append(NumPts.fit_transform(X_vr))
X_a = np.array(X_a)
X_a = np.squeeze(X_a)  
X_pe = np.array(X_pe)
X_pe = np.squeeze(X_pe)  
X_np = np.array(X_np)
X_np = np.squeeze(X_np)     
X_tot = np.concatenate((X_a, X_pe, X_np), axis=1)  

# Create pipeline, train and test
from gtda.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

RFR = GradientBoostingClassifier() 
pipe_predict_ahead = make_pipeline(RFR)
# Split into train/test sets and do training / evaluation
X_train, X_test, y_train_one_step, y_test_one_step = train_test_split(X_tot, yr_one_step, test_size=0.2, random_state=42)
# Do training / evaluation
pipe_predict_ahead.fit(X_train, y_train_one_step)
y_pred_one_step = pipe_predict_ahead.predict(X_test)
score_one_step = pipe_predict_ahead.score(X_test, y_test_one_step)
print('score:', score_one_step)

######################################################################################################
simulate_ES_change_point_detection = 1
#if simulate_ES_change_point_detection:
# Process signals - simulates realtime processing; TODO: figure out how to save and load in previously trained pipe object
import numpy as np
import networkx as nx
from kuramoto import Kuramoto
from gtda.time_series import SlidingWindow
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints

# Get the data
# Network generation
# Network generation
n_nodes = 50 
p = 1 # p=1 -> all-to-all connectivity
graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=p) 
graph = nx.to_numpy_array(graph_nx)
# Kuramoto model simulation
natural_freq_mean = 20
natural_freq_var = .1
dt = .01
T = 10
window_size = 25 
stride = 1
coherence_threshold = .8
onset_threshold = .5
graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=p) 
graph = nx.to_numpy_array(graph_nx)
runs = []
coupling_vals = [1.1] 
onset_threshold = .3
for i, coupling in enumerate(coupling_vals):    
	#print(i)    
	model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=n_nodes) 
	model.natfreqs = np.random.normal(natural_freq_mean, natural_freq_var, size=n_nodes)  # reset natural frequencies (20rad/sec 3 Hz)
	act_mat = model.run(adj_mat=graph)   
	runs.append(act_mat)
runs_array = np.array(runs)
# Get all time series for all coupling values 
coherence = []
for i, coupling in enumerate(coupling_vals):
    coherence.append(
        [model.phase_coherence(vec)
         for vec in runs_array[i, ::].T] 
    )
coherence_array = np.array(coherence) 
data = runs_array[-1]
data = np.sin(data.T)
labels = coherence_array[-1]
labels2 = np.where(labels > coherence_threshold, 1, 0) #  ES sync regime
labels1 = np.where(labels > onset_threshold, 1, 0) #  ES onset regime
labels = labels1 + labels2 # Creates 0 1 2 categories representing normal, onset, sync'd
# We want to predict the label of the next window given the current window
predict_one_step_ahead = 1
if predict_one_step_ahead:
	labels_one_step = np.roll(labels, -1)
	labels_one_step[-1] = labels_one_step[-2] # fill last val to its previous value

# Do classification
SW = SlidingWindow(window_size, stride)
PD = PearsonDissimilarity()
VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])
Ampl = Amplitude()
PerEnt = PersistenceEntropy()
NumPts = NumberOfPoints()
X_sw_sim, yr_one_step_sim = SW.fit_transform_resample(data, labels_one_step)
X_pd_sim = PD.fit_transform(X_sw_sim)
X_vr_sim = VR.fit_transform(X_pd_sim) 
X_a_sim = Ampl.fit_transform(X_vr_sim)
X_pe_sim = PerEnt.fit_transform(X_vr_sim)
X_np_sim = NumPts.fit_transform(X_vr_sim)   
X_tot_sim = np.concatenate((X_a_sim, X_pe_sim, X_np_sim), axis=1) 
y_predict_ahead_sim = pipe_predict_ahead.predict(X_tot_sim) 
print('score_predict: ', pipe_predict_ahead.score(X_tot_sim, yr_one_step_sim))

# Save the labels, data & trained model as a pickle string and / or load it and plot
import pickle
save_pickles = 0
if save_pickles:
	# Save stuff:
	filename = 'model.sav'
	pickle.dump(pipe_predict_ahead, open(filename, 'wb'))
	filename = 'features.sav'
	pickle.dump(X_tot_sim, open(filename, 'wb'))
	filename = 'data_windows.sav'
	pickle.dump(X_sw_sim, open(filename, 'wb'))
	filename = 'labels.sav'
	pickle.dump(yr_one_step_sim, open(filename, 'wb'))
import pickle
load_pickles = 0
if load_pickles:
	# Load stuff:
	filename = 'features.sav'
	X_tot_sim = pickle.load(open(filename, 'rb'))
	filename = 'data_windows.sav'
	X_sw_sim = pickle.load(open(filename, 'rb'))
	filename = 'model.sav'
	pipe_predict_ahead = pickle.load(open(filename, 'rb'))
	filename = 'labels.sav'
	yr_one_step_sim = pickle.load(open(filename, 'rb'))
	y_predict_ahead_sim = pipe_predict_ahead.predict(X_tot_sim) 

# Do plots
import matplotlib.pyplot as plt
import numpy as np

y_predict_sim = np.roll(y_predict_ahead_sim, 1)
y_predict_sim[0] = y_predict_sim[1] # fill first val to its 2nd value
yr_sim = np.roll(yr_one_step_sim, 1)
y_predict_sim[0] = yr_sim[1] # fill first val to its 2nd value
_, t1, t2 = np.where(yr_sim[:-1] != yr_sim[1:])[0]
plt.axvspan(0, t1, facecolor='c', alpha=1.0)
plt.axvspan(t1, t2, facecolor='m', alpha=1.0)
plt.axvspan(t2, len(yr_sim), facecolor='y', alpha=1.0)
x = np.sin(X_sw_sim[:, -1, :] )
x0 = np.zeros(x.shape)
x0[np.where(y_predict_sim==0)] = x[np.where(y_predict_sim==0)]
x1 = np.zeros(np.shape(x))
x1[np.where(y_predict_sim==1)] = x[np.where(y_predict_sim==1)]
x2 = np.zeros(np.shape(x))
x2[np.where(y_predict_sim==2)] = x[np.where(y_predict_sim==2)]
plt.plot(x0, 'g')
plt.plot(x1, 'b')
plt.plot(x2, 'r')
plt.ylabel(r'$\sin(\theta)$', fontsize=25)
plt.xlabel('Time Window', fontsize=25)
plt.ylim([-1., 1.])
plt.show()





