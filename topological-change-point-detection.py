# This script demonstrates using giotta-tda's API for developing a
# time-series forecasting pipeline

import numpy as np
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints
from gtda.time_series import SlidingWindow
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVM
from gtda.pipeline import make_pipeline
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from kuramoto import Kuramoto
from sklearn.model_selection import train_test_split, KFold

plt.style.use('seaborn')
sns.set_style("whitegrid")
sns.set_context("talk")

# Inits
n_nodes = 50 
p = 1 # p=1 -> all-to-all connectivity
natural_freq_mean = 20
natural_freq_var = .1
dt = .01
T = 10
window_size = 25 # ~ .5 sec should give 1.5 cycles for extracting tda features
stride = 1
coherence_threshold = .8
onset_threshold = .5
K_start = 0 # Run model with different coupling (K) parameters
K_stop = 3 
# Get the data
# Netork generation
graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=p) 
graph = nx.to_numpy_array(graph_nx)
coupling_vals = np.linspace(K_start, K_stop, 200)
runs = []
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
# Labels and data arrays. 
# coherence_array.shape = (num_coupling_vals, num_time): 
# runs_array.shape = (num_coupling_vals, num_nodes, num_time):

SW = SlidingWindow(window_size, stride)
for i in np.arange(len(coherence_array)):
	coherence = coherence_array[i, :]
	data = runs_array[i,::]
	data = np.sin(data.T)
	#data = data[coherence <= coherence_threshold, :] # Discard data where signal is already sync'd
	#labels = coherence[coherence <= coherence_threshold] # Discard labels where signal is already sync'd
	labels = coherence
	labels2 = np.where(coherence > coherence_threshold, 1, 0) #  ES sync regime
	labels1 = np.where(coherence > onset_threshold, 1, 0) #  ES onset regime
	labels = labels1 + labels2
	data_sw, labels_sw = SW.fit_transform_resample(data, labels)
	# We want to predict the label of the next window given the current window
	labels_sw_one_step = np.roll(labels_sw, -1)
	labels_sw_one_step[-1] = labels_sw_one_step[-2] # fill last val to its previous value
	if i == 0:
		yr = labels_sw
		yr_one_step = labels_sw_one_step
		X_sw = data_sw
	else:
		yr = np.concatenate((yr, labels_sw), axis=0)
		yr_one_step = np.concatenate((yr_one_step, labels_sw_one_step), axis=0)
		X_sw= np.concatenate((X_sw, data_sw), axis=0)
#plt.plot(X_sw[-1])
#plt.show()
# Plot 3-dimensional PCA space of the sliding windows for visualizaition purposes
plot_pca = 0
if plot_pca:
	# First flatten the windowed data to a [num_windows, dim1xdim2]-array 
	# This is now a point cloud: num_points = num_windwos and cloud_dim = dim1xdim2
	X_sw_flattened = np.resize(X_sw, (X_sw.shape[0], X_sw.shape[1]*X_sw.shape[2]))
	pca = PCA(n_components=3)
	Y = pca.fit_transform(X_sw_flattened)
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
	plt.show()
PD = PearsonDissimilarity()
X_pd = PD.fit_transform(X_sw)
#plt.imshow(X_pd[10])
#plt.show()
VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])
Ampl = Amplitude()
X_a = []
PerEnt = PersistenceEntropy()
X_pe = []
NumPts = NumberOfPoints()
X_np = []
for i in np.arange(len(X_pd)):
	X_vr = VR.fit_transform([X_pd[i]])  # "precomputed" required on dissimilarity data
	X_a.append(Ampl.fit_transform(X_vr)) # Seems to be a bug - must compute amplitude transform on each diagram individually as opposed to an array of diagrams
	X_pe.append(PerEnt.fit_transform(X_vr))
	X_np.append(NumPts.fit_transform(X_vr))
X_a = np.array(X_a)
X_a = np.squeeze(X_a)  
X_pe = np.array(X_pe)
X_pe = np.squeeze(X_pe)  
X_np = np.array(X_np)
X_np = np.squeeze(X_np)     
X_tot = np.concatenate((X_a, X_pe, X_np), axis=1)  
#fig = VR.plot(X_vr, sample=1000)
#fig.show()
#for i in np.arange(10):
#	print(i, yr[i], X_tot[i])

print('Train/eval phase ...')
#RFR = RandomForestRegressor()
#RFR = RandomForestClassifier() # time series test score: 0.9508196721311475
RFR = GradientBoostingClassifier() # score: 0.9743852459016393
#RFR = svm.SVC() # score: 0.9692622950819673
#pipe = make_pipeline(PD, VR, Ampl, RFR)
pipe_measure = make_pipeline(RFR)
pipe_predict = make_pipeline(RFR)
# Split into train/test sets and do training / evaluation
X_train, X_test, y_train, y_test = train_test_split(X_tot, yr, test_size=0.2, random_state=42)
X_train, X_test, y_train_one_step, y_test_one_step = train_test_split(X_tot, yr_one_step, test_size=0.2, random_state=42)
use_cv_splits = 0
if use_cv_splits:
	# REF: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-oob-py
	cv = KFold(n_splits=5)
	score = 0
	for train, test in cv.split(X_train, y_train):
	        pipe.fit(X_train[train], y_train[train])
	        score += pipescore(cv_clf, X_train[test], y_train[test])
	score /= n_splits
else:
	pipe_measure.fit(X_train, y_train)
	y_pred = pipe_measure.predict(X_test)
	score = pipe_measure.score(X_test, y_test)
	pipe_predict.fit(X_train, y_train_one_step)
	y_pred_one_step = pipe_predict.predict(X_test)
	score_one_step = pipe_predict.score(X_test, y_test_one_step)
print('score:', score)
print('score:', score_one_step)
#y_pred_bin = np.where(y_pred > onset_threshold, 1, 0) 
#print('accuracy', 1-np.sum(np.abs(y_pred_bin - y_test))/len(y_test) )
#plt.subplot(2,1,1)
#plt.plot(y_pred, 'r^')
#plt.subplot(2,1,2)
#plt.plot(y_test, 'b.')
#plt.legend(['y_pred', 'yr'])
#plt.show()

######################################################################################################
simulate_ES_change_point_detection = 1
#if simulate_ES_change_point_detection:
# Process signals - simulates realtime processing; TODO: figure out how to save and load in previously trained pipe object

# Get the data
# Network generation
runs = []
coupling_vals = [3] 
onset_threshold = .4
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
X_sw_sim, yr_one_step_sim = SW.fit_transform_resample(data, labels_one_step)
X_sw_sim, yr_sim = SW.fit_transform_resample(data, labels)
X_pd_sim = PD.fit_transform(X_sw_sim)
X_vr_sim = VR.fit_transform(X_pd_sim) 
X_a_sim = Ampl.fit_transform(X_vr_sim)
X_pe_sim = PerEnt.fit_transform(X_vr_sim)
X_np_sim = NumPts.fit_transform(X_vr_sim)   
X_tot_sim = np.concatenate((X_a_sim, X_pe_sim, X_np_sim), axis=1) 

y_pred_prob_sim = pipe_measure.predict_proba(X_tot_sim) 
y_pred_prob_one_step_sim = pipe_predict.predict_proba(X_tot_sim) 
y_pred_sim = pipe_measure.predict(X_tot_sim) 
y_meas_sim = pipe_predict.predict(X_tot_sim) 
print('score_measure: ', pipe_measure.score(X_tot_sim, yr_sim))
print('score_predict: ', pipe_predict.score(X_tot_sim, yr_one_step_sim))

# Implement the modified bayes filter
bel = np.array([0, 0, 0])
P = np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]]) 
a1, a2, p_hit, p_miss = .5, .5, .4, .15 # not so good
a1, a2, p_hit, p_miss = .6, .4, .8, .15 # a bit better
y_pred_filt_sim = []
for i, pz in enumerate(yr_sim):
	# Measurement
	if i == 0:
		bel = y_pred_prob_sim[i, :]
	else:
		state = pz#np.argmax(pz, axis=0)

		if state == 0:
			pz = np.array([p_hit, p_miss, p_miss])
		if state == 1:
			pz = np.array([p_miss, p_hit, p_miss])
		if state == 2:
			pz = np.array([p_miss, p_miss, p_hit])
		bel = np.array([pz[0]*bel[0], pz[1]*bel[1], pz[2]*bel[2]])
		bel = bel / np.sum(bel)	
	#print('measure: ', bel)	
	# Predict ahead with data driven and Markov model
	bel_ML = y_pred_prob_one_step_sim[i, :]
	bel_BF = P.dot(bel)
	# Combine predictions as weighted sum
	bel = a1 * bel_BF + a2 * bel_ML	
	#print('predict ', bel)
	#print('actual  ', yr_one_step_sim[i])
	#input('...')
	y_pred_filt_sim.append(bel)
y_pred_filt_sim = np.array(y_pred_filt_sim)

plt.plot(np.argmax(y_pred_filt_sim, axis=1), 'rs')
plt.plot(np.argmax(y_pred_prob_one_step_sim, axis=1), 'g.')

plt.plot(yr_one_step_sim, 'b^')
plt.show()
















