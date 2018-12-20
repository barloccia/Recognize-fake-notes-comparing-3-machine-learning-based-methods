import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix as cm
from mlxtend.evaluate import mcnemar_table as mt
from sklearn.metrics import accuracy_score

#
#
#   USEFUL FUNCTIONS
#
#

def classification_analysis(data, labels, idx_train, idx_test, C=None, K=None):
    assert C!=K,"Please set K or C values correctly"
    if (C != None):
        reg = LogisticRegression(penalty='l2', C=C, tol=1e-10)
        reg.fit(data[idx_train,:], labels[idx_train,0])
        prob = reg.predict_proba(data)[:, 1]
        squares = (prob - labels[:, 0]) ** 2
        return (np.mean(squares[idx_train]),
                np.mean(squares[idx_test]))
    else:
        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(data[idx_train, :], labels[idx_train, 0])
        prob = neigh.predict(data)
        squares = (prob - labels[:, 0]) ** 2
        return (np.mean(squares[idx_train]),
                np.mean(squares[idx_test]))

def classification_analysis_bayes(data, labels, idx_train, idx_val, band=1.0):
    models = compute_kde(data[idx_train,:], labels[idx_train,0], bandwidth=band)
    predicts_train = np.array(predict(models, data[idx_train], labels))
    predicts_val = np.array(predict(models, data[idx_val], labels))
    accuracy_val = accuracy_score(labels[idx_val,0], predicts_val)
    accuracy_train = accuracy_score(labels[idx_train, 0], predicts_train)
    return accuracy_val, accuracy_train

def standardization(X, bayes=False):
    assert X.shape[1] > 1, "Please verify your matrix"
    if not bayes:
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i], 0)) / np.std(X[:, i], 0)
    else:
        for i in range(X.shape[1]):
            # TO 0-1
            X[:, i] = (X[:, i] - np.min(X[:, i]))
            X[:, i] = X[:, i] / np.max(X[:, i])

    return X

def compute_kde(X, Y, bandwidth=1.0):
    assert X.shape[1] >= 2, "Please Verify X's dimensions"
    kernels = []
    classes = np.sort(np.unique(Y))
    for _class in classes:
        idx = np.where(Y == _class)
        for i in range(X.shape[1]):
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(X[idx[0],i].reshape(-1,1))
            kernels.append((i,_class,kde))
    return kernels

def predict(models, X, Y):
    # MODELS: FEATURE, CLASS, PROPERLY KDE
    class_0_models = models[:4]
    class_1_models = models[4:]
    predicts = []
    for xi in X:
        total_class_0 = calcPrior(Y,0)
        total_class_1 = calcPrior(Y,1)
        for i in range(xi.shape[0]):
            # GET THE LIKELIHOOD FOR EACH FEATURE AND CLASS. SUM ALL THEM
            total_class_0 += class_0_models[i][2].score_samples(xi[i])
            total_class_1 += class_1_models[i][2].score_samples(xi[i])
        if total_class_0 > total_class_1:
            predicts.append(0)
        else:
            predicts.append(1)
    return predicts

def mcnemar(name1, name2, predict_1, predict_2, Y_te):
    cm_1 = cm(Y_te,predict_1)
    cm_2 = cm(Y_te,predict_2)
    errorsLog = cm_1[1,0] + cm_1[0,1]
    errorsKnn = cm_2[1,0] + cm_2[0,1]
    logVsKnn = mt(y_target=Y_te.ravel(),
                       y_model1=predict_1,
                       y_model2=predict_2)
    res = ((abs(logVsKnn[1,0]-logVsKnn[0,1])-1)**2)/(logVsKnn[1,0]+logVsKnn[0,1])
    print(name1, ' VS ', name2, ' = ', res)

def min_k(data_plot):
    var = 1000
    index = 10000
    for i in range (len(data_plot)):
        if data_plot[i] <=var:
            var=data_plot[i]
            index=i
    return index

def calcPrior(mat, c):
    return np.log(len(mat[mat==c])/len(mat))

#
#
#   ISTANCES & VALUES DEFINITION - RETRIVING AND ADAPT THE DATA FORM
#
#

# DIRECTORY WHERE SAVE THE GRAPHS AND CSV DATA FILE
graph_dir = 'graphs/'
data_dir = 'data/'
# GENERAL
models = []
data_plot_tr = []
data_plot_va = []
data_plot_C = []
data_plot_K = []
folds = 2
kf = StratifiedKFold(n_splits=folds)

# CORE VALUES FOR THE ISTANCED MODELS
C_start = 1.0 # REGULARIZATION FOR LOGISTIC REGRESSION
K_start = 1 # K-NN
bandwidths = np.arange(0.01, 1, 0.02) # KDE BANDWIDTH FOR BAYES
iter_band = -1 # BAYES

# THE FILE: F1,F2,F3,F4,class
lines = open(data_dir+'TP1-data.csv').readlines()
data = []
for line in lines:
    split = line.split(',')
    variance, skewness, curtosis, entropy, _class = split[0], split[1], split[2], split[3], split[4]
    data.append( (float(variance), float(skewness), float(curtosis), float(entropy), int(_class)) )
data = np.array(data)
# SHUFFLE THE MATRIX
data_shff = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
features = data_shff[:,:4]
labels = (data_shff[:,4]).reshape((data.shape[0],1))
X_tr, X_te, Y_tr, Y_te =  train_test_split(features,labels,test_size=0.33,stratify=labels)
# 0-1 GOOD FOR BAYES IMPLEMENTATION
X_trB = standardization(X_tr,bayes=True)
X_teB = standardization(X_te,bayes=True)
# GENERAL USE
X_tr = standardization(X_tr)
X_te = standardization(X_te)

#
#
#   LOGISTIC REGRESSION CLASSIFIER
#
#

# FIND BEST C FOR REGRESSION THROUGH THE CROSS-VALIDATION
for iter in range(20):
    tr_err = va_err = 0
    C_start *= 2
    for idx_train, idx_va in kf.split(X_tr, Y_tr):
        r, v = classification_analysis(X_tr, Y_tr, idx_train, idx_va, C = C_start)
        tr_err += r
        va_err += v
    data_plot_tr.append(tr_err / folds)
    data_plot_va.append(va_err / folds)
    data_plot_C.append(C_start)

# PLOT CODING OMITTED
# THE BEST VALUE OF C FOR MOST OF THE EXPERIMENTs WAS IN THE RANGE: 6-8
C = data_plot_C[data_plot_va.index( min(data_plot_va))]
print('Best C value: ', C)
reg = LogisticRegression(tol=1e-10, C=C)
reg.fit(X_tr,Y_tr.ravel()) # RAVEL: (rows,) -> (rows,1)
prob = reg.predict_proba(X_te)
predict_log = reg.predict(X_te)
log_score = reg.score(X_te,Y_te.ravel())
print('Score for logistic classifier: ',log_score)

plt.figure(1,figsize=(12,8),frameon=False)
plt.plot(np.log10(data_plot_C),data_plot_va,'-',color='green')
plt.plot(np.log10(data_plot_C),data_plot_tr,'-',color='red')
plt.title('Tr & va error, best C = ' + str(C) + ' Test score: ' + str('%.2e' %log_score), size=18)
plt.axvline(np.log10(C))
plt.legend(('validation_min: ' + str('%.3e' % min(data_plot_va)),
           'training min: ' + str('%.3e' % min(data_plot_tr)),
           'C: ' + str('%.3e' %C)

           ))
plt.xlabel('log(C)', size=14)
plt.ylabel('Error %', size=14)
plt.savefig(graph_dir + "best_C.png")
plt.close()

#
#
#   K-NEIGHBORS CLASSIFIER
#
#

data_plot_tr = []
data_plot_va = []
# FIND THE BEST K VALUE FOR NEIGHBORS THROUGH THE CROSS-VALIDATION
for iter in range(40):
    tr_err = va_err = 0
    K_start += 2
    for idx_train, idx_va in kf.split(X_tr, Y_tr):
        r, v = classification_analysis(X_tr, Y_tr, idx_train, idx_va, K = K_start)
        tr_err += r
        va_err += v
    data_plot_tr.append(tr_err / folds)
    data_plot_va.append(va_err / folds)
    data_plot_K.append(K_start)

# PLOT CODING OMITTED
# THE BEST VALUE OF K FOR MOST OF THE EXPERIMENTs WAS IN THE RANGE: 1-5
K_min= min_k(data_plot_va)
print('Best K value: ', data_plot_K[K_min])
neigh = KNeighborsClassifier(n_neighbors=data_plot_K[K_min])
neigh.fit(X_tr,Y_tr.ravel())
prob = neigh.predict(X_te)
predict_knn = prob
kNN_score = neigh.score(X_te,Y_te)
print('Score for K-Neighbors clissifier: ',kNN_score)
print('\n------ COMPUTING NAIVE BAYES ------')

plt.figure(1,figsize=(12,8),frameon=False)
plt.plot(np.log10(data_plot_K),data_plot_va,'-',color='green')
plt.plot(np.log10(data_plot_K),data_plot_tr,'-',color='red')
plt.title('Tr & Va error, best K = ' + str( '%.3e' %data_plot_K[K_min]) + ' Test score: ' + str('%.2e' %kNN_score), size=18)
plt.axvline(np.log10(data_plot_K[K_min]))
plt.legend(('validation_min: ' + str('%.3e' % min(data_plot_va)),
           'training min: ' + str('%.3e' % min(data_plot_tr)),
           'K: ' + str('%.3e' %data_plot_K[K_min])

           ))
plt.xlabel('log(K)', size=14)
plt.ylabel('Error %', size=14)
plt.savefig(graph_dir + "best_K.png")
plt.close()

#
#
#   NAIVE BAYES CLASSIFIER
#
#

data_plot_va = []
data_plot_train = []
data_plot_band = []
cont = -1
# FIND THE BEST BANDWIDTH VALUE FOR KERNEL DENSITY FUNCTION THROUGH THE CROSS-VALIDATION
# BANDWIDTHS IS AN ARRAY DERIVED FROM LINSPACE FUNCTION. SEE VALUES AT THE TOP
for iter in range(len(bandwidths)):
    tr_err = 0
    va_err = 0
    cont+=1
    for idx_train, idx_va in kf.split(X_tr, Y_tr):
        val, train = classification_analysis_bayes(X_tr, Y_tr, idx_train, idx_va, band=bandwidths[cont])
        tr_err += (1-train)
        va_err += (1-val)
    data_plot_va.append(va_err / folds)
    data_plot_train.append(tr_err / folds)
    data_plot_band.append(bandwidths[cont])

best_band_index = min_k(data_plot_va)
print('Best bandwidth: ', bandwidths[best_band_index])

kernels = compute_kde(X_teB,Y_te,bandwidth=bandwidths[best_band_index])
predicts = np.array(predict(kernels, X_teB, Y_te))
predict_bayes = predicts
accuracy = accuracy_score(Y_te, predicts)
print('Score for KDE-BAYES clissifier: ', accuracy)
print('----- RESULTS -----')


plt.figure(1,figsize=(12,8),frameon=False)
plt.plot(data_plot_band,data_plot_va,'-',color='green')
plt.plot(data_plot_band,data_plot_train,'-',color='red')
plt.title('Tr & validation error, best bandwidth = ' + str( '%.3e' %bandwidths[best_band_index]) + ' Test score: ' + str('%.2e' %accuracy), size=18)
plt.axvline(bandwidths[best_band_index])
plt.legend(('validation_min: ' + str('%.3e' % np.min(data_plot_va)),
           'training min: ' + str('%.3e' % np.min(data_plot_tr)),
           'bandwidth: ' + str('%.3e' %bandwidths[best_band_index])

           ))
plt.xlabel('bandwidth', size=14)
plt.ylabel('Error %', size=14)
plt.savefig(graph_dir + "best_band.png")
plt.close()

# McNEMAR TESTS

mcnemar('LOGISTIC REGRESSION', 'K-NN', predict_log, predict_knn, Y_te)
mcnemar('LOGISTIC REGRESSION', 'NAIVE BAYES', predict_log, predict_bayes, Y_te)
mcnemar('BAYES', 'K-NN', predict_bayes, predict_knn, Y_te)







