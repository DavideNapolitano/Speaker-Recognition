import csv
import os
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from Functions import load_data, compute_statistical_features, lab_convertion, clean_features, freq_clean, \
     control_len
from scipy import signal
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
#%%  LOADING DATA
pathDev=r"C:\Users\david\Google Drive\Data Science Lab\Appello Settembre\development"
pathEval=r"C:\Users\david\Google Drive\Data Science Lab\Appello Settembre\evaluation"
def load_data(data_path, phase):
    files = os.listdir(data_path)
    i=0
    if phase=="dev":
        people = [file for file in files]
        X = [None] * 24449
        ID = [None] * 24449
        y = [None] * 24449
        SR = [None] * 24449
        for person in people:
            person_dir = os.listdir(data_path + '\\' + person)
            for track in person_dir:
                data, sr = librosa.load(data_path+'\\'+person+'\\'+track, sr=24000)
                # print(data)
                # print(sr)
                id = track.split('.')[0]
                ID[i]=id
                # X[i] = data.astype(np.float32)
                X[i] = data
                SR[i] = sr
                y[i] = person
                i+=1
            #     break
            # break
    elif phase == "eval":
        wav_files = [file for file in files if file.endswith(".wav")]
        X = [None] * len(wav_files)
        y = [None] * len(wav_files)
        SR = [None] * len(wav_files)
        ID = [None] * len(wav_files)
        for file in wav_files:
            data, sr = librosa.load(data_path + '\\' + file, sr=24000)
            id = file.split('.')[0]
            ID[i] = id
            # X[i] = data.astype(np.float32)
            X[i] = data
            SR[i] = sr
            i += 1
    else:
        raise Exception(f"Error - phase '{phase}' not recognised.")

    return X, y, SR, ID, i
#%%
dev_X, dev_y, dev_SR, dev_ID, dev_i = load_data(pathDev, phase="dev")
eval_X, _, eval_SR, eval_ID, eval_i = load_data(pathEval, phase="eval")
#%%
def lab_convertion(dev_y):
    num_dev_y=[]
    diz={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9}
    for lbl in dev_y:
        num_dev_y.append(diz[lbl])
    num_dev_y=np.array(num_dev_y)
    return num_dev_y
numerical_label=lab_convertion(dev_y)
#%%--------------------------------------------------CONTROLLO SR-------------------------------------------------------
d_SR=np.array(dev_SR)
e_SR=np.array(eval_SR)
if (d_SR==24000).all():
    print("DEV OK")
if (e_SR==24000).all():
    print("EVAL OK")
#%%-----------------------------------------------DISTRIBUZIONE LUGHEZZE------------------------------------------------
def control_len(dev_X, dev_SR):
    len_dev=[]
    indexes_wrong_len=[]
    for i,el in enumerate(dev_X):
        len_dev.append(len(el))
        if len(el)>12010 or len(el)<11990:
            print(f"ERRORE - {len(el)} - index: {i} - SR:{dev_SR[i]}")
            indexes_wrong_len.append(i)
    print(indexes_wrong_len)
    counter_l=collections.Counter(len_dev)
    print(counter_l)
    return
print("DEVEOPMENT")
control_len(dev_X,dev_SR)
print("\nEVALUATION")
control_len(eval_X,eval_SR)
#%%-------------------------------------------REGISTRAZIONI PER PERSONA-------------------------------------------------
counter=collections.Counter(dev_y)
# print(counter)
people=[]
num_audio_people=[]
for key in sorted(counter):
    print( f'{key} {counter[key]}')
    people.append(key)
    num_audio_people.append(counter[key])

inv_diz={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j'}
diz={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9}
plt.bar(people, height=num_audio_people)
plt.xticks(people, [el for el in people])
plt.title("Audio samples length distribution")
plt.xlabel("people")
plt.ylabel("# occurrences")
plt.show()
#%%---------------------------------------------------PADDING-----------------------------------------------------------
dev_X_cleaned=[]
numerical_label_cleaned=[]
ID_clean=[]
for i,el in enumerate(dev_X):
    if len(np.unique(np.array(el)))>1: #STD=0
        if len(el)!=12000:
            if len(el)!=11999:
                print(dev_ID[i], len(el))
            mean_track = np.mean(el)
            diff = 12000 - len(el)
            extension = np.full(diff, mean_track)
            track = np.array(el)
            track_extended = np.concatenate((track, extension), axis=0)
            dev_X_cleaned.append(track_extended)
            numerical_label_cleaned.append(numerical_label[i])
        else:
            dev_X_cleaned.append(el)
            numerical_label_cleaned.append(numerical_label[i])
        ID_clean.append(dev_ID[i])
control_len(dev_X_cleaned,dev_SR) #CONTROLLO

#%%-----------------------------------------------------WINDOW----------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# for idx in range(0, len(dev_X), 10):
ax.plot(dev_X[2], alpha=0.5)

# step=int(dev_SR[0]*0.04)
step=800
for s in range(0, 12000+1, step):
    ax.axvline(s, color="red", linewidth=2)
plt.setp( ax.get_yticklabels(), visible=False)
plt.show()
plt.close()
#%%--------------------------------------------------TIME FEAUTERS-----------------------------------------------------
def RMS(a):
    return np.sqrt(np.mean(np.power(a, 2)))

def compute_signal_statistical_features(x, max_len, step):
    feat_x = []
    for idx in range(step, max_len, step):
        seg_x = x[idx - step:idx]
        if len(seg_x) > 0:
            feat = [np.mean(seg_x),
                    np.std(seg_x),
                    np.min(seg_x),
                    np.max(seg_x),
                    RMS(seg_x)]
        else:
            feat = [0.0, 0.0, 0.0, 0.0, 0.0]
        feat_x.append(np.array(feat))
    return np.nan_to_num(np.hstack(feat_x))

def compute_statistical_features(X, max_len, step):
    features_X = []
    for x in X:
        f_x = compute_signal_statistical_features(x, max_len, step=step)
        features_X.append(f_x)
    return np.vstack(features_X)
step=800
feat_dev_X = compute_statistical_features(dev_X_cleaned, 12000+1, step=step)
print(feat_dev_X.shape)
#%%----------------------------------------------STATISTICHE------------------------------------------------------------
inv_diz={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j'}
fig, ax = plt.subplots(2,5,figsize=(24,18))
for c in np.unique(numerical_label_cleaned):
    # print(c)
    c_feat_dev_X = feat_dev_X[numerical_label_cleaned == c]
    idx = int(c/5)
    idy = int(c%5)
    # print(idx,idy)
    for i in range(0,10):
        ax[idx, idy].fill_between(x= np.arange(len(c_feat_dev_X[i,0::5])),
                        y1= c_feat_dev_X[i,0::5]-c_feat_dev_X[i,1::5], # The mean minus the standard deviation
                        y2=c_feat_dev_X[i,0::5]+c_feat_dev_X[i,1::5], # The mean plus the standard deviation
                       alpha=0.1)
        ax[idx, idy].plot(c_feat_dev_X[i,0::5]) # The mean values
        ax[idx, idy].set_title(f"Person {inv_diz[c].upper()} Dirt")
plt.show()
plt.close()
#%%
def clean_features_local(num_dev_y,feat_dev_X,IDs):
    print("Clean Features...")
    person_feat_dev_x=[] #mean,std,min,max,rms
    for c in range(10):
        c_feat_dev_X = feat_dev_X[num_dev_y == c]
        p_feat=np.mean(c_feat_dev_X,axis=0)
        # print(len(p_feat))
        person_feat_dev_x.append(p_feat)
    # print(len(person_feat_dev_x))
    print("...mean each person...")

    clean_feat_dev_x=[]
    clean_feat_dev_y=[]
    list_pers_del=[]
    final_ID=[]
    index=0
    index_deleted_files=[]
    r=15
    for c in range(10):
        c_feat_dev_X = feat_dev_X[num_dev_y == c]
        p_feat=person_feat_dev_x[c]
        p_ID=IDs[num_dev_y==c]
        person_to_del=0
        for el,id in zip(c_feat_dev_X,p_ID):
            check=0
            for w in range(r):
                if el[w*5] < p_feat[w*5]+p_feat[w*5+1] and el[w*5] > p_feat[w*5]-p_feat[w*5+1]:
                    check+=0
                else:
                    check+=1
            if check==0:
                clean_feat_dev_x.append(el)
                clean_feat_dev_y.append(c)
                final_ID.append(id)
            else:
                person_to_del+=1
                index_deleted_files.append(index)
            index+=1
        list_pers_del.append(person_to_del)
    print("...Feature Cleaned!")
    return index_deleted_files,list_pers_del,clean_feat_dev_x,clean_feat_dev_y,final_ID
track_to_delete,person_to_delete,clean_feat_X,clean_feat_y,ID_keep=clean_features_local(np.array(numerical_label_cleaned),feat_dev_X,\
                                                                                np.array(ID_clean))
#%%
# other_ID=np.load("ID_2.npy")
#%%-------------------------------------------------CLEAN---------------------------------------------------------------
print(f'#_people_deleted: {len(track_to_delete)}, people: {track_to_delete}')
print(person_to_delete)
counter=collections.Counter(clean_feat_y)
print(counter)
X=[]
Y=[]
ID=[]
for i,el in enumerate(dev_X_cleaned):
    if i not in track_to_delete:
        X.append(el)
        Y.append(numerical_label_cleaned[i])
        ID.append(dev_ID[i])
X=np.array(X)
print(X.shape)
#%%--------------------------------------------STATISTICHE PULITE-------------------------------------------------------
clean_feat_X=np.array(clean_feat_X)
clean_feat_y=np.array(clean_feat_y)
inv_diz={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j'}
fig, ax = plt.subplots(2,5,figsize=(24,18))
for c in np.unique(clean_feat_y):
    # print(c)
    c_feat_dev_X = clean_feat_X[clean_feat_y == c]
    idx = int(c/5)
    idy = int(c%5)
    # print(idx,idy)
    for i in range(0,10):
        ax[idx, idy].fill_between(x= np.arange(len(c_feat_dev_X[i,0::5])),
                        y1= c_feat_dev_X[i,0::5]-c_feat_dev_X[i,1::5], # The mean minus the standard deviation
                        y2=c_feat_dev_X[i,0::5]+c_feat_dev_X[i,1::5], # The mean plus the standard deviation
                       alpha=0.1)
        ax[idx, idy].plot(c_feat_dev_X[i,0::5]) # The mean values
        ax[idx, idy].set_title(f"Person {inv_diz[c].upper()} Clean")
plt.show()
plt.close()

#%%----------------------------------------------ANALISI IN FREQUENZA---------------------------------------------------
ptdb_person=[]
x_ptdb=[]
y_ptdb=[]
for c in np.unique(Y):
    person_dev=X[Y == c]
    tmp_mean=[]
    for el in person_dev:
        S = np.abs(librosa.stft(el))
        S_m=np.mean(S.T, axis=0)
        ptdb = librosa.power_to_db(S_m**2)
        # S = np.abs(librosa.stft(el))
        # ptdb = librosa.power_to_db(S ** 2)
        # ptdb = np.mean(ptdb, axis=1)

        tmp_mean.append(ptdb)
        x_ptdb.append(ptdb)
        y_ptdb.append(c)
    tmp_mean=np.array(tmp_mean)
    mean=np.mean(tmp_mean,axis=0)
    ptdb_person.append(mean)
#%%
tmp_save_val=ptdb_person
#%%
ptdb_person=tmp_save_val
ptdb_person=np.array(ptdb_person)
ptdb_person=ptdb_person[:,:85] #205, 9:22
x=np.arange(0,len(ptdb_person[0]),1)
plt.figure(figsize=(16,10))
b_patch = mpatches.Patch(color='blue', label='A')
g_patch = mpatches.Patch(color='green', label='B')
r_patch = mpatches.Patch(color='red', label='C')
y_patch = mpatches.Patch(color='yellow', label='D')
gr_patch = mpatches.Patch(color='grey', label='E')
bk_patch = mpatches.Patch(color='black', label='F')
o_patch = mpatches.Patch(color='orange', label='G')
p_patch = mpatches.Patch(color='purple', label='H')
c_patch = mpatches.Patch(color='cyan', label='I')
s_patch = mpatches.Patch(color='sienna', label='J')
plt.legend(handles=[b_patch,g_patch,r_patch,y_patch,gr_patch,bk_patch,o_patch,p_patch,c_patch,s_patch], fontsize=20)
plt.plot(x,ptdb_person[0],'blue',x,ptdb_person[1],'green',x,ptdb_person[2],'red',x,ptdb_person[3],'yellow',\
         x,ptdb_person[4],'grey',x,ptdb_person[5],'black',x,ptdb_person[6],'orange',x,ptdb_person[7],'purple',\
         x,ptdb_person[8],'cyan',x,ptdb_person[9],'sienna')
plt.title(f'POWER TO DB - SFTF',fontsize=20)
f = [0,2000,4000,8000,10000,12000]
f2=[0,400,800,1200,1600,2000]
f3=[100,130,160,190,220,250]
f4=[100,180,260,340,420,512]
f6=[0,200,400,600,800,1000]
plt.xticks(np.linspace(0,85,6),f6)
plt.xlabel("Frequency",fontsize=20)
plt.ylabel("dB",fontsize=20)
# plt.legend(fontsize="x-large")
x0=np.linspace(0,44,44)
y1=np.full(44,ptdb_person.max())
y2=np.full(44,ptdb_person.min())
# for s in [9,42]:
#     plt.axvline(s, color="green", linewidth=2)
plt.fill_between(x0,y1,y2,color="green",alpha=0.5)
plt.show()
#%%---------------------------------------------------COMPUTE FEATURES--------------------------------------------------
list_person=[]
list_person_all=[]
list_person_to_keep=[]
labels=[]
sample_rate=24000
for c in np.unique(Y):
    person_dev=X[Y == c]
    mean_person_dev=[]
    for el in person_dev:
        S = librosa.feature.melspectrogram(y=el, sr=24000)
        S_m=np.mean(S.T,axis=0)
        mean_val = librosa.power_to_db(S_m)
        mean_val = mean_val[1:]
        mean_person_dev.append(mean_val)
        y=librosa.effects.harmonic(el)
        chroma=np.mean(librosa.feature.chroma_cens(y, sr=sample_rate).T, axis=0)
        mfccs = np.mean(librosa.feature.mfcc(y=el, sr=24000, n_mfcc=40).T, axis=0)
        sc=np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(el)),sr=sample_rate).T,axis=0)
        flatness=librosa.feature.spectral_flatness(y=el)[0]
        list_person_all.append(np.concatenate((mean_val,mfccs,chroma,sc,flatness),axis=0)) #127,40,12,7,24
                                                                                            #127,167,179,186,210
        labels.append(c)

    mean_person_dev = np.array(mean_person_dev)
    general_mean_person = np.mean(mean_person_dev, axis=0)
    list_person.append(general_mean_person)
    print(f"DONE {c}")

tmp_save_val_2=list_person
#%%
mel=[]
l_p=[]
for c in np.unique(Y):
    person_dev=X[Y == c]
    mean_person_dev=[]
    for el in person_dev:
        S = librosa.feature.melspectrogram(y=el, sr=24000)
        S_m=np.mean(S.T,axis=0)
        mean_val = librosa.power_to_db(S_m)
        # S = librosa.feature.melspectrogram(y=el, sr=24000, n_mels=128, fmax=8000)
        # tmp = librosa.power_to_db(S)
        # mean_val = np.mean(tmp,axis=1)

        mean_person_dev.append(mean_val)
        mel.append(mean_val)
    mean_person_dev = np.array(mean_person_dev)
    general_mean_person = np.mean(mean_person_dev, axis=0)
    l_p.append(general_mean_person)
mel=np.array(mel)
#%%
rms=[]
for c in np.unique(Y):
    person_dev=X[Y == c]
    for el in person_dev:
        r = librosa.feature.rms(y=el)[0]
        tmp_sc=np.mean(r,axis=0)
        rms.append(tmp_sc)
rms=np.array(rms)
rms=rms[:,np.newaxis]
print(rms.shape)
#%%
# list_person=tmp_save_val_2
list_person=np.array(l_p)
#%%
# list_person=ptdb_person[:,:]
x=np.arange(0,len(list_person[0]),1)
plt.figure(figsize=(16,10))
b_patch = mpatches.Patch(color='blue', label='A')
g_patch = mpatches.Patch(color='green', label='B')
r_patch = mpatches.Patch(color='red', label='C')
y_patch = mpatches.Patch(color='yellow', label='D')
gr_patch = mpatches.Patch(color='grey', label='E')
bk_patch = mpatches.Patch(color='black', label='F')
o_patch = mpatches.Patch(color='orange', label='G')
p_patch = mpatches.Patch(color='purple', label='H')
c_patch = mpatches.Patch(color='cyan', label='I')
s_patch = mpatches.Patch(color='sienna', label='J')
plt.legend(handles=[b_patch,g_patch,r_patch,y_patch,gr_patch,bk_patch,o_patch,p_patch,c_patch,s_patch],fontsize=20)
plt.plot(x,list_person[0],'blue',x,list_person[1],'green',x,list_person[2],'red',x,list_person[3],'yellow',\
         x,list_person[4],'grey',x,list_person[5],'black',x,list_person[6],'orange',x,list_person[7],'purple',\
         x,list_person[8],'cyan',x,list_person[9],'sienna')
plt.title(f'POWER TO DB - MELSPECTOGRAM', fontsize=20)
f = [0, 512, 1024, 2048, 4096, 8192]
plt.xticks(np.linspace(0, 127, 6), f)
plt.xlabel("Frequency",fontsize=20)
plt.ylabel("dB",fontsize=20)
plt.show()
#%%
for i in [0]:
    y = librosa.effects.harmonic(X[i])
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(16,16))
    img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y, sr=24000),
                                    y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='Chroma_CQT')
    img3 = librosa.display.specshow(librosa.feature.chroma_cens(y, sr=24000),
                                    y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='Chroma_CENS')
    img4 = librosa.display.specshow(librosa.feature.chroma_stft(y, sr=24000),
                                    y_axis='chroma', x_axis='time', ax=ax[2])
    ax[2].set(title='Chroma_STFT_HARMONIC')
    img5 = librosa.display.specshow(librosa.feature.chroma_stft(X[i], sr=24000),
                                    y_axis='chroma', x_axis='time', ax=ax[3])
    ax[3].set(title='Chroma_STFT')
    plt.show()
#%%
list_person_all=np.array(list_person_all)
# np.save("24397_feat",list_person_all)
print(list_person_all.shape)
main_features=list_person_all[:,:].copy()
print(main_features.shape)
#%%
x_ptdb=np.array(x_ptdb)
fn=list_person_all[:,186:]
fn_mean=np.mean(fn,axis=1)
fn_mean=fn_mean[:,np.newaxis]
#%%
main_features=np.concatenate((mel,list_person_all[:,127:186], rms, fn_mean, x_ptdb[:,:44]),axis=1)
print(main_features.shape)
y=labels
#%%%%%%%%%%%%%%%%%%%%%%%%%%
# np.save("feat_d",main_features)
#%%
# i=0
# counter=0
# for el1,el2 in zip(main_features,other_feat):
#     if (el1 == el2).all():
#         print(f"ERROR element {i}")
#         counter+=1
#     i+=1
# print(counter)
#%%---------------------------------------------------tSNE--------------------------------------------------------------
# palette = sns.color_palette("Paired",10)
X_embedded = TSNE(n_components=2,perplexity=20).fit_transform(main_features)
print(X_embedded.shape)
#%%
# palette=sns.color_palette("hls", 10)
palette=sns.color_palette("tab10")
plt.figure(figsize=(30,24))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = labels)
params = {'font.size': 22,
          'legend.fontsize': 8,
          'legend.handlelength': 2}
plt.rcParams.update(params)
patches = []
inv_diz={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j'}
diz={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9}
for c in range(10):
    patch = mpatches.Patch(color=palette[c], label=f'class_{inv_diz[c]}')
    patches.append(patch)
plt.show()
#%% --------------------------------------------------PCA---------------------------------------------------------------
pca=PCA(n_components=3)#50,100,150; 3 for plot
X_pca=pca.fit_transform(X) #np.abs(np.fft.rfft(X))
df=pd.DataFrame(data=X_pca)
df['pca-one'] = X_pca[:,0]
df['pca-two'] = X_pca[:,1]
df['pca-three'] = X_pca[:,2]
df['y'] = labels
#%%
ax = plt.figure(figsize=(10,8)).gca(projection='3d')
ax.scatter(
    xs=df.loc[:,:]["pca-one"],
    ys=df.loc[:,:]["pca-two"],
    zs=df.loc[:,:]["pca-three"],
    c=df.loc[:,:]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.tick_params(axis='both', which='major', labelsize=10)
plt.show()
#%%
plt.figure(figsize=(10,8))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
)
plt.legend(fontsize="small")
plt.show()
#%%
# np.save("main_feat.npy",main_features)
#%%---------------------------------------------------CLASSIFIER--------------------------------------------------------
params = {
    "C": [10, 20, 30, 40, 50, 100],
    "gamma": [0.0001, 0.005, 0.001, 0.05, 0.01],
#     # "class_weight":[None,"balanced"],
#     # "tol":[1e-3, 1e-6]
}
# params = {
#     # "n_estimators":[100,200,500,700,1000],
#     # "n_neighbors":[1,3,5,7,10],
#     # "weights":['distance','uniform'],
#     # "p":[1,2,3]
#     "C":[0.01,0.1,1,10,100],
#     "gamma":[0.001,0.01,0.1,1]
# }
X_train, X_test, y_train, y_test=train_test_split(main_features, y, test_size=0.3, random_state=42, stratify=y)
ss=StandardScaler()
X_tr_ss=ss.fit_transform(X_train)
X_ts_ss=ss.transform(X_test)
rbf=SVC(kernel='rbf') #linear,poly
# rbf=RandomForestClassifier()
# rbf=KNeighborsClassifier()
# rbf=GaussianNB()
search = GridSearchCV(rbf, param_grid=params, n_jobs=-1, cv=3, scoring="f1_macro", verbose=3)
search.fit(X_tr_ss,y_train)
print(f"best_score: {search.best_score_}")
print(f"best_par: {search.best_params_}")
best=search.best_estimator_

clf=best.fit(X_tr_ss, y_train)
y_pred=clf.predict(X_ts_ss)
f=f1_score(y_test, y_pred, average='macro')  # VALUTAZIONE
print(classification_report(y_test, y_pred))
print(f)
#%%
conf_mat = confusion_matrix(y_test, y_pred)
# Plot the result
# label_names = np.arange(10)
label_names=["A","B","C","D","E","F","G","H","I","J"]
conf_mat_df = pd.DataFrame(conf_mat, index = label_names, columns = label_names)
conf_mat_df.index.name = 'Actual'
conf_mat_df.columns.name = 'Predicted'
sns.heatmap(conf_mat_df, annot=True, cmap='YlGnBu',
            annot_kws={"size": 12}, fmt='g', cbar=False)
plt.show()
#%%------------------------------------------------EVALUATION-----------------------------------------------------------
list_person_all_eval=[]
for el in eval_X:
    trk = el
    if len(el) != 12000:
        # print(i)
        mean_track = np.mean(el)
        diff = 12000 - len(el)
        extension = np.full(diff, mean_track)
        track = np.array(el)
        track_extended = np.concatenate((track, extension), axis=0)
        trk=track_extended
    S = librosa.feature.melspectrogram(y=trk, sr=24000)
    S_m = np.mean(S.T, axis=0)
    mean_val = librosa.power_to_db(S_m)
    mean_val = mean_val[1:]
    S = np.abs(librosa.stft(trk))
    S_m = np.mean(S.T, axis=0)
    ptdb = librosa.power_to_db(S_m)
    y = librosa.effects.harmonic(trk)
    chroma = np.mean(librosa.feature.chroma_cens(trk, sr=sample_rate).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=trk, sr=24000, n_mfcc=40).T, axis=0)
    flatness = librosa.feature.spectral_flatness(y=trk)[0]
    sc=np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(trk)),sr=sample_rate).T,axis=0)
    list_person_all_eval.append(np.concatenate((mean_val, mfccs, chroma, sc, flatness, ptdb[8:44]), axis=0))  # 127,40,12,7,24,33
#%%
rms_e=[]
for el in eval_X:
    trk = el
    if len(el) != 12000:
        mean_track = np.mean(el)
        diff = 12000 - len(el)
        extension = np.full(diff, mean_track)
        track = np.array(el)
        track_extended = np.concatenate((track, extension), axis=0)
        trk = track_extended
    r=librosa.feature.rms(y=trk)[0]
    tmp_sc=np.mean(r,axis=0)
    rms_e.append(tmp_sc)
rms_e=np.array(rms_e)
rms_e=rms_e[:,np.newaxis]
print(rms_e.shape)
#%%
mel_e=[]
for el in eval_X:
    trk = el
    if len(el) != 12000:
        # print(i)
        mean_track = np.mean(el)
        diff = 12000 - len(el)
        extension = np.full(diff, mean_track)
        track = np.array(el)
        track_extended = np.concatenate((track, extension), axis=0)
        trk=track_extended
    S = librosa.feature.melspectrogram(y=trk, sr=24000)
    S_m = np.mean(S.T, axis=0)
    mean_val = librosa.power_to_db(S_m)
    mel_e.append(mean_val)
mel_e=np.array(mel_e)
#%%
ptdb_e=[]
for el in eval_X:
    trk = el
    if len(el) != 12000:
        # print(i)
        mean_track = np.mean(el)
        diff = 12000 - len(el)
        extension = np.full(diff, mean_track)
        track = np.array(el)
        track_extended = np.concatenate((track, extension), axis=0)
        trk=track_extended
    S = np.abs(librosa.stft(trk))
    S_m = np.mean(S.T, axis=0)
    ptdb = librosa.power_to_db(S_m**2)
    ptdb_e.append(ptdb)
ptdb_e=np.array(ptdb_e)[:,:44]#44
#%%
X_e_tmp=np.array(list_person_all_eval)
X_e=X_e_tmp[:,:].copy()
print(X_e.shape)
# np.save("eval_24397",X_e_tmp)
#%%
fn_e=X_e_tmp[:,186:210]
fn_mean_e=np.mean(fn_e,axis=1)
fn_mean_e=fn_mean_e[:,np.newaxis]
#%%
# X_e=np.concatenate((X_e_tmp[:,:186], rms_e, fn_mean_e, X_e_tmp[:,210:]),axis=1)
X_e=np.concatenate((mel_e, X_e_tmp[:,127:186], rms_e, fn_mean_e, ptdb_e),axis=1)
print(X_e.shape)
#%%
# np.save("eval_daniela",X_e)
#%%
ss=StandardScaler()
input=ss.fit_transform(main_features)
evaluation=ss.transform(X_e)
clf_e=best.fit(input,labels)
y_e_pred=clf_e.predict(evaluation)
rbf=SVC(kernel='rbf',class_weight="balanced",tol=1e-6)
inv_diz={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j'}
with open("sol_d.csv","w",newline='') as csvfile:
    filewriter=csv.writer(csvfile)
    filewriter.writerow(['Id','Predicted'])
    for i in range(len(y_e_pred)):
        line = [f'{eval_ID[i]}', f'{inv_diz[y_e_pred[i]]}']
        filewriter.writerow(line)
print("CSV DONE")