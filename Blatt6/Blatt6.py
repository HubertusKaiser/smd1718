import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def aufg19():
    print("""Die Anzahl der nächsten Nachbarn, k, 
          hat einen großen Einfluss auf das Ergebnis des
          Algorithmus. Für große k sinkt die Varianz, aber
          gleichzeitig steigt der Bias. Wenn die Attribute 
          sich stark unterscheiden, führt ein zu hohes k zu...""")
    
    def kNN(sample, label_sample, data, k):
        label_data = np.zeros(len(data))
        for i in range(0, len(data)):  # habs nicht ohne schleife geschafft :(
            diff = np.absolute(sample-data[i])  # liefert den Abstand eines Ereignisses zu allen Testereignissen
            nextneighbours = np.argsort(diff)[0:k]   # liefert die indizes der nächsten Nachbarn
            nn_label = label_sample[nextneighbours]  # Label der NN
            sig_count = len([x for x in nn_label if x==1])
            bkg_count = k-sig_count
            if sig_count > bkg_count:  # Zuweisung eines Labels
                label_data[i] = 1
            else:
                label_data[i] = 0
        return label_data
        
    def calc_statvalues(tp, tn, fp, fn):
        reinheit = tp/(tp+fp)
        effizienz = tp/(tp+fn)
        # signifikanz = ?
        return reinheit, effizienz
        
    def countlabels(label_true, label_kNN):
        msk = label_true==label_kNN
        true_labels = label_kNN[msk]
        false_labels = label_kNN[~msk]
        tp = len([x for x in true_labels if x==1])
        tn = len([x for x in true_labels if x==0])
        fp = len([x for x in false_labels if x==1])
        fn = len([x for x in false_labels if x==0])
        print(tp,tn,fp,fn)
        return tp,tn,fp,fn
    
    neutrino_bkg = pd.read_hdf('NeutrinoMC.hdf5', key='Background')
    neutrino_sig = pd.read_hdf('NeutrinoMC.hdf5', key='Signal')
    neutrino_knn_bkg = neutrino_bkg[['NumberOfHits','x','y']]
    neutrino_knn_sig = neutrino_sig[['NumberOfHits','x','y']]
    
    train_bkg, test_bkg = train_test_split(neutrino_knn_bkg,
                                           train_size= 5000,
                                           test_size = 20000)

    train_sig, test_sig = train_test_split(neutrino_knn_sig,
                                           train_size= 5000,
                                           test_size = 10000)     
            
    train_sig['label'] = 1  # wirft eine Warnung weil ein wert statt array, aber scheint zu gehen
    train_bkg['label'] = 0
    test_sig['label'] = 1
    test_bkg['label'] = 0
    train_ges = pd.concat([train_sig, train_bkg])   
    test_ges = pd.concat([test_sig, test_bkg])   
    
    label_numberhits = kNN(train_ges['NumberOfHits'].values,
                           train_ges['label'].values,
                           test_ges['NumberOfHits'].values,
                           10)
  
    tp_nh, tn_nh, fp_nh, fn_nh = countlabels(test_ges['label'].values, label_numberhits)
    reinheit_nh, effizienz_nh = calc_statvalues(tp_nh, tn_nh, fp_nh, fn_nh)

    
    
    label_lognumberhits = kNN(np.log10(train_ges['NumberOfHits'].values),
                           train_ges['label'].values,
                           np.log10(test_ges['NumberOfHits'].values),
                           10)
  
    tp_lnh, tn_lnh, fp_lnh, fn_lnh = countlabels(test_ges['label'].values, label_lognumberhits)
    reinheit_lnh, effizienz_lnh = calc_statvalues(tp_lnh, tn_lnh, fp_lnh, fn_lnh)
            

            
    label_x = kNN(train_ges['x'].values,
                           train_ges['label'].values,
                           test_ges['x'].values,
                           10)
                           
    tp_x, tn_x, fp_x, fn_x = countlabels(test_ges['label'].values, label_x)
    reinheit_x, effizienz_x = calc_statvalues(tp_x, tn_x, fp_x, fn_x)
    
    
    
    label_y = kNN(train_ges['y'].values,
                           train_ges['label'].values,
                           test_ges['y'].values,
                           10)
                           
    tp_y, tn_y, fp_y, fn_y = countlabels(test_ges['label'].values, label_y)
    reinheit_y, effizienz_y = calc_statvalues(tp_y, tn_y, fp_y, fn_y)
    
    print('Reinheiten')
    print(reinheit_nh)
    print(reinheit_lnh)
    print(reinheit_x)
    print(reinheit_y)
    
    print('Effizienzen')
    print(effizienz_nh)
    print(effizienz_lnh)
    print(effizienz_x)
    print(effizienz_y)


    
if __name__ == "__main__":
    rndstate = np.random.RandomState(123)
    print('aufg19')
    aufg19()