import os
import json
import librosa

_DATASET_PATH_ = "//home//neelabh//Desktop//Audio_classification//dataset"
_JSON_PATH_ = "data.json"
_SAMPLES_ = 22050 #for 1 second of audio

def preprocess_data(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    '''
    This function extracts MFCCs from the audio dataset and saves them into a json file along with some other useful data.

    param: dataset_path(str) = Path to the dataset that will be transformed.
    param: json_path(str) = Path to where the json file will be saved.
    param: num_mfccs(int) = Number of Mel-frequency cepstral coefficients to be extracted.
    param: n_fft(int) = Interval in which Fast fourier transform is to be applied.
    param: hop_length(int) = Sliding window for FFT.
    '''
    
    #Intializing a dictionary which will hold the information.
    data = {
        "words" : [],
        "labels" : [], #These are mapped to "words".
        "MFCCs" : [],
        "files" : []
    }

    #Lopping through all files to gather information.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath not in dataset_path:

            #Adding the words such as down left etc to the words key.
            word = dirpath.split("/")[-1]
            data['words'].append(word)
            print("\nAdding: '{}'".format(word))

            #Extracting and storing MFCCs.
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                #Loading audio files.
                signal, sample_rate = librosa.load(file_path)

                #Ensuring consistency of the audio files. i,e audio files with 1 second length or more are only considered.
                if len(signal) >= _SAMPLES_:
                    signal = signal[:_SAMPLES_]

                    #Extracting MFCCs.
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)

                    #Storing the data.
                    data['MFCCs'].append(MFCCs.T.tolist()) #Transposing the ndarray and converting it into a list.
                    data['labels'].append(i-1)
                    data['files'].append(file_path)
                    print("\nProcessing on '{}': '{}'".format(file_path, i-1))
        
    
    #Saving data in json format.
    with open(_JSON_PATH_, 'w') as j:
        json.dump(data, j, indent=4)

if __name__ == "__main__":
    preprocess_data(_DATASET_PATH_, _JSON_PATH_)