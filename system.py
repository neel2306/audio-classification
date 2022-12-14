import librosa
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3)
_MODEL_PATH_ = "model.h5"
_SAMPLES_ = 22050

class _Keyword_Spotter_:

    model = tf.keras.models.load_model(_MODEL_PATH_)
    _mappings = [
        "on",
        "down",
        "stop",
        "no",
        "go",
        "left",
        "yes",
        "right",
        "off",
        "up"
    ]


    def prediction(self, MFCCs):
        '''
        Predicts the keywords in the audio file.
        param: file_path(str) = Path to the audio file.
        '''

        #Converting the MFCCs from 2d to 4d.
        MFCCs = MFCCs[np.newaxis,...,np.newaxis]

        #Predicting the keyword.
        predictions = self.model.predict(MFCCs) #Returns an np array of scores of each mapping.
        prediction_index = np.argmax(predictions)
        prediction_keyword = self._mappings[prediction_index]
        return prediction_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        '''
        Extracts MFCCs from an audio file
        param: file_path(str) = Path to the audio file.
        param: num_mfcc(int) = number of coefficients to extract from the audio.
        param: n_fft(int) = Interval for Fast Fourier Transform.
        param: hop_length(int) = Sliding window.
        '''
        #Loading the audio file.
        signal, sample_rate = librosa.load(file_path)

        #Checking for the consistency of the audio file's length.
        if len(signal) >= _SAMPLES_:
            signal = signal[:_SAMPLES_]

            #Extracting the audio's MFCCs.
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        return (MFCCs.T)


if __name__=="__main__":
    spotter_instance = _Keyword_Spotter_()
    
    #Prediction.
    MFCCs = spotter_instance.preprocess('/home/neelabh/Desktop/Audio_classification/dataset/right/0a2b400e_nohash_0.wav')
    keyword = spotter_instance.prediction(MFCCs)
    print(f'The keyword is: {keyword}')