from system import _Keyword_Spotter_

_SAMPLES_ = 22050
_SECONDS_ = 1

if __name__=="__main__":
    spotter_instance = _Keyword_Spotter_()
    
    #Recording an audio file.
    audio_file = spotter_instance.record(_SAMPLES_, _SECONDS_)
    #Prediction.
    MFCCs = spotter_instance.preprocess(audio_file)
    keyword = spotter_instance.prediction(MFCCs)
    print(f'The keyword is: {keyword}')
