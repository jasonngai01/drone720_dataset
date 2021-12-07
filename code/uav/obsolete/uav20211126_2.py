print('load Tensorflow')
import sounddevice as sd
import numpy as np
import scipy
import scipy.signal
import timeit
import datetime
import soundfile as sf

import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import yamnet as yamnet_model

# Parameters
debug_time = 1
debug_acc = 0

rec_duration = 0.975 #unit is sec
window_stride = 1 #unit is sec
sample_rate = 48000
resample_rate = 16000
num_channels = 1

#model_path
model_path_u = 'model/uav20211125.tflite'

model_path_y = 'model/yamnet.tflite'

# create Sliding window to store the tmp data
window = np.zeros(int(rec_duration * resample_rate) * 2) #32000 >> 2sec 

# Load model with interpreter

print('load UAVNET')
interpreter = Interpreter(model_path_u)
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[1]['index']
#0 is first serving layer, 1 is the embedding sequential layer, uav classification belong to embeding sequential layer

print('load YAMNET')
interpreter_y = Interpreter(model_path_y)
input_details_y = interpreter_y.get_input_details()
waveform_input_index_y = input_details_y[0]['index']
output_details_y = interpreter_y.get_output_details()
scores_output_index_y = output_details_y[0]['index']
#print (output_details_y)

yamnet_classes = yamnet_model.class_names('model/yamnet_class_map.csv')

########################################################################################################

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 1 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec
    
    #save the rec
    sf.write('rec.wav' ,rec, resample_rate)
    rec_1 = tf.expand_dims(rec , 0) # change the waveform.shape from (15600,) to (1,15600) to fit the model maker #rec is to fit the yamnet model and the rec1 is to fit the model maker uav model
    
    #############################################################################################################################
    
    # Make prediction from model
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, rec_1)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores = interpreter.get_tensor(scores_output_index)
    uav_prediction = scores.argmax()
    class_scores = tf.reduce_mean(scores, axis=0) 
    top_score = scores[0][uav_prediction] * 100 #percentage of class
    
    ###############################################################################################################################
    
    # model prediction using yamnet tflite
    interpreter_y.resize_tensor_input(waveform_input_index, [len(rec)], strict=True)
    interpreter_y.allocate_tensors()
    interpreter_y.set_tensor(waveform_input_index_y, rec)
    
    interpreter_y.invoke()
    
    scores_y = (interpreter_y.get_tensor(scores_output_index_y))
    class_scores_y = tf.reduce_mean(scores_y, axis=0)
    prediction_y = scores_y.mean(axis=0).argmax()
    inferred_class = yamnet_classes[prediction_y] # class map with csv
    top_score_y = class_scores_y[prediction_y] * 100 #percentage of class
    top_score_y = float(top_score_y)
    prediction = np.mean(scores_y, axis=0)
    
    ###############################################################################################################################
    
    # Report the highest-scoring classes and their scores.
    top3_i = np.argsort(prediction)[::-1][:3]
    inferred_class_1 = yamnet_classes[top3_i] 
        
    now = datetime.datetime.now()
    
    with open('result.txt','w')as f:
        
        print(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write('\n')
        f.close()
        
    print('')
    
    if uav_prediction == 1: # show 0 or 1 , uav detected or not
        print('{:.3f}%      UAV detected'.format(top_score)) #top_score is the precentage of class detected
        with open('result.txt','a')as f:
            f.write('UAV detected')
            f.write('\n')
            f.close()
            
        with open('uav_result.txt','w')as f_uav:
            f_uav.write('1')
            f_uav.write('\n')
            f_uav.close()
                
    else:         
        print('{:.3f}%      NO UAV detected'.format(top_score))
        with open('result.txt','a')as f:
            f.write('No UAV detected')
            f.write('\n')
            f.close()
            
        with open('uav_result.txt','w')as f_uav:
            f_uav.write('0')
            f_uav.write('\n')
            f_uav.close()
        
    #print('{:.3f} % {:12s} detected'.format(top_score_y,inferred_class)) # show the precentage of yamnet class
    
    with open('result.txt','a')as f:
        f.write(str(class_scores))
        f.write('\n')
        f.write(str(inferred_class))
        f.write('\n')
        f.write(str(top_score_y))
        f.write('\n')
        f.close()
        
    print('')
    
    n = 0 
    for i in top3_i:
        print('{:.3f} %      {:12s} detected'.format(prediction[i]*100,inferred_class_1[n]))
        n = n + 1

    print('')
    print('')
    print('')
    

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
