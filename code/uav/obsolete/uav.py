"""
Connect a resistor and LED to board pin 8 and run this script.
Whenever you say "stop", the LED should flash briefly
"""
print('load Tensorflow')
import sounddevice as sd
import numpy as np
import scipy
import scipy.signal
import timeit

import soundfile as sf

import datetime

#import RPi.GPIO as GPIO
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

model_path = 'model/uav20210913.tflite'
model_path_0 = 'model/yamnet.tflite'
# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2) #32000 >> 2sec 


# Load model (interpreter)

print('load UAVNET')
interpreter = Interpreter(model_path)
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']

print('load YAMNET')
interpreter0 = Interpreter(model_path_0)
input_details0 = interpreter0.get_input_details()
waveform_input_index0 = input_details0[0]['index']
output_details0 = interpreter0.get_output_details()
scores_output_index0 = output_details0[0]['index']

yamnet_classes = yamnet_model.class_names('model/yamnet_class_map.csv')


print(input_details)

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

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    #GPIO.output(led_pin, GPIO.LOW)

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
    #librosa.output.write_wav('audio/rec.wav',rec,resample_rate)
    sf.write('rec.wav' ,rec, resample_rate)


    # Make prediction from model
    interpreter.resize_tensor_input(waveform_input_index, [rec.size], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, rec)
    
    interpreter.invoke()
    
    scores = interpreter.get_tensor(scores_output_index)
    uav_prediction = scores.argmax()
    class_scores = tf.reduce_mean(scores, axis=0) * 100
    #top_score = class_scores[uav_prediction] * 100 #percentage of class
    
    
    # model prediction using yamnet tflite
    interpreter0.resize_tensor_input(waveform_input_index, [len(rec)], strict=True)
    interpreter0.allocate_tensors()
    interpreter0.set_tensor(waveform_input_index0, rec)
    
    interpreter0.invoke()
    
    scores_0 = (interpreter0.get_tensor(scores_output_index0))
    class_scores_0 = tf.reduce_mean(scores_0, axis=0)
    prediction_0 = scores_0.mean(axis=0).argmax()
    inferred_class = yamnet_classes[prediction_0] # class map with csv
    top_score_0 = class_scores_0[prediction_0] * 100 #percentage of class
    
    #prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    #top3_i = np.argsort(prediction)[::-1][:3]
    
        
    now = datetime.datetime.now()
    
    
    with open('result.txt','w')as f:
        
        print(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write('\n')
        f.close()
    
    if uav_prediction == 1:
        print('UAV detected')
        with open('result.txt','a')as f:
            f.write('UAV detected')
            f.write('\n')
            f.close()
            
        with open('uav_result.txt','w')as f_uav:
            f_uav.write('1')
            f_uav.write('\n')
            f_uav.close()
                
    else:         
        print('NO UAV detected ')
        with open('result.txt','a')as f:
            f.write('No UAV detected')
            f.write('\n')
            f.close()
            
        with open('uav_result.txt','w')as f_uav:
            f_uav.write('0')
            f_uav.write('\n')
            f_uav.close()
            
    print(class_scores)
    print(inferred_class)    
    print(top_score_0)
    with open('result.txt','a')as f:
        f.write(str(class_scores))
        f.write('\n')
        f.write(str(inferred_class))
        f.write('\n')
        f.write(str(top_score_0))
        f.write('\n')
        f.close()
    
    #print(file_name, ':\n' + '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])for i in top3_i))
    
    print('')
    
    


# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
