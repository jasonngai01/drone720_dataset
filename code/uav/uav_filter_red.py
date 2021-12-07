print('load Tensorflow')
import sounddevice as sd
import numpy as np
import scipy
import scipy.signal
from scipy import signal
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

numtaps = 101 #Length of the filter (number of coefficients, i.e. the filter order + 1). numtaps must be odd if a passband includes the Nyquist frequency.
cutoff_400 = 400 #Cutoff frequency of filter (expressed in the same units as fs) OR an array of cutoff frequencies (that is, band edges). In the latter case, the frequencies in cutoff should be positive and monotonically increasing between 0 and fs/2. The values 0 and fs/2 must not be included in cutoff.
cutoff_1000 = 1000

#model_path
model_path_u = 'model/uav20211203.tflite' #binary classification model

model_path_um = 'model/uav20211125m.tflite' #multi classifcation model 

model_path_y = 'model/yamnet.tflite' # yamnet model


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

# bin class 400Hz 
interpreter_b4 = Interpreter(model_path_u)
input_details_b4 = interpreter_b4.get_input_details()
waveform_input_index_b4 = input_details_b4[0]['index']
output_details_b4 = interpreter_b4.get_output_details()
scores_output_index_b4 = output_details_b4[1]['index']

# bin class 1000Hz 
interpreter_b10 = Interpreter(model_path_u)
input_details_b10 = interpreter_b10.get_input_details()
waveform_input_index_b10 = input_details_b10[0]['index']
output_details_b10 = interpreter_b10.get_output_details()
scores_output_index_b10 = output_details_b10[1]['index']


print('load UAVNET_mutli')
interpreter_um = Interpreter(model_path_um)
input_details_um = interpreter_um.get_input_details()
waveform_input_index_um = input_details_um[0]['index']
output_details_um = interpreter_um.get_output_details()
scores_output_index_um = output_details_um[0]['index']

# mul class 400Hz 
interpreter_um_b4 = Interpreter(model_path_um)
input_details_um_b4 = interpreter_um_b4.get_input_details()
waveform_input_index_um_b4 = input_details_um_b4[0]['index']
output_details_um_b4 = interpreter_um_b4.get_output_details()
scores_output_index_um_b4 = output_details_um[0]['index']

# mul class 1000Hz 
interpreter_um_b10 = Interpreter(model_path_um)
input_details_um_b10 = interpreter_um_b10.get_input_details()
waveform_input_index_um_b10 = input_details_um_b10[0]['index']
output_details_um_b10 = interpreter_um_b10.get_output_details()
scores_output_index_um_b10 = output_details_um_b10[0]['index']

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
    sf.write('rec_orginal.wav' ,rec, resample_rate)
    
    #filter 400Hz
    filter_rec_400 = signal.firwin(numtaps=numtaps, cutoff=cutoff_400 , fs=resample_rate, pass_zero=False)
    rec_400 = signal.lfilter(filter_rec_400, [1.0], rec)
    rec_400 = tf.cast(rec_400, dtype = tf.float32)
    sf.write('rec_400.wav' ,rec_400, resample_rate)
    rec_400 = tf.expand_dims(rec_400 , 0)
    
    #filter 1000Hz
    filter_rec_1000 = signal.firwin(numtaps=numtaps, cutoff=cutoff_1000 , fs=resample_rate, pass_zero=False)
    rec_1000 = signal.lfilter(filter_rec_1000, [1.0], rec)
    rec_1000 = tf.cast(rec_1000, dtype = tf.float32)
    sf.write('rec_1000.wav' ,rec_1000, resample_rate)
    rec_1000 = tf.expand_dims(rec_1000 , 0)
    
    #save the rec
    
    rec_original = tf.expand_dims(rec , 0) # change the waveform.shape from (15600,) to (1,15600) to fit the model maker #rec is to fit the yamnet model and the rec1 is to fit the model maker uav model
    
    
    
    #############################################################################################################################
    
    # Make prediction from bin model and original rec
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, rec_original)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores = interpreter.get_tensor(scores_output_index)
    uav_prediction = scores.argmax()
    class_scores = tf.reduce_mean(scores, axis=0) 
    top_score = scores[0][uav_prediction] * 100 #percentage of class
    
    #############################################################################################################################
    
    # Make prediction from bin model and Filter 400Hz rec
    
    interpreter_b4.allocate_tensors()
    interpreter_b4.set_tensor(waveform_input_index_b4, rec_400)
    
    interpreter_b4.invoke()
    
    output_data_b4 = interpreter_b4.get_tensor(output_details[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_b4 = interpreter_b4.get_tensor(scores_output_index)
    uav_prediction_b4 = scores_b4.argmax()
    class_scores_b4 = tf.reduce_mean(scores_b4, axis=0) 
    top_score_b4 = scores_b4[0][uav_prediction_b4] * 100 #percentage of class
    
    #############################################################################################################################
    
    # Make prediction from bin model and Filter 1000Hz rec
    
    interpreter_b10.allocate_tensors()
    interpreter_b10.set_tensor(waveform_input_index_b10, rec_1000)
    
    interpreter_b10.invoke()
    
    output_data_b10 = interpreter_b10.get_tensor(output_details[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_b10 = interpreter_b10.get_tensor(scores_output_index)
    uav_prediction_b10 = scores_b10.argmax()
    class_scores_b10 = tf.reduce_mean(scores_b10, axis=0) 
    top_score_b10 = scores_b10[0][uav_prediction_b10] * 100 #percentage of class
    
    #############################################################################################################################
    
    # Make prediction from model
    
    interpreter_um.allocate_tensors()
    interpreter_um.set_tensor(waveform_input_index_um, rec_original)
    
    interpreter_um.invoke()
    
    output_data_um = interpreter_um.get_tensor(output_details_um[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_um = interpreter_um.get_tensor(scores_output_index_um)
    uav_prediction_um = scores_um.argmax()
    class_scores_um = tf.reduce_mean(scores_um, axis=0) 
    top_score_um = scores_um[0][uav_prediction_um] * 100 #percentage of class
    
    #############################################################################################################################
    
    # Make prediction from multi model Filter 400Hz
    
    interpreter_um_b4.allocate_tensors()
    interpreter_um_b4.set_tensor(waveform_input_index_um_b4, rec_400)
    
    interpreter_um_b4.invoke()
    
    output_data_um_b4 = interpreter_um_b4.get_tensor(output_details_um_b4[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_um_b4 = interpreter_um_b4.get_tensor(scores_output_index_um_b4)
    uav_prediction_um_b4 = scores_um_b4.argmax()
    class_scores_um_b4 = tf.reduce_mean(scores_um_b4, axis=0) 
    top_score_um_b4 = scores_um_b4[0][uav_prediction_um_b4] * 100 #percentage of class
    
    #############################################################################################################################
    
    # Make prediction from multi model Filter 400Hz
    
    interpreter_um_b10.allocate_tensors()
    interpreter_um_b10.set_tensor(waveform_input_index_um_b10, rec_1000)
    
    interpreter_um_b10.invoke()
    
    output_data_um_b10 = interpreter_um_b10.get_tensor(output_details_um_b10[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_um_b10 = interpreter_um_b10.get_tensor(scores_output_index_um_b10)
    uav_prediction_um_b10 = scores_um_b10.argmax()
    class_scores_um_b10 = tf.reduce_mean(scores_um_b10, axis=0) 
    top_score_um_b10 = scores_um_b10[0][uav_prediction_um_b10] * 100 #percentage of class
    
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
    
    # Report the highest-scoring classes and their scores.
    top3_i = np.argsort(prediction)[::-1][:3]
    inferred_class_1 = yamnet_classes[top3_i] 
    
    list1 = top3_i # yamnet model top 3 result
    #list2 = [329, 121, 131, 406, 32, 278, 330, 331, 332, 333, 334, 338, 340, 341, 398, 490, 514, 515]
    list2 = [329, 121, 131, 406, 32, 278, 330, 331, 332, 333, 334, 338, 340, 341, 398, 490, 298] #no white and pink noise
 
    a = [x for x in list1 if x in list2]
    
    ###############################################################################################################################
    
    # the format of result reporting
    # since the binary model is sensitive, uav detected in binary model show yellow colour.
    # multi model is relatively stable. uav output  1 2 3 will be shown in red colour.
    #yamnet model result such as aircraft 329, insert 121, whale 131, mechanical fan 406, Humming 32, Rustling leaves 278,Aircraft engine 330,
    #Jet engine 331, Propeller, airscrew 332, Helicopter 333, Fixed-wing aircraft 334, 
    #Light engine (high frequency) 338, Lawn mower340, Chainsaw 341, Mechanisms 398, Hum 490, White noise 514, pink noise 515
    
    
    now = datetime.datetime.now()
    
    with open('result.txt','w')as f:
        print('----------------------------------------------------------- ')
        print(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write(now.strftime("%a %d-%m-%Y  %H:%M:%S"))
        f.write('\n')
        f.close()
        
    print('')
    print('Bin UAV Classification:')
    
    r = "0"
    if (uav_prediction == 0):
        r = "No"
    else :
        r = " "
        
    r_b4 = "0"
    if (uav_prediction_b4 == 0):
        r_b4 = "No"
    else :
        r_b4 = " "
        
    r_b10 = "0"
    if (uav_prediction_b10 == 0):
        r_b10 = "No"
    else :
        r_b10 = " "
    
    print('    Original        Audio: {:.3f} %      {} UAV detected'.format(top_score, r)) #top_score is the precentage of class detected
    print('    400Hz Filtered  Audio: {:.3f} %      {} UAV detected'.format(top_score_b4, r_b4)) #top_score is the precentage of class detected
    print('    1000Hz Filtered Audio: {:.3f} %      {} UAV detected'.format(top_score_b10, r_b10)) #top_score is the precentage of class detected
    
    with open('result.txt','a')as f:
            f.write('    Original        Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score, uav_prediction))
            f.write('    400Hz Filtered  Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score_b4, uav_prediction_b4))
            f.write('    1000Hz Filtered Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score_b10, uav_prediction_b10))
            f.write('\n')
            f.close()
    
    #if ((uav_prediction == 1 or uav_prediction_b4 == 1 or uav_prediction_b10 == 1) and (top_score > 75 or top_score_b4 > 75 or top_score_b10 > 75) and (uav_prediction_um != 0 or uav_prediction_um_b4 != 0 or uav_prediction_um_b10 != 0)) : # show 0 or 1 , uav detected or not
        
            
        #with open('result_ora.txt','w')as f_ora:
            #f_ora.write('1')
            #f_ora.write('\n')
            #f_ora.close()        
                
    #else:         
                  
        #with open('result_ora.txt','w')as f_ora:
            #f_ora.write('0')
            #f_ora.write('\n')
            #f_ora.close()
        
    #print('{:.3f} % {:12s} detected'.format(top_score_y,inferred_class)) # show the precentage of yamnet class
    
    print('')
    print('Multi UAV Classification:')
    
    r_um = "0"
    if (uav_prediction_um == 0):
        r_um = "No"
    elif (uav_prediction_um == 1):
        r_um = "A far"
    elif (uav_prediction_um == 2):
        r_um = "A Hovering"
    else:
        r_um = "A Moving"
        
    r_um_b4 = "0"
    if (uav_prediction_um_b4 == 0):
        r_um_b4 = "No"
    elif (uav_prediction_um_b4 == 1):
        r_um_b4 = "A far"
    elif (uav_prediction_um_b4 == 2):
        r_um_b4 = "A Hovering"
    else:
        r_um_b4 = "A Moving"
        
    r_um_b10 = "0"
    if (uav_prediction_um_b10 == 0):
        r_um_b10 = "No"
    elif (uav_prediction_um_b10 == 1):
        r_um_b10 = "A far"
    elif (uav_prediction_um_b10 == 2):
        r_um_b10 = "A Hovering"
    else:
        r_um_b10 = "A Moving"
    
        
    print('    Original        Audio: {:.3f} %      {} UAV detected'.format(top_score_um, r_um)) #top_score is the precentage of class detected
    print('    400Hz Filtered  Audio: {:.3f} %      {} UAV detected'.format(top_score_um_b4, r_um_b4)) #top_score is the precentage of class detected
    print('    1000Hz Filtered Audio: {:.3f} %      {} UAV detected'.format(top_score_um_b10, r_um_b10)) #top_score is the precentage of class detected
    
    if ((uav_prediction_b4 != 0 or uav_prediction_b10 != 0) ):
    #if ((uav_prediction != 0 or uav_prediction_b4 != 0 or uav_prediction_b10 != 0) and (top_score > 60 or top_score_b4 >60 or top_score_b10 > 60)): 
    #if ((uav_prediction_um != 0 or uav_prediction_um_b4 != 0 or uav_prediction_um_b10 != 0)and (uav_prediction == 1 or uav_prediction_b4 == 1 or uav_prediction_b10 == 1) and (top_score > 70 or top_score_b4 >70 or top_score_b10 > 70) and len(a) > 0): # show 0 or 1 , uav detected or not
        with open('result_red.txt','w')as f_red:
            f_red.write('1')
            f_red.write('\n')
            f_red.close()
            
    else :
        with open('result_red.txt','w')as f_red:
            f_red.write('0')
            f_red.write('\n')
            f_red.close()
    
    with open('result.txt','a')as f:
            f.write('    Original        Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score_um, uav_prediction_um))
            f.write('    400Hz Filtered  Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score_um_b4, uav_prediction_um_b4))
            f.write('    1000Hz Filtered Audio: {:.3f} %      {:.0f} UAV detected'.format(top_score_um_b10, uav_prediction_um_b10))
            f.write('\n')
            f.close()
      
    print('')
    print('Yamnet Classification:')
    #print(uav_prediction)
    #print(uav_prediction_um)
    #print(prediction_y)
    #print('{:.3f}% '.format(top_score_um))
    
    n = 0 
    for i in top3_i:
        print('    {:.3f} %      {:12s} detected'.format(prediction[i]*100,inferred_class_1[n]))
        with open('result.txt','a')as f:
            f.write('{:.3f} %      {:12s} detected'.format(prediction[i]*100,inferred_class_1[n]))
            f.write('\n')
            f.close()
        n = n + 1
    
   
    # yamnet model result comparison
    #yamnet model result such as aircraft 329, insert 121, whale 131, mechanical fan 406, Humming 32, Rustling leaves 278,Aircraft engine 330,
    #Jet engine 331, Propeller, airscrew 332, Helicopter 333, Fixed-wing aircraft 334, 
    #Light engine (high frequency) 338, Lawn mower340, Chainsaw 341, Mechanisms 398, Hum 490, White noise 514, pink noise 515, motorboat 298
    
    if top_score_y > 15 :
        
        #print(top3_i)
        #print(top_score_y)
        #print(a)
        #print(len(a))
    
        if len(a) > 0 :
            with open('result_yel.txt','w')as f_yel:
                f_yel.write('1')
                f_yel.write('\n')
                f_yel.close()
    
        else :
            with open('result_yel.txt','w')as f_yel:
                f_yel.write('0')
                f_yel.write('\n')
                f_yel.close()
                
    else :
        with open('result_yel.txt','w')as f_yel:
                f_yel.write('0')
                f_yel.write('\n')
                f_yel.close()
        
    
    #if top3_i
    print('----------------------------------------------------------- ')
    print('')
    
    

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass