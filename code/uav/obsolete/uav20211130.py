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
model_path_u = 'model/uav20211125.tflite' #binary classification model

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

print('load UAVNET_mutli')
interpreter_um = Interpreter(model_path_um)
input_details_um = interpreter_um.get_input_details()
waveform_input_index_um = input_details_um[0]['index']
output_details_um = interpreter.get_output_details()
scores_output_index_um = output_details_um[1]['index']

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
    
    
    #############################################################################################################################
    
    # Make prediction from model
    
    interpreter_um.allocate_tensors()
    interpreter_um.set_tensor(waveform_input_index, rec_1)
    
    interpreter_um.invoke()
    
    output_data_um = interpreter_um.get_tensor(output_details_um[1]['index'])
    #print(output_data) will should softmax [x, 1-x] which is the percentage of uav
    scores_um = interpreter_um.get_tensor(scores_output_index_um)
    uav_prediction_um = scores_um.argmax()
    class_scores_um = tf.reduce_mean(scores, axis=0) 
    top_score_um = scores_um[0][uav_prediction_um] * 100 #percentage of class
    
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
    if uav_prediction == 1 and top_score > 75 and uav_prediction_um != 0 : # show 0 or 1 , uav detected or not
        print('    {:.3f} %      UAV detected'.format(top_score)) #top_score is the precentage of class detected
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %      UAV detected in Bin-Classication Model'.format(top_score))
            f.write('\n')
            f.close()
            
        with open('result_ora.txt','w')as f_ora:
            f_ora.write('1')
            f_ora.write('\n')
            f_ora.close()
                
    else:         
        print('    {:.3f} %      NO UAV detected '.format(top_score))
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %      NO UAV detected in Bin-Classication Model'.format(top_score))
            f.write('\n')
            f.close()
            
        with open('result_ora.txt','w')as f_ora:
            f_ora.write('0')
            f_ora.write('\n')
            f_ora.close()
        
    #print('{:.3f} % {:12s} detected'.format(top_score_y,inferred_class)) # show the precentage of yamnet class
    
    print('')
    print('Multi UAV Classification:')
    if uav_prediction_um == 0: # show 0 or 1 , uav detected or not
        print('    {:.3f} %       NO UAV detected'.format(top_score_um))
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %       NO UAV detected in Mul-Classication Model'.format(top_score_um))
            f.write('\n')
            f.close()
            
        with open('result_red.txt','w')as f_red:
            f_red.write('0')
            f_red.write('\n')
            f_red.close()
        
    elif uav_prediction_um == 1 and uav_prediction == 1 and top_score > 70 and len(a) > 0: # show 0 or 1 , uav detected or not
        print('    {:.3f} %       UAV from far detected'.format(top_score_um))
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %       UAV from far detected'.format(top_score_um))
            f.write('\n')
            f.close()
        
        with open('result_red.txt','w')as f_red:
            f_red.write('1')
            f_red.write('\n')
            f_red.close()
        
    elif uav_prediction_um == 2 and top_score_um > 65: # show 0 or 1 , uav detected or not
        print('{:.3f} %       Hovering UAV detected'.format(top_score_um))
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %       Hovering UAV detected'.format(top_score_um))
            f.write('\n')
            f.close()
            
        with open('result_red.txt','w')as f_red:
            f_red.write('1')
            f_red.write('\n')
            f_red.close()
    
    elif uav_prediction_um == 3 and top_score_um > 65: # show 0 or 1 , uav detected or not
        print('{:.3f} %       Moving UAV detected'.format(top_score_um))
        
        with open('result.txt','a')as f:
            f.write('{:.3f} %       Hovering UAV detected'.format(top_score_um))
            f.write('\n')
            f.close()
        
        with open('result_red.txt','w')as f_red:
            f_red.write('1')
            f_red.write('\n')
            f_red.close()
            
    else :
        print('    {:.3f} %       NO UAV detected'.format(top_score_um))
        with open('result.txt','a')as f:
            f.write('{:.3f} %       NO UAV detected'.format(top_score_um))
            f.write('\n')
            f.close()
            
        with open('result_red.txt','w')as f_red:
            f_red.write('0')
            f_red.write('\n')
            f_red.close()
    
     
    
    
    print('')
    print('Yamnet Classification:')
    #print(uav_prediction_um)
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