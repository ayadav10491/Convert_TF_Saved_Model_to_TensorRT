# Convert_TF_Saved_Model_to_TensorRT
Conversion of TensorFlow saved model to TensorRT

**Conversion to TensorRT from TensorFlow might require following procedures depending the format of saved model:**

## If the model is available in (.h5) format
   
   + In this case we need to convert the model into TensorFlow saved model using "Convert_h5_to_SavedModel.py".
     Once the "tf_saved_model" folder is achieved proceed to next step.
     
## Once the TensorFlow saved model directory is already available or created from previous process.

   + In this case run "tf2trt.py" to achieve "TRT_Model" from TensorFlow saved model directory.
