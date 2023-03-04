import sys
from modelFunctionsLite import *

MODEL_PATH = './worker/MODELS/fruit_model'
IMAGE_PATH = './worker/DataImages/'+str(sys.argv[1])
CLASS_NAMES = get_class_names(MODEL_PATH)
my_model = get_model(MODEL_PATH)
photo_data = photo_data_maker(IMAGE_PATH)
myModel_output = model_output(photo_data, CLASS_NAMES, my_model)
json_data = json_output(CLASS_NAMES,myModel_output)
print(json_data)