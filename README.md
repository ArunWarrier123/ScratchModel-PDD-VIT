pip install any packages whenever error comes(patchify,scikitlearn-name might be wrong and any other)
Procedure of working-

#running train.py for vit model creation
keep a folder named Dataset inside have cinnamon plants disease folders
run train.py first check no, of epochs classnames and num_classes. also check dataset path,model path and csv path
on succesfull running of train.py u get model in h5 format you can use for anything later

vit.py no need to run(if u want to print model arch only then)


#running train_vgg16.py for vgg model creation
run train_vgg16.py to use pretrained imagenet vgg16 to train on cinnamon dataset (same dataset path as in train.py)
only if u ever change classes
change this line
model.add(layers.Dense(2, activation='softmax'))  # Assuming 2 classes
adjust epochs and run it it will save vgg model in root directory as output u can use anywhere

#run ensemble.py
make sure num clases,classnames and datasetpath,modelpaths are right
number of images used by both models are printed during execution. make sure they are same
else error
