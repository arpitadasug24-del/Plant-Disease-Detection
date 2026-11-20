"""
Plant Disease Detection — EfficientNetB4
Standalone training/inference scaffold based on the supplied Colab notebook.
Baseline activation is sigmoid to preserve the notebook design; softmax is a separate
corrected experiment and must be retrained. Never transfer the baseline score to it.
"""
import argparse, json, os
import tensorflow as tf
from tensorflow import keras
IMG_SIZE=(160,160); BATCH_SIZE=32

def build_model(num_classes, activation="sigmoid"):
    inputs=keras.Input(shape=(*IMG_SIZE,3))
    base=keras.applications.EfficientNetB4(include_top=False,weights="imagenet",
                                            input_shape=(*IMG_SIZE,3))
    base.trainable=False
    x=base(inputs,training=False)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=keras.layers.Dropout(0.2)(x)
    outputs=keras.layers.Dense(num_classes,activation=activation)(x)
    return keras.Model(inputs,outputs),base

def train(data_dir,out_dir,activation):
    os.makedirs(out_dir,exist_ok=True)
    def ds(split,shuffle):
        return keras.utils.image_dataset_from_directory(
            os.path.join(data_dir,split),image_size=IMG_SIZE,batch_size=BATCH_SIZE,
            shuffle=shuffle)
    train_ds=ds("train",True); val_ds=ds("val",True); test_ds=ds("test",False)
    class_names=train_ds.class_names
    train_ds=train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds=val_ds.prefetch(tf.data.AUTOTUNE); test_ds=test_ds.prefetch(tf.data.AUTOTUNE)
    model,base=build_model(len(class_names),activation)
    def compile_model():
        model.compile(optimizer=keras.optimizers.Adam(),
          loss=keras.losses.SparseCategoricalCrossentropy(),
          metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    compile_model()
    model.fit(train_ds,epochs=6,validation_data=val_ds)
    base.trainable=True
    for layer in base.layers[:100]: layer.trainable=False
    compile_model()
    model.fit(train_ds,epochs=16,initial_epoch=6,validation_data=val_ds)
    loss,acc=model.evaluate(test_ds)
    model.save(os.path.join(out_dir,"plant_disease_efficientnetb4.keras"))
    json.dump(class_names,open(os.path.join(out_dir,"class_names.json"),"w"),indent=2)
    json.dump({"test_loss":float(loss),"test_accuracy":float(acc),
               "activation":activation},open(os.path.join(out_dir,"test_metrics.json"),"w"),indent=2)
    print(f"Test accuracy={acc:.6f}; test loss={loss:.6f}")

def predict(model_path,image_path,class_json):
    model=keras.models.load_model(model_path); names=json.load(open(class_json))
    img=keras.utils.load_img(image_path,target_size=IMG_SIZE)
    arr=keras.utils.img_to_array(img)
    scores=model.predict(arr[None,...],verbose=0)[0]; idx=int(scores.argmax())
    print(json.dumps({"predicted_class":names[idx],"class_index":idx,
                      "top_score":float(scores[idx])},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out",default="plant_disease_output")
    ap.add_argument("--activation",choices=["sigmoid","softmax"],default="sigmoid")
    ap.add_argument("--model"); ap.add_argument("--image")
    ap.add_argument("--classes",default="plant_disease_output/class_names.json")
    a=ap.parse_args()
    if a.model and a.image: predict(a.model,a.image,a.classes)
    elif a.data: train(a.data,a.out,a.activation)
    else: ap.error("Provide --data for training or --model and --image for inference.")
