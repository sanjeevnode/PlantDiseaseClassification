from django.shortcuts import render
from django.core.files.storage import default_storage
from django.templatetags.static import static
import tensorflow as tf
from tensorflow import keras
import pickle,os
import numpy as np
from predict.treatment import treatments,readmores,t
from imagepredict.settings import MEDIA_URL




model= keras.models.load_model("models\densenet_dropout.h5")
categories = pickle.load(open("models\CATEGORIES.pkl","rb"))
print(len(categories),categories,sep="\n")

# Create your views here.
def leaf(request):
    return render(request,"leaf.html")


def test(request):
    if len(os.listdir(MEDIA_URL)) >10:
        for f in os.listdir(MEDIA_URL) :
            if f !="leaf-scanning.gif":
                os.remove(os.path.join(MEDIA_URL,f))
    context={}
    fn = default_storage.url("leaf-scanning.gif")
    context['img_path']=fn
    if request.method=="POST" :
        fileobj = request.FILES['imagepath']
        filename=default_storage.save(fileobj.name,fileobj)
        filepath = default_storage.path(filename)
        f =default_storage.url(filename)

        # testpath="."+filepath
        img = keras.preprocessing.image.load_img(filepath,target_size=(120,120))
        x=keras.preprocessing.image.img_to_array(img)
        x=x/255
        x=x.reshape(1,120,120,3)
        # img_arr = preprocessing(testpath)
        y_predict =model.predict(x)
        label = categories[np.argmax(y_predict)]
        print(label)
        # print(filepath,type(filepath),bd,f,sep="\n")
        context['img_path']=f
        context['label']=label
        context['treatment']=treatments[label]
        context['readmores']=readmores[label]
        return render(request,"test.html",context)
    return render(request,"test.html",context)

    
def tnk(request):
    context={}
    context["data"]=t["b"]
    return render(request,"tnk.html",context) 