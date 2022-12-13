from pathlib import Path
import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import time


# st.title("this is my app")

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Please Select Your Desire Model",
    ("YoloV4", "YoloV5")
)

def detect_objects(our_image):
    
    # st.subheader("Detected object with box and confidence :")

    net = cv2.dnn.readNet("yolov4-custom_final.weights", "yolov4-custom.cfg")

    classes = []
    with open(r"C:\Users\HP\Desktop\streamlit_data\data .names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    

    # LOAD THE IMAGE
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608,608), (0,0,0), True, crop = False)  

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      

    colors = np.random.uniform(0,255,size=(len(confidences), 3))  
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object 
            label = str.upper((classes[class_ids[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
            items.append(label)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 2)
            print(f'this is i = {i}')
    st.image(img,width = 800)

    # st.set_option('deprecation.showPyplotGlobalUse', False)
def object_main():
        """Car Detection APP"""

        st.title("Car Detection Using YOLOv4")

        choice = st.radio("", ("Show Demo", "Browse an Image"))
        st.write()

        if choice == "Browse an Image":
            st.set_option('deprecation.showfileUploaderEncoding', False)
            image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

            if image_file is not None:
                our_image = Image.open(image_file)  
                detect_objects(our_image)

        elif choice == "Show Demo":
            our_image = Image.open("P1030336.jpg")
            detect_objects(our_image)

if add_selectbox=="YoloV4":

        if __name__ == '__main__':
            object_main()
        
else:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")  # load scratch

    def detect_main():
        """Car Detection APP"""

        st.title("Car Detection Using YOLOv5")

        choice = st.radio("", ("Show Demo", "Browse an Image"))
        st.write()

        if choice == "Browse an Image":
            st.set_option('deprecation.showfileUploaderEncoding', False)
            image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

            if image_file is not None:
                our_image = Image.open(image_file) 
                img  = model(our_image)
                st.image(np.squeeze(img.render())) 
                

        elif choice == "Show Demo":
            our_image = Image.open(r"C:\Users\HP\Desktop\Object_Detection_Streamlit\Car_Detection_YoloV5\P1030336.jpg")
            img = model(our_image)
            st.image(np.squeeze(img.render()))

    if __name__ == '__main__':
        detect_main()

    