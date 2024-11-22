import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/Classification Leaf/efficientnetb3-Classification_daun-99.95.h5')

def preprocess_image(img_path):
    img_size = (256, 256)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    # img = img / 255.0  # Normalisasi nilai pixel
    img = np.expand_dims(img, axis=0)
    return img

def get_class_name(predicted_class):
    class_names = [
        'other_leaf',
        'tomato_leaf',
        'undifined'
        ]
    if 0 <= predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Undefined"

def predict_image(img_path):
    image_preproses = preprocess_image(img_path)
    
    prediction = model.predict(image_preproses)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction) * 100 
    
    class_name = get_class_name(predicted_class)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_name} ({confidence_score:.2f}%)")
    plt.axis('off')
    plt.show()

    print(prediction)
    print(f"Hasil Prediksi: {class_name}")
    print(f"Confidence: {confidence_score:.2f}%")
    return class_name, confidence_score


img_paths = [
    'E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/Classification Leaf/Testing file image/Bacterial_spot.JPG',
    'E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/Classification Leaf/Testing file image/bacterial-spot_tomatoes_featured.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/tobacco-mosaic-virus-eggplant-1580133832.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/bercak-daun-septoria.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/bacterial-spot_tomatoes_featured.jpg',
    # 'E:/KULIAHAHAHAH/Studi Independet Bersertifikat (Bangkit)/project/New folder/images.jpg'
]

for img_path in img_paths:
    predict_image(img_path)
