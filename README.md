Machine Learning

# machine_learning_with_supervised_models

📘 Bu repo, gözetimli öğrenme (supervised learning) algoritmalarını öğrenmek amacıyla hazırlanmış kapsamlı bir çalışma koleksiyonudur.              
🧠 Her proje, gerçek veri setleri üzerinde model geliştirme, hiperparametre optimizasyonu ve performans değerlendirme aşamalarını içermektedir.

📘 This repository is a comprehensive collection of studies designed to learn supervised learning algorithms.            
🧠 Each project includes steps for model development, hyperparameter optimization, and performance evaluation using real-world datasets.

📌 **Personal learning project on supervised machine learning algorithms using real-world datasets.**  
📌 **Gerçek dünya verileriyle gözetimli makine öğrenmesi algoritmalarını öğrenme projesidir.**

---

📚 İçindekiler / Table of Contents
🌳 Proje 1: Multiple Linear Regression

🔍 Proje 1 – Problem Tanımı

📘 Proje 1 – Kullanılan Konular

🛠️ Proje 1 – Kullanılan Teknolojiler

🏡 Proje 2: K-Nearest Neighbors (KNN)

🔍 Proje 2 – Problem Tanımı

📘 Proje 2 – Kullanılan Konular

🛠️ Proje 2 – Kullanılan Teknolojiler

💊 Proje 3: Decision Tree Classifier

🔍 Proje 3 – Problem Tanımı

📘 Proje 3 – Kullanılan Konular

🛠️ Proje 3 – Kullanılan Teknolojiler

🎯 Proje 3 – Kısa Özet

---

📌 Proje 1: Multiple Linear Regression

## 🎯 Problem Tanımı / Problem Definition

Bu projede, öğrenci performans verileri üzerinden çoklu doğrusal regresyon (Multiple Linear Regression) uygulanarak bir tahmin modeli geliştirilmiştir.  
In this project, a prediction model has been built by applying Multiple Linear Regression on student performance data.

---

## 🔍 Kullanılan Konular / Topics Covered

- 🔹 Veri temizleme / Data Cleaning  
- 🔹 Eksik verilerle başa çıkma / Handling missing data  
- 🔹 Özellik seçimi / Feature Selection  
- 🔹 Çoklu doğrusal regresyon / Multiple Linear Regression  
- 🔹 Model değerlendirme / Model Evaluation  
- 🔹 Hata metrikleri: MAE, MSE, RMSE, R²

---

## 🧪 Kullanılan Teknolojiler / Technologies Used

- Python  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib
- seaborn 
- Jupyter Notebook

---

📌 Proje 2: K-Nearest Neighbors (KNN)

## 📊 Problem Tanımı / Problem Definition

Bu projede **House Prices** veri seti kullanılarak, evlerin fiyatlarının düşük (0) veya yüksek (1) olarak sınıflandırılması amaçlanmıştır.  
Amacımız, KNN algoritması ile ev fiyatlarını sınıflandırmak ve modelin başarımını değerlendirmektir.

In this project, the **House Prices** dataset is used to classify houses as either low (0) or high (1) priced.  
The goal is to apply the KNN algorithm to classify house prices and evaluate the model performance.

---

## 📚 Kullanılan Konular / Topics Covered
- Veri temizleme / Data Cleaning  
- Eksik verilerle başa çıkma / Handling Missing Data  
- Kategorik verilerin sayısallaştırılması / Categorical Encoding (TargetEncoder)  
- Özelliklerin ölçeklenmesi / Feature Scaling (StandardScaler)  
- KNN algoritması ile sınıflandırma / Classification with KNN  
- Model değerlendirme metrikleri / Model Evaluation Metrics  
  - Accuracy, Precision, Recall, F1-score  
  - Confusion Matrix  

---

## 🛠️ Kullanılan Teknolojiler / Technologies Used
- Python  
- Pandas  
- NumPy  
- scikit-learn  
- category_encoders  
- Matplotlib  
- Seaborn  
- Jupyter Notebook

---

📌 Proje 3: Decision Tree Classifier (İlaç Sınıflandırma)

## 🧩 Problem Tanımı / Problem Definition

Bu projede, ilaç sınıflandırması (drug classification) veri seti kullanılarak, hastaların yaş, tansiyon (BP), kolesterol, sodyum-potasyum oranı (Na_to_K) gibi özelliklerine göre alacakları ilacın (drugA, drugB, drugC, drugX, drugY) tahmin edilmesi amaçlanmıştır.
Amaç, Decision Tree algoritması ile ilaç sınıflarını yüksek doğrulukla tahmin etmek ve modelin performansını değerlendirmektir.

In this project, we use a drug classification dataset to predict which drug a patient should receive based on features such as age, blood pressure (BP), cholesterol, and sodium-potassium ratio (Na_to_K).
The objective is to classify drugs using a Decision Tree algorithm and evaluate the model’s performance.

---

## 📘 Kullanılan Konular / Topics Covered
- Veri keşfi (EDA) / Exploratory Data Analysis
- Veri görselleştirme / Data Visualization
- Veri temizleme ve ön işleme / Data Cleaning & Preprocessing
- Kategorik verilerin dönüştürülmesi / Categorical Encoding
- Eğitim–test ayrımı / Train-Test Split
- Decision Tree modeli kurulumu / Building a Decision Tree Classifier
- Hiperparametre optimizasyonu / Hyperparameter Tuning (GridSearchCV)
- Model değerlendirme / Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Eğitim–test performans karşılaştırması
- Confusion Matrix
- Decision Tree görselleştirme

--- 

## 🛠 Kullanılan Teknolojiler / Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🎯 Projenin Kısa Özeti / Short Summary of the Project

Bu çalışmada, Decision Tree algoritması kullanılarak hastalara uygun ilaç sınıfının tahmini yapılmıştır. Veri ön işleme, model eğitimi, GridSearchCV ile hiperparametre optimizasyonu ve performans değerlendirme adımları uygulanmıştır.
Optimizasyon sonrası model hem eğitim hem test verisinde yüksek doğruluk göstermiştir.

In this study, the Decision Tree algorithm is applied to predict the appropriate drug class for patients. Data preprocessing, model training, hyperparameter tuning with GridSearchCV, and performance evaluation steps were performed.
The optimized model achieved high accuracy on both training and test sets.
