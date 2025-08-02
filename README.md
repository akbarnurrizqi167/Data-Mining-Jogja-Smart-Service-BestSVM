# 📊 Analisis Sentimen Jogja Smart Service dengan Best SVM

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**Project ini dikembangkan sebagai bagian dari kompetisi GEMASTIK XVII bidang Data Mining untuk meningkatkan kualitas layanan publik melalui analisis sentimen pengguna aplikasi Jogja Smart Services (JSS).**

## 🎯 **Tujuan Project**

Menganalisis sentimen pengguna terhadap aplikasi Jogja Smart Services menggunakan teknik data mining dan machine learning untuk:
- Mengidentifikasi area kritis yang memerlukan perbaikan
- Memahami persepsi pengguna terhadap layanan JSS
- Memberikan insights untuk peningkatan kualitas layanan publik
- Mencapai performa optimal dalam klasifikasi sentimen menggunakan algoritma SVM

## 📋 **Deskripsi Project**

Project ini mengimplementasikan pipeline lengkap analisis sentimen dari data ulasan Google Play Store aplikasi Jogja Smart Services, meliputi:

### **Tahapan Utama:**
1. **Data Collection**: Web scraping ulasan dari Google Play Store
2. **Data Preprocessing**: Cleaning, normalisasi, dan transformasi data
3. **Feature Engineering**: TF-IDF Vectorization untuk ekstraksi fitur
4. **Model Training**: Implementasi dan optimisasi algoritma SVM
5. **Evaluation**: Analisis performa model dengan berbagai metrik
6. **Visualization**: Pembuatan visualisasi insights hasil analisis

## 🗂️ **Struktur Project**

```
Data-Mining-Jogja-Smart-Service-BestSVM/
├── 📊 Dataset/
│   ├── content jss_final.csv                           # Dataset utama ulasan JSS
│   ├── content hasil resampling (300 per sentimen).csv # Dataset balanced setelah resampling
│   ├── Data Hasil Pelabelan Lexicon.csv               # Hasil pelabelan dengan lexicon-based
│   ├── Data Oke Setelah Di-Stemming (Masing2 300).csv # Data setelah stemming
│   └── Data Oke Setelah Preprocessing.csv             # Data setelah preprocessing lengkap
│
├── 🔧 Preprocessing Resources/
│   ├── kata-kata setelah di-stopword (untuk dikoreksi).csv  # Data intermediate stopword removal
│   ├── list koreksi penulisan (tambahan sendiri).txt       # Kamus koreksi typo custom
│   └── list stopword baru (tambahan sendiri).txt           # Stopwords custom Indonesia
│
├── 🤖 Models/
│   ├── model_svc_best.pkl    # Model SVM terbaik yang telah dilatih
│   └── vectorizer_tfidf.pkl  # TF-IDF Vectorizer yang sudah di-fit
│
├── 📈 Visualisasi/
│   ├── Confusion Matrix.png                               # Confusion matrix hasil evaluasi
│   ├── ROC_curve.png                                     # Kurva ROC dan AUC score
│   ├── Komposisi Score pada Data.png                     # Distribusi rating/score
│   ├── Top Words Sentimen Negatif.png                    # Kata-kata dominan sentimen negatif
│   ├── Top Words Sentimen Positif.png                    # Kata-kata dominan sentimen positif
│   ├── Wordcloud Data Asli (Kotor).png                   # Wordcloud data mentah
│   ├── Wordcloud Data Asli (Kotor) Sentimen Negatif.png  # Wordcloud negatif data mentah
│   ├── Wordcloud Data Asli (Kotor) Sentimen Positif.png  # Wordcloud positif data mentah
│   ├── Wordcloud Data Clean Sentimen Negatif.png         # Wordcloud negatif data bersih
│   └── Wordcloud Data Clean Sentimen Positif.png         # Wordcloud positif data bersih
│
├── 📓 Notebooks/
│   ├── Data_Mining_JSS_BestSVM.ipynb  # Notebook utama lengkap
│   └── Codingan_PakDzaki.ipynb        # Notebook development
│
└── 📖 README.md                       # Dokumentasi project
```

## 🛠️ **Teknologi dan Library**

### **Core Libraries:**
- **Python 3.8+** - Bahasa pemrograman utama
- **Pandas** - Manipulasi dan analisis data
- **NumPy** - Komputasi numerik
- **Scikit-learn** - Machine learning framework
- **NLTK/Sastrawi** - Natural Language Processing Indonesia

### **Data Collection:**
- **google-play-scraper** - Scraping ulasan Google Play Store

### **Visualization:**
- **Matplotlib** - Plotting dan visualisasi
- **Seaborn** - Statistical visualization
- **WordCloud** - Pembuatan word clouds

### **Model & Algorithms:**
- **Support Vector Machine (SVM)** - Algoritma klasifikasi utama
- **TF-IDF Vectorizer** - Feature extraction
- **Grid Search CV** - Hyperparameter tuning

## 🚀 **Cara Menjalankan Project**

### **1. Persiapan Environment**
```bash
# Clone repository
git clone https://github.com/akbarnurrizqi167/Data-Mining-Jogja-Smart-Service-BestSVM.git
cd Data-Mining-Jogja-Smart-Service-BestSVM

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
pip install google-play-scraper nltk sastrawi
```

### **2. Menjalankan Notebook**
```bash
# Jalankan Jupyter Notebook
jupyter notebook Data_Mining_JSS_BestSVM.ipynb
```

### **3. Menjalankan Model Terlatih**
```python
import pickle
import pandas as pd

# Load model dan vectorizer
with open('model_svc_best.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer_tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Prediksi sentimen teks baru
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]
```

## 📊 **Hasil dan Performa Model**

### **Metrik Evaluasi:**
- **Accuracy**: ~XX% (akan diupdate setelah eksekusi lengkap)
- **Precision**: ~XX%
- **Recall**: ~XX%
- **F1-Score**: ~XX%
- **AUC-ROC**: ~XX%

### **Dataset Statistics:**
- **Total Ulasan**: 1,195 ulasan
- **Distribusi Sentimen**: 
  - Positif: 300 ulasan (balanced)
  - Negatif: 300 ulasan (balanced)
  - Netral: ~595 ulasan

### **Key Insights:**
- Kata-kata dominan sentimen **positif**: [akan diupdate]
- Kata-kata dominan sentimen **negatif**: [akan diupdate]
- Area layanan yang paling sering dikomplain: [akan diupdate]

## 🔍 **Metodologi**

### **1. Data Collection & Preprocessing**
- **Web Scraping**: Menggunakan `google-play-scraper` untuk mengumpulkan ulasan
- **Data Cleaning**: Removal noise, URL, mention, hashtag
- **Text Normalization**: Koreksi typo menggunakan kamus custom
- **Stopword Removal**: Menggunakan kombinasi stopwords Indonesia + custom
- **Stemming**: Menggunakan Sastrawi Indonesian Stemmer

### **2. Feature Engineering**
- **TF-IDF Vectorization**: Ekstraksi fitur numerik dari teks
- **N-gram Analysis**: Unigram dan bigram features
- **Feature Selection**: Pemilihan fitur paling informatif

### **3. Model Development**
- **Algorithm**: Support Vector Machine (SVM) dengan kernel RBF
- **Hyperparameter Tuning**: Grid Search CV untuk optimisasi parameter
- **Cross Validation**: 5-fold CV untuk validasi model
- **Class Balancing**: SMOTE/Resampling untuk mengatasi imbalanced data

### **4. Evaluation & Validation**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Confusion Matrix**: Analisis error classification
- **ROC Curve**: Visualisasi trade-off sensitivity vs specificity

## 📈 **Visualisasi Hasil**

Project ini menghasilkan berbagai visualisasi informatif:

1. **Word Clouds**: Representasi visual kata-kata dominan per sentimen
2. **Bar Charts**: Top words positif dan negatif
3. **Confusion Matrix**: Heatmap performa klasifikasi
4. **ROC Curve**: Kurva ROC dengan AUC score
5. **Distribution Plots**: Distribusi rating dan sentimen

## 🏆 **Kontribusi untuk GEMASTIK XVII**

Project ini memberikan kontribusi dalam bentuk:
- **Inovasi Metodologi**: Pipeline preprocessing khusus teks Indonesia informal
- **Kamus Custom**: Pengembangan kamus koreksi typo dan stopwords Bahasa Indonesia
- **Insights Layanan Publik**: Analisis mendalam feedback pengguna JSS
- **Model Reproducible**: Model dan pipeline yang dapat direproduksi dan dikembangkan

## 🔮 **Future Improvements**

- [ ] Implementasi Deep Learning (BERT, LSTM)
- [ ] Real-time sentiment monitoring dashboard
- [ ] Analisis aspek-based sentiment analysis
- [ ] Integrasi dengan lebih banyak sumber data (media sosial)
- [ ] Deployment model sebagai API service

## 👥 **Tim Pengembang**

- **Lead Developer**: Akbar Nur Rizqi
- **Universitas**: [Nama Universitas]
- **Kompetisi**: GEMASTIK XVII - Data Mining

## 📝 **Lisensi**

Project ini dikembangkan untuk keperluan kompetisi GEMASTIK XVII dan penelitian akademik.

## 📞 **Kontak**

- **GitHub**: [@akbarnurrizqi167](https://github.com/akbarnurrizqi167)
- **Email**: [email]

---

### 📌 **Note**: 
Project ini merupakan implementasi lengkap pipeline analisis sentimen dengan fokus pada optimisasi performa SVM untuk klasifikasi sentimen ulasan aplikasi layanan publik Indonesia.

**⭐ Jika project ini membantu, jangan lupa berikan star di repository ini!**

