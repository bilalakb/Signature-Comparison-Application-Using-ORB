import cv2  # OpenCV kütüphanesini yükler
import numpy as np  # NumPy kütüphanesini yükler, görüntü verisi ve matris işlemleri için kullanılır
from matplotlib import pyplot as plt  # Matplotlib ile görselleştirme yapmamıza olanak sağlar
import tkinter as tk  # Tkinter ile GUI (grafiksel kullanıcı arayüzü) oluşturmak için kullanılır
from tkinter import filedialog, messagebox  # Tkinter'in dosya seçme ve hata mesajlarını gösterme araçları
from PIL import Image, ImageTk  # Tkinter ile görüntü işlemede yardımcı olur

# İmzaları yükleyen ve gri tonlamaya çeviren fonksiyon
def load_image(file_path):
    img = cv2.imread(file_path)  # Dosya yolundan görüntüyü yükler
    if img is None:  # Eğer dosya yolu geçersizse hata fırlatır
        raise ValueError(f"Dosya bulunamadı: {file_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya çevirir
    return gray_img  # Gri tonlamaya çevrilmiş görüntüyü döndürür

# ORB algoritması ile anahtar noktaları (keypoints) ve tanımlayıcıları (descriptors) bulma
def extract_features(img):
    orb = cv2.ORB_create()  # ORB algoritması oluşturulur
    keypoints, descriptors = orb.detectAndCompute(img, None)  # Anahtar noktalar ve tanımlayıcılar tespit edilir
    return keypoints, descriptors  # Anahtar noktalar ve tanımlayıcılar döndürülür

# İki imza arasındaki benzerliği hesaplayan fonksiyon
def compare_signatures(img1, img2):
    kp1, des1 = extract_features(img1)  # İlk imzanın anahtar noktaları ve tanımlayıcıları çıkarılır
    kp2, des2 = extract_features(img2)  # İkinci imzanın anahtar noktaları ve tanımlayıcıları çıkarılır
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Brute-Force matcher, tanımlayıcıları karşılaştırmak için kullanılır
    matches = bf.match(des1, des2)  # İki imzanın tanımlayıcıları eşleştirilir
    
    matches = sorted(matches, key=lambda x: x.distance)  # Eşleşmeler mesafeye göre sıralanır (mesafe ne kadar küçükse o kadar iyi eşleşme)
    
    total_matches = len(matches)  # Toplam eşleşme sayısı
    good_matches = [m for m in matches if m.distance < 50]  # İyi eşleşmeler (mesafesi 50'den küçük olanlar)
    
    # Benzerlik skoru hesaplanır: iyi eşleşmelerin toplam eşleşmelere oranı
    similarity_score = len(good_matches) / total_matches if total_matches != 0 else 0
    return similarity_score, matches, kp1, kp2  # Benzerlik skoru, eşleşmeler, ilk ve ikinci imzanın anahtar noktaları döndürülür

# İmzaları karşılaştırma ve sonucu görselleştirme
def compare_and_display_signatures(img1_path, img2_path):
    img1 = load_image(img1_path)  # İlk imza yüklenir ve gri tonlamaya çevrilir
    img2 = load_image(img2_path)  # İkinci imza yüklenir ve gri tonlamaya çevrilir
    
    score, matches, kp1, kp2 = compare_signatures(img1, img2)  # İmzalar karşılaştırılır, benzerlik skoru ve eşleşmeler elde edilir
    
    # İlk 10 eşleşmeyi içeren bir sonuç görüntüsü oluşturulur
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # Görüntü BGR formatından RGB formatına çevrilir (Matplotlib'de doğru gösterim için)
    
    plt.figure(figsize=(10, 5))  # Görüntüyü göstermek için bir grafik figürü oluşturulur
    plt.imshow(result_img)  # Sonuç görüntüsü ekrana çizilir
    plt.title(f"Benzerlik Skoru: {score:.4f}")  # Başlıkta benzerlik skoru gösterilir
    plt.show()  # Grafik gösterilir
    
    return score  # Benzerlik skoru döndürülür

# Tkinter arayüzü
class SignatureComparerApp:
    def __init__(self, root):
        self.root = root  # Tkinter ana pencere nesnesi
        self.root.title("İmza Karşılaştırma")  # Pencere başlığı
        self.root.geometry("400x300")  # Pencere boyutu ayarlanır
        self.root.configure(bg='#f0f0f0')  # Arka plan rengi
    
        self.img1_path = ""  # İlk imzanın dosya yolu
        self.img2_path = ""  # İkinci imzanın dosya yolu
        
        # Başlık etiketi
        self.title_label = tk.Label(root, text="İmza Karşılaştırma", font=("Arial", 16, "bold"), bg='#f0f0f0')
        self.title_label.pack(pady=10)  # Başlık etiketi pencere içinde gösterilir
        
        # İmza 1 seçme butonu
        self.btn_select_img1 = tk.Button(root, text="İmza 1 Seç", command=self.select_img1, bg='#4CAF50', fg='white', font=("Arial", 12))
        self.btn_select_img1.pack(pady=10)  # Buton pencereye eklenir
        
        # İmza 2 seçme butonu
        self.btn_select_img2 = tk.Button(root, text="İmza 2 Seç", command=self.select_img2, bg='#4CAF50', fg='white', font=("Arial", 12))
        self.btn_select_img2.pack(pady=10)  # Buton pencereye eklenir
        
        # İmzaları karşılaştır butonu
        self.btn_compare = tk.Button(root, text="İmzaları Karşılaştır", command=self.compare_signatures, bg='#2196F3', fg='white', font=("Arial", 12))
        self.btn_compare.pack(pady=10)  # Buton pencereye eklenir
        
        # Sonuç etiketi (benzerlik skorunu göstermek için)
        self.label_result = tk.Label(root, text="Benzerlik Skoru: -", font=("Arial", 12, "bold"), bg='#f0f0f0')
        self.label_result.pack(pady=10)  # Etiket pencereye eklenir
        
    # İmza 1'i seçen fonksiyon
    def select_img1(self):
        self.img1_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])  # Dosya seçme penceresi açılır
        if self.img1_path:  # Eğer dosya seçilmişse
            self.btn_select_img1.config(text="İmza 1 Seçildi")  # Buton metni güncellenir
    
    # İmza 2'yi seçen fonksiyon
    def select_img2(self):
        self.img2_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])  # Dosya seçme penceresi açılır
        if self.img2_path:  # Eğer dosya seçilmişse
            self.btn_select_img2.config(text="İmza 2 Seçildi")  # Buton metni güncellenir
    
    # İmzaları karşılaştıran fonksiyon
    def compare_signatures(self):
        if not self.img1_path or not self.img2_path:  # Eğer her iki imza da seçilmemişse hata mesajı gösterilir
            messagebox.showerror("Hata", "Lütfen her iki imzayı da seçin.")
            return
        
        score = compare_and_display_signatures(self.img1_path, self.img2_path)  # İmzalar karşılaştırılır ve skor elde edilir
        self.label_result.config(text=f"Benzerlik Skoru: {score:.4f}")  # Sonuç etiketi güncellenir ve skor gösterilir

# Ana Tkinter döngüsü (program çalıştırıldığında GUI'nin gösterilmesi için)
if __name__ == "__main__":
    root = tk.Tk()  # Tkinter ana pencere nesnesi oluşturulur
    app = SignatureComparerApp(root)  # Uygulama sınıfı başlatılır
    root.mainloop()  # Tkinter ana döngüsü başlatılır
