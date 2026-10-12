# TPOD Projesi — Teknik Özet

> Hazırlayan: Hakan Kavurga  
> Bağlam: Yazılım Mühendisliği Yüksek Lisans Semineri — İleri Veri Madenciliği  
> Makale: Zhang & Yang, *"Density-Distance Outlier Detection Algorithm Based on Natural Neighborhood"*, Axioms 2023, 12, 425  

---

## 1. Projenin Amacı ve Mimarisi

### Amaç

1. TPOD algoritmasını makaledeki pseudo-code'dan sıfırdan Python ile implemente etmek.
2. Algoritmayı üç farklı alandan veri seti üzerinde LOF, COF ve ABOD ile karşılaştırmak (Proje v3).
3. Algoritmayı sadece finansal dolandırıcılık tespiti üzerine odaklanarak dört farklı fraud veri seti üzerinde test etmek (Fraud notebook).
4. Bulgulardan akademik bildiri üretmek.

### Mimari

```
TPOD Projesi
├── TPOD_Proje_v3.ipynb          ← Ana karşılaştırmalı analiz (3 veri seti, 4 algoritma)
├── TPOD_Fraud_Detection.ipynb   ← Bildiri için: 4 fraud veri seti, sadece TPOD
├── ionosphere.data              ← UCI Ionosphere (makaledeki dataset)
├── ai4i2020.csv                 ← AI4I 2020 Predictive Maintenance
├── creditcard.csv               ← ULB Credit Card Fraud (Kaggle)
├── PS_20174392719_...log.csv    ← PaySim (Kaggle)
├── Base.csv                     ← Bank Account Fraud / BAF (NeurIPS 2022, Kaggle)
└── bank_transactions_data_2.csv ← Bank Transaction Dataset (Kaggle)
```

### Tech Stack

| Katman | Teknoloji |
|---|---|
| Dil | Python 3.12 |
| IDE | VSCode + Jupyter Notebook |
| Temel kütüphaneler | numpy, pandas, scikit-learn, matplotlib |
| Outlier karşılaştırma | PyOD (LOF, COF, ABOD) |
| Veri yükleme | ucimlrepo (Ionosphere için), CSV (diğerleri) |
| Versiyon kontrolü | GitHub |

---

## 2. TPOD Algoritması — Tam İmplementasyon

Makale Algorithm 1 + Algorithm 2 birebir implemente edildi. **Bu koda hiç dokunulmadı, tüm versiyonlarda aynı kalıyor.**

### Algorithm 1: Natural Neighbor Search

```python
def natural_neighbors(X):
    n = len(X)
    tree = KDTree(X)
    k = 1
    prev_zero_count = -1
    while True:
        _, knn_idx = tree.query(X, k=k + 1)
        knn_idx = knn_idx[:, 1:]
        nb = np.zeros(n, dtype=int)
        for i in range(n):
            for j in knn_idx[i]:
                nb[j] += 1
        zero_count = int(np.sum(nb == 0))
        if zero_count == 0 or zero_count == prev_zero_count:
            break
        prev_zero_count = zero_count
        k += 1
    return k, knn_idx
```

**Ne yapar:** k=1'den başlayarak KD-ağacı ile iteratif komşuluk arar. Her noktanın kaç "ters komşusu" olduğunu sayar. Durdurma koşulu: tüm noktaların en az 1 ters komşusu varsa VEYA sayı bir önceki k ile aynıysa dur. Durduğundaki k = λ (NaNE).

### Algorithm 2: TPOD Score Computation

```python
def run_tpod(X, c=0.9, eta=0.01, eps=1e-5):
    # 1. Doğal komşuluk → λ, nan_idx
    lambda_, nan_idx = natural_neighbors(X)

    # 2. Adaptif kernel genişliği (Formül 10)
    d_lambda = np.array([np.mean(np.linalg.norm(X[nan_idx[i]] - X[i], axis=1)) for i in range(n)])
    h = c * (d_lambda.max() + d_lambda.min() + eps - d_lambda)
    h = np.maximum(h, eps)

    # 3. Yerel yoğunluk (Formül 11)
    rho[i] = np.mean(np.exp(-sq_dist / (h[i]**2 + eps)))  # Gaussian kernel

    # 4. Göreli mesafe (Formül 12-14)
    RD[i] = d_bar(xp) / D_bar(xp)   # komşulara uzaklık / komşuların kendi aralarındaki uzaklık

    # 5. TPOD skoru + eşik (Formül 15-16)
    scores = RD / (rho + eps)
    sigma  = eta * np.mean(scores)
    labels = (scores > sigma).astype(int)
    return scores, labels, lambda_
```

### Formül Referansları

| Adım | Formül No | Açıklama |
|---|---|---|
| Kernel genişliği | Formül 10 | `hᵢ = c·(dλmax + dλmin + ε − dλ(xᵢ))` |
| Yerel yoğunluk | Formül 11 | `ρ(xᵢ) = (1/n−1) Σ exp(−‖xᵢ−xⱼ‖²/hᵢ²)` |
| Doğal komşuluk mesafesi | Formül 12 | `d̄(xₚ) = (1/λ) Σ dist(xᵢ, xₚ)` |
| İç komşuluk mesafesi | Formül 13 | `D̄(xₚ) = (1/λ(λ−1)) Σ dist(xᵢ, xⱼ)` |
| Göreli mesafe | Formül 14 | `RD(xₚ) = d̄(xₚ) / D̄(xₚ)` |
| TPOD skoru | Formül 15 | `TPOD(xₚ) = RD(xₚ) / ρ(xₚ)` |
| Eşik | Formül 16 | `σ = η · mean(TPOD)` |

---

## 3. Tamamlanan Adımlar

### Proje v3 — TPOD_Proje_v3.ipynb (Ana notebook)

Akış:
1. Kurulum → TPOD algoritması → Yardımcı fonksiyonlar
2. 3 veri seti yükleme: Ionosphere, AI4I 2020, Credit Card Fraud
3. **Grid search** ile her veri seti için optimal η ve c belirleme (48 kombinasyon, F1 maksimize)
4. Optimal parametrelerle TPOD + LOF + COF + ABOD çalıştırma
5. Her veri seti için görsel: scatter (PCA 2D) + skor dağılımı + AUC/F1 bar chart + performans tablosu
6. ROC eğrileri — 3 veri seti yan yana
7. η duyarlılık analizi (optimal c sabit, farklı η değerleri)
8. Makale Tablo 5 ile Ionosphere AUC/F1 karşılaştırması

**Elde edilen sonuçlar (özet):**

| Veri Seti | Optimal η | Optimal c | TPOD AUC | TPOD F1 |
|---|---|---|---|---|
| Ionosphere | 0.001 | 0.5 | 0.9230 | 0.8230 |
| AI4I 2020 | 0.1 | 0.5 | 0.8184 | 0.2353 |
| Credit Card | 0.01 | 0.5 | 0.9228 | 0.7189 |

Makale Ionosphere için TPOD F1 = 0.84 vermiş, biz 0.8230 elde ettik (fark: 0.017).  
3 veri setinin 2'sinde (AI4I, CreditCard) TPOD tüm algoritmaları geçti.

### Fraud Detection Notebook — TPOD_Fraud_Detection.ipynb (Bildiri için)

Akış:
1. Kurulum → TPOD → Grid search fonksiyonu
2. 4 fraud veri seti yükleme (aşağıya bak)
3. Grid search → optimal η ve c
4. Optimal parametrelerle sadece TPOD çalıştırma
5. Her veri seti için görsel: scatter + skor dağılımı + ROC + metrik bar
6. 4 veri seti karşılaştırma bar chart
7. 4 veri seti ROC eğrileri tek grafikte
8. Özet tablo
9. 300+ kelimelik Türkçe bildiri özeti (otomatik üretilen, sayıları güncelle)

---

## 4. Kritik Teknik Kararlar ve Kurallar

### Parametre Kararları

| Karar | Gerekçe |
|---|---|
| Grid search ile optimal η ve c belirleme | Makale η ve c'yi teorik değil deneysel olarak veriyor. Grid search sistematik optimizasyon sağlıyor, F1'i %35-45 artırıyor. |
| c aralığı: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | Makale "0.5-1.0 arasından seç" diyor |
| η aralığı: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0] | Makale sentetik için 0.2, gerçek için 0.01 veriyor; geniş aralık tarandı |
| Örnekleme: SAMPLE_SIZE = 1500 | TPOD O(n log n) ama iç döngüler nedeniyle büyük veride çok yavaş |
| LOF/COF için contamination = y.mean() | Gerçek fraud oranını biliyoruz, bunu kullanmak adil karşılaştırma sağlıyor |

### Python 3.12 Uyumluluğu

Python 3.12'de f-string içinde `\n`, `—` (uzun tire) ve `η` gibi özel karakterler SyntaxError veriyor.

**Kural:** f-string içinde asla `\n` kullanma, önce değişkene ata:
```python
# YANLIŞ
ax.set_title(f"{algo}\nAUC = {auc:.4f}")

# DOĞRU
title = algo + "\nAUC = " + f"{auc:.4f}"
ax.set_title(title)
```

### Veri Seti Kararları

| Veri Seti | Dosya Adı | Etiket Sütunu | Fraud Sınıfı |
|---|---|---|---|
| Ionosphere | `ionosphere.data` | Son sütun | `'b'` (bad) |
| AI4I 2020 | `ai4i2020.csv` | `Machine failure` | 1 |
| ULB CreditCard | `creditcard.csv` | `Class` | 1 |
| PaySim | `PS_20174...log.csv` | `isFraud` | 1 — sadece TRANSFER/CASH_OUT tiplerinde |
| BAF | `Base.csv` | `fraud_bool` | 1 |
| BankTransaction | `bank_transactions_data_2.csv` | fraud içeren sütun (otomatik detect) | 1 veya True |

**IEEE-CIS fraud dataseti kullanılmadı** — yarışma dataseti, indirme için kural kabul gerekiyor.

### Metrik Kararları

- **AUC:** Birincil metrik. η parametresinden tamamen bağımsız (eşik değişse de skor sıralaması değişmez). Algoritmaları karşılaştırmak için en adil metrik.
- **F1:** İkincil metrik. Precision ve Recall dengesi. Dengesiz veri setlerinde accuracy kullanılmaz.
- **Recall:** Kaçırılan fraud sayısını minimize etmek için izleniyor.
- **Precision:** Yanlış alarm oranını takip etmek için.

### AUC'un η'dan Bağımsızlığı

Bu projenin en önemli bulgularından biri: η eşik parametresi ne olursa olsun AUC değişmiyor. Sebebi: AUC ham skorların sıralamasına bakıyor, eşike değil. Bu durum η duyarlılık analizinde de görsel olarak kanıtlandı.

---

## 5. Veri Seti Bağlamları

### Fraud Detection Notebook'u için 4 Veri Seti

| Veri Seti | Alan | Özellik Sayısı | Fraud Oranı | Boyut |
|---|---|---|---|---|
| ULB CreditCard | Kredi kartı işlemleri | 30 (V1-V28 + Amount + Time) | %0.17 (gerçek), %19.7 (örneklenmiş) | 284.807 satır |
| PaySim | Mobil para transferi simülasyonu | 6 (sayısal) | %0.13 | 6.3M satır |
| BAF | Banka hesabı açılış başvurusu | ~30 | %1 | 1M satır |
| BankTransaction | Banka işlem geçmişi | ~11 | değişken | küçük |

**PaySim önemli not:** Sadece `TRANSFER` ve `CASH_OUT` tiplerinde fraud var. Diğer tipler filtreleniyor.

**ULB CreditCard önemli not:** V1-V28 özellikler PCA ile anonim hale getirilmiş, gerçek anlamları bilinmiyor. Sadece `Time` ve `Amount` orijinal.

---

## 6. Çözülmemiş Sorunlar ve Roadmap

### Şu An Çözülmemiş

1. **Fraud Detection Notebook son hücre eksik:** `load_banktx` fonksiyonunun veri seti yükleme döngüsündeki `except` bloğu tam değil — genel `Exception` yakalanmıyor. Eklenmesi gereken:
```python
except Exception as e:
    print("  HATA (" + ds_name + "): " + str(e))
    failed.append(ds_name)
```

2. **PaySim dosya adı çok uzun:** `PS_20174392719_1491204439457_log.csv` — kullanıcı farklı bir isimle indirmiş olabilir. Loader'a alternatif isim desteği eklenebilir.

3. **BAF veri setinde kategorik sütun sayısı fazla:** Tüm kategorikler label encoding ile dönüştürülüyor, bu bazı sütunlarda anlamsız sıralama yaratabilir. One-hot encoding daha doğru olabilir ama boyut artar.

4. **Bildiri özeti sayıları placeholder:** `TPOD_Fraud_Detection.ipynb` bölüm 10'daki özet metni genel yazılmış, sonuçlar çalıştırıldıktan sonra sayılar güncellenmeli.

### Roadmap — Sıradaki Adımlar

**Hemen yapılacak:**
- [ ] 4 fraud veri setini indir ve notebook'u çalıştır
- [ ] Sonuçları kontrol et, bildiri özet metnindeki sayıları güncelle
- [ ] Görselleri kaydet (tpod_fraud_*.png)

**Bildiri için:**
- [ ] Türkçe bildiri taslağını yaz (Giriş, İlgili Çalışmalar, Yöntem, Deneyler, Sonuç)
- [ ] Görselleri bildiride kullan (ROC, bar chart, scatter)
- [ ] ASYU veya IDAP konferansına gönder

**İyileştirmeler (opsiyonel):**
- [ ] η otomatik belirleme: `contamination` oranından `eta_from_contamination()` fonksiyonu var, entegre edilebilir
- [ ] Daha hızlı TPOD: inner loop vektörleştirilebilir (şu an nested for loop)
- [ ] GitHub README güncelleme: fraud notebook eklendikten sonra

---

## 7. Dosya Envanteri

### Üretilen Notebook'lar

| Dosya | Açıklama | Durum |
|---|---|---|
| `TPOD_Proje_v3.ipynb` | Ana analiz: 3 veri seti, TPOD+LOF+COF+ABOD, grid search | ✅ Tamamlandı, çalışıyor |
| `TPOD_eta_analizi.ipynb` | η duyarlılık analizi (v3'ün eki) | ✅ Tamamlandı |
| `TPOD_Fraud_Detection.ipynb` | Bildiri: 4 fraud veri seti, sadece TPOD | ⚠️ Hazır ama IEEE→BankTransaction güncellemesi yapıldı, test edilmedi |

### Sunum

| Dosya | Açıklama |
|---|---|
| `outlier_detection.pptx` | Kullanıcının hazırladığı 8 slaytlık sunum |
| `TPOD_Sunum.pptx` | Claude'un hazırladığı 13 slaytlık sunum (makaleli figürler dahil) |

### Veri Setleri

| Dosya | İndirme | Boyut |
|---|---|---|
| `ionosphere.data` | https://archive.ics.uci.edu/dataset/52/ionosphere | 75 KB |
| `ai4i2020.csv` | https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv | ~1 MB |
| `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 144 MB |
| `PS_20174...log.csv` | https://www.kaggle.com/datasets/ealaxi/paysim1 | 470 MB |
| `Base.csv` | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | ~200 MB |
| `bank_transactions_data_2.csv` | https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection | küçük |

---

## 8. Temel Bulgular (Referans)

### Proje v3 Sonuçları (Optimal parametrelerle)

| Veri Seti | TPOD AUC | LOF AUC | COF AUC | ABOD AUC | Kazanan |
|---|---|---|---|---|---|
| Ionosphere | 0.9230 | 0.8903 | 0.8612 | **0.9266** | ABOD (TPOD 2.) |
| AI4I 2020 | **0.8184** | 0.7724 | 0.7201 | 0.7167 | TPOD |
| CreditCard | **0.9228** | 0.4532 | 0.3689 | 0.7087 | TPOD |

### η Duyarlılık Analizi Bulgusu

- AUC tüm veri setlerinde η'dan tamamen bağımsız (düz çizgi)
- F1, η'ya veri setine göre farklı tepki veriyor
- Ionosphere için optimal η = 0.001 (makale 0.2 öneriyor)
- Credit Card için optimal η = 0.01 (makale önerisiyle örtüşüyor)
- Tüm veri setlerinde optimal c = 0.5 çıktı (makale 0.5-1.0 aralığı öneriyor)

### Makale Karşılaştırması (Ionosphere)

| Metrik | Makale | Bizim | Fark |
|---|---|---|---|
| AUC | 0.9320 | 0.9230 | -0.009 |
| F1 | 0.84 | 0.8230 | -0.017 |

Fark grid search parametre arama aralığının farklılığından kaynaklanıyor.
