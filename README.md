# **Projekt zaliczeniowy – Zastosowania Uczenia Maszynowego**

## **1. Informacje ogólne**
**Nazwa projektu:**  
Porównanie PCA vs CNN vs Transformer na datasecie MNIST

**Autor:**  
Michał Król

**Kierunek, rok i tryb studiów:**  
Informatyka Data Science, 2 rok, II stopnia internetowe.

**Data oddania projektu:**  
15.01.2026

---

## **2. Opis projektu**
Celem projektu jest porównanie trzech podejść do klasyfikacji cyfr odręcznych na zbiorze MNIST:  
- klasyczny model oparty o PCA (klasyfikator podprzestrzeni),  
- sieć konwolucyjna (CNN),  
- transformer w trybie fine-tuningu (DeiT).  
Porównanie obejmuje wyniki z notebooków oraz prostą analizę wpływu doboru liczby składowych PCA.

---

## **3. Dane**
**Źródło danych:**  
MNIST (torchvision.datasets.MNIST)

**Link do danych:**  
`http://yann.lecun.com/exdb/mnist/`

**Opis danych:**  
- liczba próbek: 60 000 train + 10 000 test (w projekcie podział train/val: 55 000 / 5 000),  
- liczba cech / kolumn: 28 x 28 = 784 piksele,  
- format danych: obrazy w skali szarości, tensor `1 x 28 x 28`, wartości znormalizowane do [0, 1],  
- rodzaj etykiet / klas: 10 klas (cyfry 0–9),  
- licencja: zgodnie z informacją na stronie MNIST (Yann LeCun).

**Uwagi dotyczące danych i preprocessingu:**  
- Dane są pobierane automatycznie przez `torchvision` podczas uruchamiania notebooków.  
- Dla modelu transformerowego obrazy są skalowane do 224 x 224 i duplikowane do 3 kanałów.

---

## **4. Cel projektu**
- Klasyfikacja cyfr odręcznych (0–9).  
- Porównanie skuteczności klasycznego podejścia PCA z CNN i transformerem.  
- Ocena wpływu doboru liczby komponentów PCA na accuracy.

---

## **5. Struktura projektu**
Projekt jest podzielony na cztery notebooki:

| Etap | Nazwa pliku | Opis |
|------|--------------|------|
| 1 | `01_eda_mnist.ipynb` | EDA, podgląd danych, wizualizacje przykładowych obrazów |
| 2 | `02_pca_mnist.ipynb` | PCA subspace classifier, analiza accuracy vs k, rekonstrukcje | 
| 3 | `03_cnn_mnist.ipynb` | Trening i ewaluacja CNN na MNIST |
| 4 | `04_transformer.ipynb` | Fine-tuning transformera (DeiT) i ewaluacja |

---

## **6. Modele**
Projekt obejmuje trzy różne podejścia do modelowania danych:

### **6.1 Model klasyczny ML (PCA subspace classifier)**
- Algorytm: klasyfikator podprzestrzeni PCA liczony osobno dla każdej klasy.  
- Parametry: `k=26` (sprawdzono zakres k=1..30 w notebooku).  
- Wyniki / metryki: `test_acc = 0.9580`.

### **6.2 Sieć neuronowa (CNN)**
- Architektura: 2x Conv (1→32, 32→64, kernel 3x3) + MaxPool + 2x FC (128 → 10).  
- Funkcje aktywacji i optymalizator: ReLU, Adam (`lr=1e-3`).  
- Trening: 5 epok, batch size 128.  
- Wyniki: `test_acc = 0.9879`, `test_loss = 0.0386`.

### **6.3 Model transformerowy (fine-tuning)**
- Nazwa modelu: `facebook/deit-tiny-patch16-224`.  
- Biblioteka: HuggingFace Transformers.  
- Zakres dostosowania: fine-tuning całego modelu, wejście 224x224, 3 kanały, AdamW (`lr=1e-5`, `weight_decay=0.05`), early stopping.  
- Wyniki: `test_acc = 0.9890`.

---

## **7. Ewaluacja**
**Użyte metryki:** accuracy (train/val/test w logach Lightning).  

**Porównanie modeli:**

| Model | Metryka główna | Wynik | Uwagi |
|--------|----------------|--------|--------|
| PCA subspace | Accuracy | 0.9580 | k=26 |
| CNN | Accuracy | 0.9879 | 5 epok |
| Transformer (DeiT) | Accuracy | 0.9890 | fine-tuning |

---

## **8. Wnioski i podsumowanie**
- Najlepszy wynik uzyskał transformer (0.9890), ale przewaga nad CNN jest niewielka.  
- CNN zapewnia bardzo wysoki wynik przy niższym koszcie obliczeniowym.  
- PCA daje wyraźnie słabszy wynik, ale jest szybkim baseline i pozwala analizować wpływ liczby komponentów.  
- Główne wyzwania: dobór k w PCA oraz większy koszt obliczeń i konieczność resize do 224x224 w transformerze.  

Możliwe usprawnienia: dłuższy trening, augmentacje danych, tuning hiperparametrów, dodatkowe metryki (np. macierz pomyłek).

---

## **9. Struktura repozytorium**
```
.
├── notebooks/
│   ├── 01_eda_mnist.ipynb
│   ├── 02_pca_mnist.ipynb
│   ├── 03_cnn_mnist.ipynb
│   └── 04_transformer.ipynb
├── src/
│   ├── data/
│   │   └── datamodule.py
│   ├── models/
│   │   └── cnn.py
│   └── modules/
│       ├── lit_cnn.py
│       ├── pca_subspace.py
│       └── transformer.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## **10. Technologia i biblioteki**
- Python 3.12  
- PyTorch, PyTorch Lightning  
- torchvision  
- transformers, torchmetrics  
- NumPy, Matplotlib  

---

## **11. Uruchomienie (uv)**
**Instalacja uv:**  
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Instalacja zależności:**  
```
uv sync
```
---

## **12. Licencja projektu**
Licencja projektu: MIT.  
Źródło danych: MNIST (Yann LeCun) – CC BY-SA 3.0.
