# **Projekt zaliczeniowy – Zastosowania Uczenia Maszynowego**

## **1. Informacje ogólne**
**Nazwa projektu:**  
*(np. Klasyfikacja wydźwięku w tweetach o wojnie)*  

**Autor:**  
Imię i nazwisko  

**Kierunek, rok i tryb studiów:**  
*(np. Informatyka, semestr V)*  

**Data oddania projektu:**  
dd.mm.rrrr  

---

## **2. Opis projektu**
Krótki opis projektu – czego dotyczy, jaki problem rozwiązuje, w jakim kontekście może być użyteczny.  

**Przykład:**  
> Celem projektu jest klasyfikacja emocji w tekstach z mediów społecznościowych. Projekt pozwala na analizę nastrojów użytkowników Twittera dot. wojny z wykorzystaniem metod NLP i uczenia maszynowego.

---

## **3. Dane**
**Źródło danych:**  
*(np. HuggingFace, Kaggle, własny zbiór danych)*  

**Link do danych:**  
`https://...`  

**Opis danych:**  
- liczba próbek:  
- liczba cech / kolumn:  
- format danych:  
- rodzaj etykiet / klas:  
- licencja:  


**Uwaga dotycząca danych:**  
Nie wrzucaj do repozytorium pełnego zbioru danych!  
- Jeśli dane są duże, zamieść **jedynie niewielką próbkę** (np. 100–200 rekordów lub kilka obrazów / plików audio) w folderze `data/sample/`, aby pokazać strukturę danych.  
- Pełen zbiór powinien być pobierany w kodzie (np. z HuggingFace lub Kaggle API) lub dostępny przez link w README.  
- Dzięki temu repozytorium pozostanie czyste i zgodne z zasadami licencyjnymi.

**Uwagi:**  
*(np. dane zostały oczyszczone z duplikatów, usunięto brakujące wartości, itp.)*

---

## **4. Cel projektu**
Sprecyzuj cel biznesowy lub badawczy projektu:  
- Co ma robić model?  
- Jakie pytanie ma odpowiadać lub jaką klasyfikację przeprowadzać?  
- Jakie decyzje lub wnioski można z projektu wyciągnąć?

---

## **5. Struktura projektu**
Projekt składa się z czterech głównych etapów, każdy w osobnym notatniku `.ipynb`:

| Etap | Nazwa pliku | Opis |
|------|--------------|------|
| 1 | `1_EDA.ipynb` | Analiza danych, wizualizacje, wnioski |
| 2 | `2_Preprocessing_Features.ipynb` | Czyszczenie danych, preprocessing, inżynieria cech |
| 3 | `3_Models_Training.ipynb` | Trening modeli klasycznego ML, sieci neuronowej i transformera |
| 4 | `4_Evaluation.ipynb` | Ewaluacja, porównanie modeli, wizualizacje wyników |

---

## **6. Modele**
Projekt obejmuje trzy różne podejścia do modelowania danych:

### **6.1 Model klasyczny ML**
- Algorytm:  
- Krótki opis działania:  
- Wyniki / metryki:  

### **6.2 Sieć neuronowa zbudowana od zera**
- Architektura:  
- Liczba warstw / neuronów:  
- Funkcje aktywacji i optymalizator:  
- Wyniki:  

### **6.3 Model transformerowy (fine-tuning)**
- Nazwa modelu:  
- Zastosowana biblioteka (np. HuggingFace Transformers):  
- Zakres dostosowania:  
- Wyniki:  

---

## **7. Ewaluacja**
**Użyte metryki:**  
*(np. accuracy, precision, recall, F1, ROC-AUC, MSE itp.)*  

**Porównanie modeli:**

| Model | Metryka główna | Wynik | Uwagi |
|--------|----------------|--------|--------|
| Klasyczny ML |  |  |  |
| Sieć neuronowa |  |  |  |
| Transformer |  |  |  |

**Wizualizacje:**  
- Macierz pomyłek  
- Krzywa ROC  
- Learning curve  

---

## **8. Wnioski i podsumowanie**
- Który model okazał się najlepszy i dlaczego?  
- Jakie trudności pojawiły się podczas pracy?  
- Co można by poprawić w przyszłości?  
- Jakie potencjalne zastosowania ma ten projekt?  

---

## **9. Struktura repozytorium**
```
projekt_zum_2025/
│
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 1_EDA.ipynb
│ ├── 2_Preprocessing_Features.ipynb
│ ├── 3_Models_Training.ipynb
│ └── 4_Evaluation.ipynb
├── models/
├── results/
├── README.md
└── requirements.txt
```
## **10. Technologia i biblioteki**
- Python 3.x  
- NumPy, Pandas, Matplotlib, Plotly, Seaborn  
- scikit-learn  
- TensorFlow lub PyTorch  
- HuggingFace Transformers  

---

## **11. Licencja projektu**
Projekt udostępniony na licencji:  
*(np. MIT License, CC-BY 4.0, GPL-3.0)*  

Źródło danych: zgodnie z licencją wskazaną w sekcji **Dane**.

