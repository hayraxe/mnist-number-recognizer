# Määrittelydokumentti: MNIST-numerontunnistus NumPy-tasolla

## 1. Perustiedot
- **Ohjelmointikieli:** Python 
- **Muut kielet (vertaisarviointia varten):** TypeScript, Scala ja R
- **Keskeisimmät paketit:** - **NumPy** (matriisilaskenta ja optimointi)
    - **Matplotlib** (Visualisointi ja testitulosten esittäminen)
- **Opinto-ohjelma:** Taloustieteen kandidaatti (VTK)
- **Dokumentaation kieli:** Suomi
- **Aihe / kategoria:** Koneoppiminen (neuroverkot)

---

## 2. Harjoitustyön ydin
Työn tavoitteena on toteuttaa  neuroverkko (MLP) täysin tyhjästä hyödyntäen **NumPy-kirjastoa**, jotta ohjelman ajoajat saadaan pythonille kohtalaisiksi. Projekti keskittyy neuroverkon matemaattisten perusteiden, kuten vastavirta-algoritmin (backpropagation) ja gradienttimenetelmää (SGD), ymmärtämiseen ja implementointiin.

Keskeisin haaste on vastavirta-algoritmin toteuttaminen.

---

## 3. Ratkaistava ongelma
Tehtävänä on luokitella 28x28 pikselin kokoisia käsin kirjoitettuja numeroita (0–9) MNIST-aineistosta.

- **Syöte:** Litistetty (flattened) 784-alkioinen vektori, jossa kukin arvo on pikselin intensiteetti välillä 0.0 – 1.0 (suhteellistettu jakamalla 256)
- **Tuloste:** 10-ulotteinen todennäköisyysvektori (Softmax), jossa suurin arvo vastaa ennustettua numeroa.

---

## 4. Toteutettavat algoritmit ja mekaniikka

### 4.1 Neuroverkon rakenne
- **Input Layer:** 784 neuronia (vastaa kuvan pikseleitä).
- **Hidden Layer:** Esim. 128 neuronia, aktivointifunktiona **ReLU** ($f(x) = \max(0, x)$) tai klassinen sigmoid.
- **Output Layer:** 10 neuronia, aktivointifunktiona **Softmax** (todennäköisyysjakauma).

### 4.2 Eteenpäinkytkentä (Forward Propagation)
Lasketaan kerroskohtaisesti lineaariset transformaatiot ja epälineaariset aktivoinnit:
1. $Z = W \cdot A_{prev} + b$
2. $A = \sigma(Z)$

### 4.3 Vastavirta-algoritmi (Backpropagation)
Lasketaan virhegradientit ketjusäännön avulla lähtien ulostulosta taaksepäin:
1. Ulostulon virhe: $dZ_2 = A_2 - Y$ (missä Y on "totuus").
2. Painojen gradientit: $dW_1, db_1, dW_2, db_2$ osittaisderivaattojen avulla.
3. Virheen välittäminen piilokerrokseen ReLU-funktion derivaatan läpi.

### 4.4 Optimointi (Gradient Descent)
Päivitetään mallin parametrit kunkin iteraation jälkeen:
- $W := W - \alpha \cdot dW$
- $b := b - \alpha \cdot db$
(Missä $\alpha$ on oppimisnopeus).

---

## 5. Datan käsittely ja tallennus
- **Esikäsittely:** MNIST-kuvien normalisointi (0-255 -> 0-1) ja labelien One-Hot-koodaus.
- **Tallennus:** Valmiiksi koulutetun mallin parametrit ($W_1, b_1, W_2, b_2$) tallennetaan levylle NumPyn `.npz`-binääriformaatissa.

---

## 6. Aika- ja tilavaativuudet (O-analyysit)

Olkoon $N$ syöte-ulottuvuus, $H$ piiloneuronien määrä, $K$ ulostuloluokat ja $m$ batch-koko.

- **Aikavaativuus:** $O(m \cdot (N \cdot H + H \cdot K))$ per iteraatio. Laskenta on matriisikertolasku-dominoitua.
- **Tilavaativuus:** $O(N \cdot H + H \cdot K)$ parametrien tallentamiseen.

---

## 7. Arviointi ja testaus
- **Metriikat:** Accuracy (tarkkuusprosentti) ja Cross-Entropy Loss.
- **Visualisointi:** Häviökäyrän piirtäminen ja satunnaisten testikuvien ennustaminen visuaalisesti.

---

## 8. Projektin vaiheistus
1. Datan lataus, normalisointi ja matriisien alustus.
2. Forward pass ja aktivointifunktioiden toteutus.
3. Backpropagation-algoritmin matemaattinen johtaminen ja koodaus.
4. Koulutussilmukan hienosäätö ja tarkkuuden validointi.
5. Mallin tallennus/lataus ja CLI-käyttöliittymä.