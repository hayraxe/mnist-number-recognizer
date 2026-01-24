# Viikkoraportti 2


## Mitä olen tehnyt tällä viikolla?
Tällä viikolla projektin aihe vaihtui täysin, sillä alkuperäinen aihe osoittautui liian haastavaksi toteuttaa (ilman syvempää esitietoa neuroverkoista) Uusi aihe on klassinen käsinkirjoitettujen numeroiden (MNIST) tunnistus neuroverkolla.

Viikon aikana olen:
* Määritellyt uuden projektisuunnitelman ja neuroverkon arkkitehtuurin (784 -> 16 -> 16 -> 10).
* Alustanut projektin rakenteen & testausympäristön (vaikka testejä ei vielä olekkkaan)
* Perehtynyt MLP-neuroverkkoihin

## Miten ohjelma on edistynyt?
Teknine toteutus on aloitettu
* **Malli:** `NeuralNetwork`-luokka on luotu. Painojen alustus (`__init__`) ja eteenpäinkytkentä (`forward_propagation`) on toteutettu matriisitasolla.
* **Testaus:** Pitää vielä tehdä

## Mitä opin tällä viikolla / tänään?
* Opin MLP-verkon perustoiminnan (treenauksen jälkeen sekä, miten takaisinvirtauslgoritmi pääpiirteittäin toimii)
* Opin eri aktivointifunktioiden toimintaperiaatteita (sigmoid & ReLu)

## Mikä jäi epäselväksi tai tuottanut vaikeuksia?
* Suurin haaste on takaisinvirtausalgoritmi

## Mitä teen seuraavaksi?
1.  **Vastavirta-algoritmi (Backpropagation):** Toteutan varsinaisen oppimislogiikan eli gradienttien laskennan ja painojen päivityksen.
2.  **Datan lataaja:** Lataa levylle tallennetun MNIST-datan ja esikäsittelee seb (normalisointi ja lytistäminen).
3.  **Harjoitussilmukka:** Harjoituksien ajaminen pienemmissä sekoitetuissa "bätcheissä"
4.  **Testit:** Kattavat testit