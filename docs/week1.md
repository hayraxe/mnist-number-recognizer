# Viikkoraportti 1

## Mitä olen tehnyt tällä viikolla?
- Valitsin harjoitustyön aiheen ja rajasin sen kurssille sopivaksi (omasta mielestä).
- Halusin tehdä jotain, mikä pyrkii ymmärtämään tai oppimaan, miten jokin systeemi toimii pelkästään havaintodatan perusteella. Pyörittelin pitkään erilaisia kausaaliseen koneoppimiseen liittyviä ideoita.
- Idean inspiraationa oli "world modelit", mitkä ovat puhuttaneet tekoälymaailmassa viime aikoina. Tutustuin alkuun Yann LeCunin näkemyksiin "world modeleista" sekä hänen uusimpiin JEPA (joint embedding predictive architecture) malleihin.
- Huomasin kuitenkin nopeasti, että tällaisten systeemien rakentaminen on todella kompleksista. Niimpä pyrin rajaaman idean niin, että pysytään samassa hengessä huomattavasti helpommin lähestyttävällä leluversiolla.
- Päädyin rajaukseen vertailla kolmea erilaista menetelmää vuorovaikutusverkkojen päättelyyn hiukkasdatasta. Hiukkasdata tuotetaan simulaatiolla.
  1) korrelaatiopohjainen kynnysmenetelmä  
  2) ennustettavuuspohjainen lineaarinen testi (kaksi mallia: oma historia vs oma+toinen)  
  3) neuroverkkomenetelmä (1D-CNN aikasarjaluokittimena)
- Laitoin perusteet GitHub repositoriosta tulille.

## Miten ohjelma on edistynyt?
- Projektin suunta ja toteutustapa ovat nyt selkeät: simulaattori → data → verkon inferenssi → metriikat + (kevyt) visualisointi.
- Ideoista olisi ollut toki mielekästä jutella kurssin henkilökunnan kanssa, mutta myöhäisen ilmoittautumisen ja aikataulukiireiden takia se ei tällä kertaa onnistunut.

## Mitä opin tällä viikolla / tänään?
- Opin, että kannattaa lähteä pienemmillä askeleilla liikkeelle, mikä toimii hyvänä alustuksena jatkoa varten.
- Opin myös, että 1D-CNN pitäisi olla käytännössä lähestyttävämpi tapa luokitella aikasarjaikkunoita ilman monimutkaisia toistoverkkoja (LSTM/GRU).

## Mikä jäi epäselväksi tai tuottanut vaikeuksia?
- Miten laaja simulaattori ja datagenerointi tarvitaan, jotta inferenssimenetelmien erot näkyvät luotettavasti (esim. sopivat jousivakioiden vaihteluvälit, vaimennus, aika-askeleen valinta). Huojennuksena tähän projektiin pystyy helpohkosti tuottamaan synteettistä dataa.

## Mitä teen seuraavaksi?
- Toteutan hiukkassimulaattorin (N hiukkasta, jousiverkko, integraattori) ja datan tallennuksen muotoon `[M, T, N, 4]`.
- Teen ensimmäisen pienen testidatasetin (pieni N ja T) ja varmistusvisualisoinnit (trajektorit + todellinen verkko).
- Implementoin korrelaatiopohjaisen menetelmän ja teen ensimmäiset F1/precision/recall-mittaukset.
- Asetan projektin perusajettavuuden kuntoon (Poetry ja perusajokomennot).

## Tuntikirjanpito
- Käytetty aika tällä viikolla: **8 h**
