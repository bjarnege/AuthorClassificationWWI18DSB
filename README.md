# AuthorClassificationWWI18DSB

## Übersicht 
* [Frontend nutzen](#frontend)



## Vorbereitung
Um die Module verwenden zu können, müssen vorbereitend, die fehlender Binärdateien bezogen werden, welche Trainierte Modelle und Daten beeinhalten.

Dies wird durchgeführt, indem aus dem Hauptverzeichnis heraus die Datei ```setup.py``` ausgeführt wird:

```python ./scr/setup.py```

Bitte beachten, dass die fehlenden Dateien etwa 4.14 GB groß sind und der Download daher entsprechend lange dauert.
Sollte das obige Skirpt nicht dazu führen, dass der Ordner ```resource``` entsteht, kann die Datei manuell unter [Google Drive](https://drive.google.com/file/d/1-UMB3mltSgvWE-JxduJW9GZaniXmAa3w/view?usp=sharing) bezogen werden.


## Setup (manuell)
Um die Modelle zu verwenden, muss wie folgt verfahren werden:

1. Installieren der benötigten Python-libaries

```pip install -r requirements.txt```

2. Ausführen der Flask-APP

`` cd ./src/frontend/``

```python ./app.py```
Achtung: Es ist notwendig, die app.py Datei direkt aus dem Ordner auszuführen! Ein Ausführen über bspw. ``` python ./src/frontend```
ist nicht möglich.

## Setup (docker)
Um die Modelle über docker anzusprechen, bitte den folgenden Befehl ausführen:

1. ```docker build -t authorclassification .```
2. ```docker run -p 5000:5000 authorclassification```

Aufgrund der Dateigröße ist hier leider ein wenig Geduld erfodrerlich.

## Notebooks & Module

Die Notebooks die im Rahmen des Integrationsseminars erstellt worden sind, befinden sich in dem Ordner ``` notebooks```. Diese lassen sich nicht innerhallb des Docker-Containers ausführen, können aber sofern die benötigten Module lokal installiert worden sind, und der obige Schritt ```Vorbereitungen``` durchgeüfhrt worden ist, eingesehen und verwendet werden.
Die erstellten Module lassen sich innerhalb des ```src```-Ordners finden.

## Frontend nutzen
Um die trainierten Modelle nutzerfreundlich über ein Frontend auf dem loclahost nutzen zu können,sind folgende Schritte zu befolgen:
1. Das Github-Repository muss für die Nutzung zunächst heruntergeladen werden. Dies kann über die Verwendung von [```git clone``` in der Command Line](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository) geschehen (hierfür muss aber zuvor [Git für die Komamndozeile installiert werden](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git)). Alternativ kann das Repository auch einfach als [ZIP-Datei heruntergeladen](https://github.com/bjarnege/AuthorClassificationWWI18DSB/archive/main.zip) und entpackt werden.
2. Um das Frontend zu starten, muss sichergestellt werden, dass [Python3](https://wiki.python.org/moin/BeginnersGuide/Download) und die folgenden Bibliotheken via [```pip install```](https://packaging.python.org/tutorials/installing-packages/) installiert sind:
    * Pandas 🐼
    * Pillow ✏️
    * Flask 🖥
3. Das Frontend kann gestartet werden, indem die Python-Datei ```./frontend/app.py``` ausgeführt wird. Wenn die Datei über eine IDE wie bespielsweise Visual Studio Code ausgeführt wird, muss darauf geachtet werden, dass das Working Directory der Ordner ```./frontend``` ist.
4. Wenn Der Flask-Server gestartet ist, kann das Frontend über den Browser unter folgendem Link erreicht werden: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

**Im folgenden wird die Nutzung des Frontends genauer beschrieben:**\
Von der Startseite aus kann der Benutzer zu einem Formular wechseln, in welchem er den Text wählt, dessen Autor in den nächsten Schritten analysiert werden soll. Die darunterliegende Dropdown-Liste ermöglicht die Wahl der vorherzusagenden Zielvariable. Dabei kann zwischen dem Alter, dem Geschlecht, dem Genre und dem Sternzeichen gewählt werden. Im folgenden Schritt müssen Werte für die übrigen drei Variablen angegeben werden, was voraussetzt, dass diese bereits bekannt sind. Anschließend werden die Ergebnisse der Klassifikation ausgegeben. Diese basieren auf dem gewichteten Stacking des textuellen und numerischen Modells. Auf der Ergebnisseite wird links oben der Eingabetext wiederholt. Rechts davon ist das Ergebnis des Genres und des Alters des Autors zu sehen. Unten links wird die Vorhersage für das Geschlecht, daneben das Ergebnis der aus dem Clustering gewonnenen Variablen und das Sternzeichen ausgegeben. Je nachdem welche Variable vorab als Unbekannte ausgewählt wurde, bezieht sich nur eines der Ergebnisse und das Clustering auf die berechnete Prognose. Die anderen basieren hingegen auf der vorangegangenen manuellen Eingabe. Unter dem Überblick zu den Eigenschaften wird ein maschinell gernerierter Text angezeigt, welcher dem Eingabetext ähnelt. Über die Menüleiste kann zwischen der Startseite, der Texteingabe, dem zuletzt erhaltenen Ergebnis und einer Übersicht der an dieser Arbeit beteiligten Teammitglieder gewechselt werden.
