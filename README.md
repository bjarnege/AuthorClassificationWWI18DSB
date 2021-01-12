# AuthorClassificationWWI18DSB

## Übersicht 
* [Frontend nutzen](#frontend)

## Frontend nutzen
Um die trainierten Modelle nutzerfreundlich über ein Frontend auf dem loclahost nutzen zu können,sind folgende Schritte zu befolgen:
1. Das Github-Repository muss für die Nutzung zunächst heruntergeladen werden. Dies kann über die Verwendung von [```git clone``` in der Command Line](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository) geschehen (hierfür muss aber zuvor [Git für die Komamndozeile installiert werden](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git)). Alternativ kann das Repository auch einfach als [ZIP-Datei heruntergeladen](https://github.com/bjarnege/AuthorClassificationWWI18DSB/archive/main.zip) und entpackt werden.
2. Um das Frontend zu starten, muss sichergestellt werden, dass [Python3](https://wiki.python.org/moin/BeginnersGuide/Download) und die folgenden Bibliotheken via [```pip install```](https://packaging.python.org/tutorials/installing-packages/) installiert sind:
    * Pandas 🐼
    * Pillow ✏️
    * Flask 🖥
3. Das Frontend kann gestartet werden, indem die Python-Datei ```./frontend/app.py``` ausgeführt wird. Wenn die Datei über eine IDE wie bespielsweise Visual Studio Code ausgeführt wird, muss darauf geachtet werden, dass das Working Directory der Ordner ```./frontend``` ist.
4. Wenn Der Flask-Server gestartet ist, kann das Frontend über den Browser unter volgendem Link erreicht werden: [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

**Im folgenden wird die Nutzung des Frontends genauer beschrieben:**\
Von der Startseite aus kann der Benutzer zu einer Texteingabe wechseln, in welche der Text eingefügt werden muss, dessen Autor in den nächsten Schritten analysiert werden soll. Die darunterliegende Dropdown-Liste ermöglicht die Wahl der vorherzusagenden Zielvariable. Dabei kann zwischen dem Alter, dem Geschlecht, dem Genre und dem Sternzeichen gewählt werden. Wird keine Wahl getroffen, so wird sofort ein Ergebnis ausgegeben, welches lediglich auf der textuellen Klassifikation basiert. Wird aber eine der vier Variablen für eine bessere vorhersage ausgewählt und fortgefahren, müssen Werte für die übrigen drei Variablen angegeben werden, was voraussetzt, dass diese bereits bekannt sind. Anschließend werden ebenfalls die Ergebnisse der Klassifikation ausgegeben. Nun basieren sie aber auf dem gewichteten Stacking beider Modelle. Auf der Ergebnisseite wird links oben der Eingabetext wiederholt. Rechts davon ist das Ergebnis des Genres und des Alters des Autors zu sehen. Unten links wird die Vorhersage für das Geschlecht, daneben das Ergebnis der aus dem Clustering gewonnenen Variablen, bei einer reinen Texteingabe jedoch nur eine zusätzliche Eigenschaft und das Sternzeichen ausgegeben. Im Falle der verbesserten Prognose bezieht sich nur eines der Ergebnisse und das Clustering auf die berechnete Prognose. Die anderen basieren hingegen auf der vorangegangenen Eingabe. Über die Menüleiste kann zwischen der Startseite, der Texteingabe, dem zuletzt erhaltenen Ergebnis und einer Übersicht der an dieser Arbeit beteiligten Teammitglieder gewechselt werden.
