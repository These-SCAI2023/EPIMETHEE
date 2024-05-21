# Épiméthée dans Pandore (Toolbox-site ObTIC)

## Installation

## Via docker
1. Veillez à avoir docher et docker-compose installés sur votre machine. Pour se faire, veuillez suivre les instructions sur le site officiel de docker : https://docs.docker.com/engine/install/ et suivre les instructions pour votre système d'exploitation.

2. Clonez le répertoire actuel sur votre machine en utilisant la commande suivante : 
```bash
git clone https://github.com/These-SCAI2023/EPIMETHEE.git -b docker
```

2. (bis) Vous pouvez également télécharger le fichier `docker-compose.yml` directement depuis le site github et le placer dans un dossier de votre choix (de préférence vide et nommé `EPIMETHEE` pour éviter les conflits de noms de dockers).

3. Placez-vous dans le répertoire `EPIMETHEE` et lancez la commande suivante pour démarrer les dockers : 
```bash
docker-compose up -d
```
Il est possible que vous ayez besoin de droits d'administrateur pour lancer cette commande. Dans ce cas, ajoutez `sudo` devant la commande.
Une fois cette commande lancée, les dockers se téléchargeront et se lanceront automatiquement. Vous pourrez accéder à l'interface de la toolbox en vous rendant à l'adresse http://localhost:8080 dans votre navigateur.

4. Pour arrêter les dockers, placez-vous dans le répertoire `EPIMETHEE` et lancez la commande suivante : 
```bash
docker-compose down
```
Il est possible que vous ayez besoin de droits d'administrateur pour lancer cette commande. Dans ce cas, ajoutez `sudo` devant la commande.

### Mise à jour de l'application

Pour mettre à jour l'application, il suffit de se placer dans le répertoire `EPIMETHEE` et de lancer la commande suivante : 
```bash
docker-compose pull
```
Il est possible que vous ayez besoin de droits d'administrateur pour lancer cette commande. Dans ce cas, ajoutez `sudo` devant la commande.

Puis relancez les dockers avec la commande `docker-compose up -d` (toujours dans le répertoire `EPIMETHEE` et en ayant les droits d'administrateur si nécessaire).


### Installation manuelle (non recommandée) pour linux et mac

1. Installez les paquets nécessaires à l'exécution de l'application :

```bash
sudo apt-get install git python3.10 python3.10-venv python3.10-pip tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-osd poppler-utils
```

2. Clonez le répertoire EPIMETHEE :

```bash
git clone https://github.com/These-SCAI2023/EPIMETHEE.git
```
et placez-vous dans le répertoire EPIMETHEE :

```bash
cd EPIMETHEE
```

3. Créez et activez un environnement virtuel pour Python 3.10 :

```bash
python3.10 -m venv venv
source venv/bin/activate
```

5. Installez les paquets Python nécessaires à l'exécution de l'application :

```bash
pip install -r requirements.txt
```


### Lancer l'application

Placé dans le dossier Toolbox-site, activez l'environnement virtuel :
```bash
source venv/bin/activate
```

Puis lancez l'application :

```bash
python toolbox_app.py
```

Ouvrir le lien http://localhost:8080 dans un navigateur pour accéder à l'interface de la Toolbox ObTIC.

## Version en ligne

Une [version de démonstration](http://pp-obtic.sorbonne-universite.fr/toolbox/) est disponible en ligne.
Une nouvelle version pour diffusion plus large est en cours de conception.

____



# Bibliographie
Koudoro-Parfait, C., Alrahabi, M., Dupont, Y., Lejeune, G., & Roe, G. (2023, juin 30). Mapping spatial named entities from noisy OCR output: Epimetheus from OCR to map. Digital Humanities 2023. Collaboration as Opportunity (DH2023), Graz, Austria. https://doi.org/10.5281/zenodo.8108050

Motasem ALRAHABI, Valentina FEDCHENKO, Ljudmila PETKOVIC, Glenn ROE. Pandore: a toolbox for digital humanities text-based workflows. [soumission acceptée [DH2023](https://dh2023.adho.org/?page_id=390)]

Johanna Cordova, Yoann Dupont, Ljudmila Petkovic, James Gawley, Motasem Alrahabi, et al.. Toolbox : une chaîne de traitement de corpus pour les humanités numériques. *Traitement Automatique des Langues Naturelles*, 2022, Avignon, France. pp.11-13. ⟨[hal-03701464](https://hal.archives-ouvertes.fr/TALN-RECITAL2022/hal-03701464)⟩


# Mentions légales

Le code est distribué sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) par l'[équipe ObTIC](https://obtic.sorbonne-universite.fr/) (Sorbonne Université).

# 
