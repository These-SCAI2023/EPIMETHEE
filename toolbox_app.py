#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import csv
import json
from collections import Counter
from os import getenv, urandom
from random import choice
from string import ascii_lowercase
from sys import stderr
from io import StringIO
from pathlib import Path

import pandas as pd
from redis import Redis
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from werkzeug.exceptions import HTTPException
from tqdm.auto import tqdm

from flask import Flask, request, render_template, url_for, redirect, Response, stream_with_context, session
from flask_babel import Babel
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect


import ocr
from cluster import freqs2clustering
from forms import ContactForm
from txt_ner import txt_ner_params

# from txt_ner import txt_ner_params

# Redis cache for geocoding and geolocator
r = Redis(
    host=getenv("REDIS_HOST", "localhost"),
    port=getenv("REDIS_PORT", 6379),
    db=getenv("REDIS_DB", 0),
    protocol=getenv("REDIS_PROTOCOL", 3),
    password=getenv("REDIS_PASSWORD", ""),
)
geolocator = Nominatim(user_agent="http")

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'static/models'
UTILS_FOLDER = 'static/utils'
ROOT_FOLDER = Path(__file__).parent.absolute()

csrf = CSRFProtect()
SECRET_KEY = urandom(32)

app = Flask(__name__)


# Babel config
def get_locale():
    return request.accept_languages.best_match(['fr', 'en'])


babel = Babel(app, locale_selector=get_locale)

# App config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = SECRET_KEY

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['UTILS_FOLDER'] = UTILS_FOLDER
app.config['LANGUAGES'] = {'fr': 'FR', 'en': 'EN', }
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)
csrf.init_app(app)

# -----------------------------------------------------------------
# BABEL
# -----------------------------------------------------------------
"""@babel.localeselector
def get_locale():
	if request.args.get('language'):
		session['language'] = request.args.get('language')
	return session.get('language', 'fr')
"""


@app.context_processor
def inject_conf_var():
    return dict(AVAILABLE_LANGUAGES=app.config['LANGUAGES'],
                CURRENT_LANGUAGE=session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'])))


@app.route('/language=<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(url_for('index'))


# -----------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------
@app.route('/')
@app.route('/index')
@app.route('/ocr_map')
def ocr_map():
    form = FlaskForm()
    return render_template('ocr_map.html', form=form)


@app.route('/projet')
def projet():
    return render_template('projet.html')


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route('/contact')
def contact():
    form = ContactForm()
    return render_template('contact.html', form=form)


# -----------------------------------------------------------------
# ERROR HANDLERS
# -----------------------------------------------------------------
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.errorhandler(413)
def file_too_big(e):
    return render_template('413.html'), 413


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({"code": e.code, "name": e.name, "description": e.description, })
    response.content_type = "application/json"
    return response


# -----------------------------------------------------------------
# FONCTIONS
# -----------------------------------------------------------------
@app.route('/send_msg', methods=["GET", "POST"])
def send_msg():
    if request.method == 'POST':
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        res = pd.DataFrame({'name': name, 'email': email, 'message': message}, index=[0])
        res.to_csv('./contactMsg.csv')
        return render_template('validation_contact.html')
    return render_template('contact.html', form=form)


#   NUMERISATION TESSERACT
@app.route('/run_tesseract', methods=["GET", "POST"])
@stream_with_context
def run_tesseract():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("tessfiles")
        model = request.form['tessmodel']

        up_folder = app.config['UPLOAD_FOLDER']
        rand_name = 'ocr_' + ''.join((choice(ascii_lowercase) for x in range(8)))

        text = ocr.tesseract_to_txt(uploaded_files, model, rand_name, ROOT_FOLDER, up_folder)
        response = Response(text, mimetype='text/plain',
                            headers={"Content-disposition": "attachment; filename=" + rand_name + '.txt'})

        return response
    return render_template('numeriser.html', erreur=erreur)


@app.route('/collecter_corpus')
def collecter_corpus():
    form = FlaskForm()
    return render_template('collecter_corpus.html', form=form)


@app.route('/correction_erreur')
def correction_erreur():
    form = FlaskForm()
    return render_template('correction_erreur.html', form=form)


@app.route('/entites_nommees')
def entites_nommees():
    form = FlaskForm()
    return render_template('entites_nommees.html', form=form)


@app.route('/etiquetage_morphosyntaxique')
def etiquetage_morphosyntaxique():
    form = FlaskForm()
    return render_template('etiquetage_morphosyntaxique.html', form=form)


@app.route('/conversion_xml')
def conversion_xml():
    form = FlaskForm()
    return render_template('conversion_xml.html', form=form)


# --------------------------
# OCR2MAP
# --------------------------

def to_geoJSON_point(coordinates, name):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [coordinates.longitude, coordinates.latitude]},
            "properties": {"name": name}, }


@app.route("/run_ocr_map", methods=["POST"])
def run_ocr_map():
    # paramètres globaux
    uploaded_files = request.files.getlist("inputfiles")
    # paramètres OCR
    ocr_model = request.form['tessmodel']
    # paramètres NER
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN = request.form['moteur_REN']
    modele_REN = request.form['modele_REN']

    rand_name = 'ocr_ner_' + ''.join((choice(ascii_lowercase) for x in range(8)))
    if ocr_model != "raw_text":
        contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
    else:
        liste_contenus = []
        for uploaded_file in uploaded_files:
            try:
                liste_contenus.append(uploaded_file.read().decode(encodage))
            finally:  # ensure file is closed
                uploaded_file.close()
        contenu = "\n\n".join(liste_contenus)

        del liste_contenus

    entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)
    ensemble_mentions = {text.strip() for label, start, end, text in entities if label == "LOC" and len(text.strip()) > 2}
    coordonnees = []
    for texte in tqdm(ensemble_mentions):
        # Use the text as the key
        key = f"text:{texte}"
        # Check if we have this in the cache
        location_data = r.get(key)
        if location_data is None:
            # If not in cache, fetch data from geolocator
            location = geolocator.geocode(texte, timeout=30)
            if location:
                location_data = [location.latitude, location.longitude]
                r.set(key, json.dumps(location_data))
                print(f"Saved {key} to cache")
        else:
            # If in cache, load the data from cache
            location_data = json.loads(location_data)
            print(f"Loaded {key} from cache")
        if location_data:
            coordonnees.append(to_geoJSON_point(location_data, texte))

    return {"points": coordonnees}


# ---------------------------------------------------------
# AFFICHAGE MAP des résultats pour plusieurs outils de NER
# ---------------------------------------------------------

@app.route("/run_ocr_map_intersection", methods=["GET", "POST"])
def run_ocr_map_intersection():
    # paramètres globaux
    uploaded_files = request.files.getlist("inputfiles")
    # paramètres OCR
    ocr_model = request.form['tessmodel']
    # paramètres NER
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN1 = request.form['moteur_REN1']
    modele_REN1 = request.form['modele_REN1']
    moteur_REN2 = request.form['moteur_REN2']
    modele_REN2 = request.form['modele_REN2']
    frequences_1 = Counter()
    frequences_2 = Counter()
    frequences = Counter()
    outil_1 = f"{moteur_REN1}/{modele_REN1}"
    outil_2 = (f"{moteur_REN2}/{modele_REN2}" if moteur_REN2 != "aucun" else "aucun")

    # print(moteur_REN1, moteur_REN2)

    rand_name = 'ocr_ner_' + ''.join((choice(ascii_lowercase) for x in range(8)))

    if ocr_model != "raw_text":
        contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
    else:
        liste_contenus = []
        for uploaded_file in uploaded_files:
            # print(uploaded_file, file=sys.stderr)
            try:
                liste_contenus.append(uploaded_file.read().decode(encodage))
            finally:  # ensure file is closed
                uploaded_file.close()
        contenu = "\n\n".join(liste_contenus)

        del liste_contenus

    # TODO: ajout cumul
    entities_1 = txt_ner_params(contenu, moteur_REN1, modele_REN1, encodage=encodage)
    ensemble_mentions_1 = {text for label, start, end, text in entities_1 if label == "LOC"}
    ensemble_positions_1 = {(text, start, end) for label, start, end, text in entities_1 if label == "LOC"}
    ensemble_positions = {(text, start, end) for label, start, end, text in entities_1 if label == "LOC"}

    # TODO: ajout cumul
    if moteur_REN2 != "aucun":
        entities_2 = txt_ner_params(contenu, moteur_REN2, modele_REN2, encodage=encodage)
        ensemble_mentions_2 = {text for label, start, end, text in entities_2 if label == "LOC"}
        ensemble_positions_2 = {(text, start, end) for label, start, end, text in entities_2 if label == "LOC"}
        ensemble_positions |= {(text, start, end) for label, start, end, text in entities_2 if label == "LOC"}
    else:
        entities_2 = ()
        ensemble_positions_2 = set()
        ensemble_mentions_2 = set()

    ensemble_mentions_commun = ensemble_mentions_1 & ensemble_mentions_2
    ensemble_mentions_1 -= ensemble_mentions_commun
    ensemble_mentions_2 -= ensemble_mentions_commun

    for text, start, end in ensemble_positions_1:
        frequences_1[text] += 1
    for text, start, end in ensemble_positions_2:
        frequences_2[text] += 1
    for text, start, end in ensemble_positions:
        frequences[text] += 1

    # print("TEST1")

    text2coord = {}
    for text in tqdm({p[0] for p in ensemble_positions}):
        text = text.strip()
        if len(text) < 3:
            continue

        # Use the text as the key
        key = f"text:{text}"
        # Check if we have this in the cache
        location_data = r.get(key)
        if location_data is None:
            # If not in cache, fetch data from geolocator
            try:
                location = geolocator.geocode(text, timeout=30)  # check for everyone
                if location:
                    location_data = [location.latitude, location.longitude]
                    # Store the result in cache for 1 hour
                    r.set(key, json.dumps(location_data))
                    print(f"Saved {key} to cache")
            except GeocoderTimedOut:
                stderr.write(f'geocoder marche pas pour EN: "{text}"\n')
        else:
            # If in cache, load the data from cache
            location_data = json.loads(location_data)
            print(f"Loaded {key} from cache")
        try:
            text2coord[text] = location_data
        except KeyError:
            stderr.write(f"Could not find {text} in location data\n")
            continue

    # TODO: faire clustering pour cumul + outil 1 / outil 2 / commun
    clusters_1 = freqs2clustering(frequences_1)
    clusters_2 = freqs2clustering(frequences_2)
    clusters = freqs2clustering(frequences)

    # print("TEST2")
    frequences_cumul_1 = {}
    for centroid in clusters_1:
        frequences_cumul_1[centroid] = 0
        for forme_equivalente in clusters_1[centroid]["Termes"]:
            frequences_cumul_1[centroid] += frequences_1[forme_equivalente]
    frequences_cumul_2 = {}
    for centroid in clusters_2:
        frequences_cumul_2[centroid] = 0
        for forme_equivalente in clusters_2[centroid]["Termes"]:
            frequences_cumul_2[centroid] += frequences_2[forme_equivalente]
    frequences_cumul = {}
    for centroid in clusters:
        frequences_cumul[centroid] = 0
        for forme_equivalente in clusters[centroid]["Termes"]:
            frequences_cumul[centroid] += frequences[forme_equivalente]

    # print("TEST3")

    # TODO: ajout cumul
    liste_keys = ["commun", outil_1, outil_2]
    liste_ensemble_mention = [ensemble_mentions_commun, ensemble_mentions_1, ensemble_mentions_2]
    dico_mention_marker = {key: [] for key in liste_keys}
    for key, ensemble in zip(liste_keys, liste_ensemble_mention):
        if key == "commun":
            my_clusters = clusters
            my_frequences = frequences_cumul
        elif key == outil_1:
            my_clusters = clusters_1
            my_frequences = frequences_cumul_1
        elif key == outil_2:
            my_clusters = clusters_2
            my_frequences = frequences_cumul_2
        else:
            raise NotImplementedError(f"Clustering pour {key} non implémenté")

        sous_ensemble = [texte for texte in my_frequences if texte in ensemble]
        for texte in sous_ensemble:
            # forms = (" / ".join(my_clusters[texte]["Termes"]) if my_clusters else "")
            # SAVE forms = [(form, [0, 0]) for form in my_clusters[texte]["Termes"]]
            forms = []
            for form in my_clusters[texte]["Termes"]:
                try:
                    coords = text2coord[form]
                except KeyError:
                    continue
                if coords:
                    coords = coords
                else:
                    coords = [0.0, 0.0]
                forms.append([form, coords])

            location = text2coord.get(texte)
            # print(location, file=sys.stderr)
            if location:
                dico_mention_marker[key].append((location[0], location[1], texte, my_frequences[texte], forms))

    return dico_mention_marker


@app.route("/nermap_to_csv", methods=['GET', "POST"])
@stream_with_context
def nermap_to_csv():
    input_json_str = request.data
    print(input_json_str)
    input_json = json.loads(input_json_str)
    print(input_json)
    keys = ["nom", "latitude", "longitude", "outil", "fréquence", "cluster"]
    output_stream = StringIO()
    writer = csv.DictWriter(output_stream, fieldnames=keys, delimiter="\t")
    writer.writeheader()
    for point in input_json["data"]:
        row = {
            "latitude": point[0],
            "longitude": point[1],
            "nom": point[2],
            "outil": point[3],
            "fréquence": point[4],
            "cluster": point[5],
        }
        writer.writerow(row)
    # name not useful, will be handled in javascript
    response = Response(output_stream.getvalue(), mimetype='text/csv',
                        headers={"Content-disposition": "attachment; filename=export.csv"})
    output_stream.seek(0)
    output_stream.truncate(0)
    return response


@app.route("/nermap_to_csv2", methods=['GET', "POST"])
@stream_with_context
def nermap_to_csv2():
    from lxml import etree

    keys = ["nom", "latitude", "longitude", "outil", "cluster"]
    output_stream = StringIO()
    writer = csv.DictWriter(output_stream, fieldnames=keys, delimiter=",")
    writer.writeheader()

    input_json = json.loads(request.data)
    html = etree.fromstring(input_json["html"])
    base_clusters = input_json["clusters"]
    name2coordinates = {}
    print(base_clusters)
    for root_cluster in base_clusters.values():
        for *_, clusters in root_cluster:
            for txt, coords in clusters:
                name2coordinates[txt] = coords
    print(name2coordinates)

    for toolnode in list(html):
        for item in list(toolnode):
            tool = item.text.strip()
            for centroid_node in list(list(item)[0]):
                print(centroid_node)
                centroid = etree.tostring(next(centroid_node.iterfind("div")), method="text", encoding=str).strip()
                # centroid = centroid_node.text_content().strip()
                try:
                    data = next(centroid_node.iterfind('ol'))
                except StopIteration:  # cluster with no children
                    data = []
                the_cluster = []
                for cluster_item_node in list(data):
                    try:
                        cluster_item = etree.tostring(cluster_item_node, method="text", encoding=str).strip()
                        the_cluster.append(cluster_item.split(" / ")[0])
                    except Exception:
                        stderr.write("\t\tDid not work")
                nom = centroid  # .split(' / ')[0]
                #  latitude = centroid.split(' / ')[1].split(',')[0],
                #  longitude = centroid.split(' / ')[1].split(',')[1],
                print(nom, nom in name2coordinates)
                try:
                    latitude, longitude = name2coordinates[nom]
                except KeyError:
                    stderr.write(f"Could not find {nom} in coordinates")
                    continue
                writer.writerow(
                    {
                        "nom": nom,
                        "latitude": latitude,
                        "longitude": longitude,
                        "outil": tool,
                        "cluster": ', '.join(the_cluster),
                    }
                )

    # name not useful, will be handled in javascript
    response = Response(output_stream.getvalue(), mimetype='text/csv',
                        headers={"Content-disposition": "attachment; filename=export.csv"})
    output_stream.seek(0)
    output_stream.truncate(0)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
