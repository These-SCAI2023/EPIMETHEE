#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask, abort, request, render_template, url_for, redirect, send_from_directory, Response, stream_with_context, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from forms import ContactForm, SearchForm
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_babel import Babel, get_locale, gettext
import os
from io import StringIO, BytesIO
import string
import random
from bs4 import BeautifulSoup
import urllib
import urllib.request
from urllib.parse import urlparse
import re
from lxml import etree
import csv
import sys
import shutil
import subprocess
import glob
from pathlib import Path
import json
import collections
#from txt_ner import txt_ner_params

from geopy.exc import GeocoderTimedOut

import pandas as pd

import ocr

from cluster import freqs2clustering

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'static/models'
UTILS_FOLDER = 'static/utils'
ROOT_FOLDER = Path(__file__).parent.absolute()

csrf = CSRFProtect()
SECRET_KEY = os.urandom(32)

app = Flask(__name__)

# Babel config
def get_locale():
	return request.accept_languages.best_match(['fr', 'en'])
babel = Babel(app, locale_selector=get_locale)

# App config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = SECRET_KEY

app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 # Limit file upload to 8MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['UTILS_FOLDER'] = UTILS_FOLDER
app.config['LANGUAGES'] = ['fr', 'en']
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)
csrf.init_app(app)


#-----------------------------------------------------------------
# BABEL
#-----------------------------------------------------------------
"""@babel.localeselector
def get_locale():
	if request.args.get('language'):
		session['language'] = request.args.get('language')
	return session.get('language', 'fr')
"""
@app.context_processor
def inject_conf_var():
	return dict(AVAILABLE_LANGUAGES=app.config['LANGUAGES'], CURRENT_LANGUAGE=session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'])))

@app.route('/language=<language>')
def set_language(language=None):
	session['language'] = language
	return redirect(url_for('index'))

#-----------------------------------------------------------------
# ROUTES
#-----------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/pandore')
def pandore():
	return render_template('pandore.html')

@app.route('/projet')
def projet():
	return render_template('projet.html')

@app.route('/outils')
def outils():
	form = SearchForm()
	return render_template('outils.html', form=form)

@app.route('/documentation')
def documentation():
	return render_template('documentation.html')

@app.route('/contact')
def contact():
	form = ContactForm()
	return render_template('contact.html', form=form)

@app.route('/outils_corpus')
def outils_corpus():
	return render_template('corpus.html')

@app.route('/outils_fouille')
def outils_fouille():
	return render_template('fouille_de_texte.html')

@app.route('/outils_visualisation')
def outils_visualisation():
	return render_template('visualisation.html')

@app.route('/numeriser')
def numeriser():
	form = FlaskForm()
	return render_template('numeriser.html', form=form)

@app.route('/normalisation')
def normalisation():
	return render_template('normalisation.html')

@app.route('/categories_semantiques')
def categories_semantiques():
	return render_template('categories_semantiques.html')

@app.route('/resume_automatique')
def resume_automatique():
	return render_template('resume_automatique.html')

@app.route('/extraction_mots_cles')
def extraction_mots_cles():
	form = FlaskForm()
	return render_template('extraction_mots_cles.html', form=form, res={})

@app.route('/topic_modelling')
def topic_modelling():
	form = FlaskForm()
	return render_template('topic_modelling.html', form=form, res={})

@app.route('/outils_pipeline')
def outils_pipeline():
	return render_template('pipeline.html')

@app.route('/ocr_ner')
def ocr_ner():
	form = FlaskForm()
	return render_template('ocr_ner.html', form=form)

@app.route('/ocr_map')
def ocr_map():
	form = FlaskForm()
	return render_template('ocr_map.html', form=form)

@app.route('/extraction_gallica')
def extraction_gallica():
	form = FlaskForm()
	return render_template('extraction_gallica.html', form=form)

#-----------------------------------------------------------------
# ERROR HANDLERS
#-----------------------------------------------------------------
"""@app.errorhandler(500)
def internal_server_error(e):
	return render_template('500.html'), 500"""

@app.errorhandler(413)
def file_too_big(e):
	return render_template('413.html'), 413

"""@app.errorhandler(Exception)
def handle_exception(e):
	# pass through HTTP errors
	if isinstance(e, HTTPException):
		return e

	# now you're handling non-HTTP exceptions only
	return render_template("500_custom.html", e=e), 500"""

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

#-----------------------------------------------------------------
# FONCTIONS
#-----------------------------------------------------------------
@app.route('/send_msg',  methods=["GET","POST"])
def send_msg():
	if request.method == 'POST':
		name =  request.form["name"]
		email = request.form["email"]
		message = request.form["message"]
		res = pd.DataFrame({'name':name, 'email':email,'message':message}, index=[0])
		res.to_csv('./contactMsg.csv')
		return render_template('validation_contact.html')
	return render_template('contact.html', form=form)


#   NUMERISATION TESSERACT
@app.route('/run_tesseract',  methods=["GET","POST"])
@stream_with_context
def run_tesseract():
	if request.method == 'POST':
		uploaded_files = request.files.getlist("tessfiles")
		model = request.form['tessmodel']

		up_folder = app.config['UPLOAD_FOLDER']
		rand_name =  'ocr_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))

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


#--------------------------
#OCR2MAP
#--------------------------

def to_geoJSON_point(coordinates, name):
	return {
		"type": "Feature",
		"geometry": {
			"type": "Point",
			"coordinates": [coordinates.longitude, coordinates.latitude]
		},
		"properties": {
			"name": name
		},
	}


@app.route("/run_ocr_map", methods=["POST"])
def run_ocr_map():
	from txt_ner import txt_ner_params
	from geopy.geocoders import Nominatim
	geolocator = Nominatim(user_agent="http")

	# paramètres globaux
	uploaded_files = request.files.getlist("inputfiles")
	# paramètres OCR
	ocr_model = request.form['tessmodel']
	# paramètres NER
	up_folder = app.config['UPLOAD_FOLDER']
	encodage = request.form['encodage']
	moteur_REN = request.form['moteur_REN']
	modele_REN = request.form['modele_REN']

	rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
	if ocr_model != "raw_text":
		contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
	else:
		liste_contenus = []
		for uploaded_file in uploaded_files:
			try:

				liste_contenus.append(uploaded_file.read().decode(encodage))
			finally: # ensure file is closed
				uploaded_file.close()
		contenu = "\n\n".join(liste_contenus)

		del liste_contenus

	entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)
	ensemble_mentions = set(text for label, start, end, text in entities if label == "LOC")
	coordonnees = []
	for texte in ensemble_mentions:
		location = geolocator.geocode(texte, timeout=30)
		if location:
			coordonnees.append(to_geoJSON_point(location, texte))

	return {"points": coordonnees}

#---------------------------------------------------------
#AFFICHAGE MAP des résultats pour plusieurs outils de NER
#---------------------------------------------------------

@app.route("/run_ocr_map_intersection", methods=["GET", "POST"])
def run_ocr_map_intersection():
	from txt_ner import txt_ner_params
	from geopy.geocoders import Nominatim
	geolocator = Nominatim(user_agent="http")
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
	frequences_1 = collections.Counter()
	frequences_2 = collections.Counter()
	frequences = collections.Counter()
	outil_1 = f"{moteur_REN1}/{modele_REN1}"
	outil_2 = (f"{moteur_REN2}/{modele_REN2}" if moteur_REN2 != "aucun" else "aucun")

	# print(moteur_REN1, moteur_REN2)

	rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))

	if ocr_model != "raw_text":
		contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
	else:
		liste_contenus = []
		for uploaded_file in uploaded_files:
			# print(uploaded_file, file=sys.stderr)
			try:
				liste_contenus.append(uploaded_file.read().decode(encodage))
			finally: # ensure file is closed
				uploaded_file.close()
		contenu = "\n\n".join(liste_contenus)

		del liste_contenus

	# TODO: ajout cumul
	entities_1 = txt_ner_params(contenu, moteur_REN1, modele_REN1, encodage=encodage)
	ensemble_mentions_1 = set(text for label, start, end, text in entities_1 if label == "LOC")
	ensemble_positions_1 = set((text, start, end) for label, start, end, text in entities_1 if label == "LOC")
	ensemble_positions = set((text, start, end) for label, start, end, text in entities_1 if label == "LOC")

	# TODO: ajout cumul
	if moteur_REN2 != "aucun":
		entities_2 = txt_ner_params(contenu, moteur_REN2, modele_REN2, encodage=encodage)
		ensemble_mentions_2 = set(text for label, start, end, text in entities_2 if label == "LOC")
		ensemble_positions_2 = set((text, start, end) for label, start, end, text in entities_2 if label == "LOC")
		ensemble_positions |= set((text, start, end) for label, start, end, text in entities_2 if label == "LOC")
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
	for text in set(p[0] for p in ensemble_positions):
		try:
			text2coord[text] = geolocator.geocode(text, timeout=30) # check for everyone
		except GeocoderTimedOut:
			sys.stderr.write(f'geocoder marche pas pour EN: "{text}"\n')

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
		sous_ensemble = [texte for texte in my_frequences if texte in ensemble]
		for texte in sous_ensemble:
			# forms = (" / ".join(my_clusters[texte]["Termes"]) if my_clusters else "")
			#SAVE forms = [(form, [0, 0]) for form in my_clusters[texte]["Termes"]]
			forms = []
			for form in my_clusters[texte]["Termes"]:
				coords = text2coord[form]
				if coords:
					coords = [text2coord[form].latitude, text2coord[form].longitude]
				else:
					coords = [0.0, 0.0]
				forms.append([form, coords])
			# location = geolocator.geocode(texte, timeout=30) # déjà fait avant
			location = text2coord[texte]
			# print(location, file=sys.stderr)
			if location:
				dico_mention_marker[key].append((
					location.latitude,
					location.longitude,
					texte,
					my_frequences[texte],
					forms
				))

	# for key, value in dico_mention_marker.items():
	# 	print(key, value, file=sys.stderr)

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
			"latitude" : point[0],
			"longitude" : point[1],
			"nom" : point[2],
			"outil" : point[3],
			"fréquence" : point[4],
			"cluster" : point[5],
		}
		writer.writerow(row)
	# name not useful, will be handled in javascript
	response = Response(output_stream.getvalue(), mimetype='text/csv', headers={"Content-disposition": "attachment; filename=export.csv"})
	output_stream.seek(0)
	output_stream.truncate(0)
	return response


@app.route("/nermap_to_csv2", methods=['GET', "POST"])
@stream_with_context
def nermap_to_csv2():
	from lxml import etree

	keys = ["nom", "latitude", "longitude", "outil", "cluster"]
	output_stream = StringIO()
	writer = csv.DictWriter(output_stream, fieldnames=keys, delimiter="\t")
	writer.writeheader()

	input_json = json.loads(request.data)
	html = etree.fromstring(input_json["html"])
	base_clusters = input_json["clusters"]
	name2coordinates = {}
	print(base_clusters)
	for root_cluster in base_clusters.values():
		for _, _, _, _, clusters in root_cluster:
			for txt, coords in clusters:
				name2coordinates[txt] = coords
	print(name2coordinates)

	for toolnode in list(html):
		for item in list(toolnode):
			tool = item.text.strip()
			for centroid_node in list(list(item)[0]):
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
						print("\t\tDid not work")
				nom = centroid#.split(' / ')[0]
				#  latitude = centroid.split(' / ')[1].split(',')[0],
				#  longitude = centroid.split(' / ')[1].split(',')[1],
				print(nom, nom in name2coordinates)
				latitude, longitude = name2coordinates[nom]
				writer.writerow({
					"nom" : nom,
					"latitude" : latitude,
					"longitude" : longitude,
					"outil" : tool,
					"cluster" : ', '.join(the_cluster),
				})

	# name not useful, will be handled in javascript
	response = Response(output_stream.getvalue(), mimetype='text/csv', headers={"Content-disposition": "attachment; filename=export.csv"})
	output_stream.seek(0)
	output_stream.truncate(0)
	return response


if __name__ == "__main__":
	app.run()
