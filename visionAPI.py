#!/usr/bin/env python
"""
Author: Jesus Aguas Acin
Date: May 7th, 2020
Usage: python visionAPI.py [path to image]
Notes: For this to work you will need to include your key.json in the same folder
        as this file. For more information check:
        https://codelabs.developers.google.com/codelabs/cloud-vision-api-python
visionAPI.py: Given the path of an image (local or remote),
                Returns image web entities, labels, faces, objects, logos and
                text detection using the Cloud Vision API.
"""

import os                               # To set environment variables
import io                               # For reading from files
import argparse                         # To accept input filenames as arguments
from google.cloud import vision         # For accessing the Vision API
from google.cloud.vision import types   # For constructing requests

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'key.json'

def annotate(path):
    """ Returns label/web/face/logo/text/object annotations given the path to an image. """
    client = vision.ImageAnnotatorClient()
    if path.startswith('http') or path.startswith('gs:'):
        image = types.Image()
        image.source.image_uri = path
    else:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)

    # LABELS
    labels_response = client.label_detection(image=image)
    labels_report(labels_response)

    # WEB ENTITIES
    web_response = client.web_detection(image=image).web_detection
    web_report(web_response)

    # FACE
    face_response = client.face_detection(image=image)
    face_report(face_response)

    # LOGO
    logo_response = client.logo_detection(image=image)
    logo_report(logo_response)

    # TEXT
    text_response = client.text_detection(image=image)
    text_report(text_response)

    # OBJECTS
    object_response = client.object_localization(image=image).localized_object_annotations
    object_report(object_response)


def object_report(objects):
    print('\nObjects:')
    print('=' * 79)
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

def text_report(response):
    print('\nTexts:')
    print('=' * 79)
    texts = response.text_annotations
    for text in texts:
        print('\n"{}"'.format(text.description))
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])
        print('bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def logo_report(response):
    print('\nLogos:')
    print('=' * 79)
    logos = response.logo_annotations
    for logo in logos:
        print(logo.description)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def face_report(response):
    print('\nFaces:')
    print('=' * 79)
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    faces = response.face_annotations
    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])
        print('face bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def labels_report(response):
    print('\nLabels (and confidence score):')
    print('=' * 79)
    for label in response.label_annotations:
        print(f'{label.description} ({label.score*100.:.2f}%)')

def web_report(annotations):
    print('\nWeb entities (and relevancy score):')
    print('=' * 79)
    """Prints detected features in the provided web annotations."""
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images retrieved'.format(
            len(annotations.pages_with_matching_images)))
        for page in annotations.pages_with_matching_images:
            print('Url   : {}'.format(page.url))

    if annotations.full_matching_images:
        print('\n{} Full Matches found: '.format(
              len(annotations.full_matching_images)))
        for image in annotations.full_matching_images:
            print('Url  : {}'.format(image.url))

    if annotations.partial_matching_images:
        print('\n{} Partial Matches found: '.format(
              len(annotations.partial_matching_images)))
        for image in annotations.partial_matching_images:
            print('Url  : {}'.format(image.url))

    if annotations.web_entities:
        print('\n{} Web entities found: '.format(
              len(annotations.web_entities)))
        for entity in annotations.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    path_help = str('The image to detect, can be web URI, '
                    'Google Cloud Storage, or path to local file.')
    parser.add_argument('image_url', help=path_help)
    args = parser.parse_args()
    annotate(args.image_url)
