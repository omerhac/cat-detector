import json
import subprocess
import requests


def authorize(client):
    """Generate new token based on client (1 or 2)credentials and save it to token.dat"""
    subprocess.run(f'./petfinder_auth{client}.sh', shell=True, universal_newlines=False)


def get_token(client):
    """Return client token value"""
    with open(f'token{client}.dat', 'r') as f:
        t = f.readlines()[0]
    return t[:-1]


def get_authorized_session(client, new_token=False):
    """Return an authorized session for a client for handling petfinder API
    Args:
        client: client number
        new_token: flag whether to generate new token
    """

    if new_token:
        authorize(client)

    token = get_token(client)
    sess = requests.Session()
    bearer = 'Bearer ' + token
    sess.headers.update({'Authorization': bearer})

    return sess


def petfinder_request(request, client, new_token=False):
    """Make a petfinder API request. Return response dictionary.
    Args:
        request: API request string, with pattern {CATEGORY}/{ACTION}?{parameter_1}={value_1}&{parameter_2}={value_2}
        client: client number
        new_token: flag whether to generate new token

    Returns:
        response dictionary
        response code
    """

    # get authorized session
    with get_authorized_session(client, new_token=new_token) as sess:
        response = sess.get("https://api.petfinder.com/v2/" + request)

    return json.loads(response.text), response.status_code


def get_cat_images(cat):
    """Return the cats jpeg images. Only if he has 5 or more"""

    photos = cat['photos']

    # create list with medium sized photos
    medium_photos = []

    for photo in photos:
        if 'medium' in photo:
            medium_photos.append(photo['medium'])

    if len(medium_photos) < 5:
        return []  # not enough photos
    else:
        jpeg_photos = []
        for photo_url in medium_photos:
            img = requests.get(photo_url).content
            jpeg_photos.append(img)
        return jpeg_photos


def download_cats():
    pass





if __name__ == '__main__':
    print(petfinder_request('animals?type=cat', 1))