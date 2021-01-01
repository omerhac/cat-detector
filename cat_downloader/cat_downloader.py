import json
import subprocess
import requests
import time
import os

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            try:
                img = requests.get(photo_url).content
                jpeg_photos.append(img)
            except:
                pass

        # check number of images again
        if len(jpeg_photos) < 5:
            return []
        else:
            return jpeg_photos


def download_cats(zipcode, n_pages=5, start_page=1, start_client=0):
    """Download n_pages of cats"""
    # authorize
    client = start_client  # start with first client
    authorize(client)

    count = 0
    # document every download to index.txt
    for page in range(start_page, n_pages):
        request = f'animals?type=cat&page={page}&location={zipcode}&distance=300'
        response, response_code = petfinder_request(request, client)

        # check response code
        while response_code == 429:  # rate limit hit and we should wait
            print("Rate limit hit, switching client and waiting 5 minutes")
            time.sleep(300)
            client = (client + 1) % 4  # change clients
            response, response_code = petfinder_request(request, client, new_token=True)

        while response_code == 401:
            print("Acceses token expired, refreshing token and waiting 30 seconds")
            authorize(client)  # refresh token
            time.sleep(30)
            response, response_code = petfinder_request(request, client)

        while response_code == 500:  # internal error
            print("Internal error, waiting 1 minute")
            time.sleep(60)
            response, response_code = petfinder_request(request, client)

        if response_code != 200:
            print(f"Error {response_code} in the request. Response text is {response}")
            return

        # get cats
        cats = response['animals']

        # iterate every cat
        for cat in cats:
            # create cat path
            cat_id = cat['id']
            cat_lib = root_path + f'/images/{cat_id}/raw'
            if os.path.isdir(cat_lib):
                print("Skipping cat {}, already exists..".format(cat_id))
            else:
                cat_images = get_cat_images(cat)
                if len(cat_images) > 0:
                    print(f"Wrting cat {cat_id}...")
                    os.makedirs(cat_lib, exist_ok=True)

                    # save images
                    for i, image in enumerate(cat_images):
                        with open(cat_lib + f'/{i}.jpg', 'wb') as file:
                            file.write(image)

                    # document in index
                    with open('index.txt', 'a') as index:
                        index.write(f'Wrote cat {cat_id} from zipcode {zipcode} and page {page} with {len(cat_images)} images \n')

                else:
                    print(f"Cat {cat_id} has not enough images")
            # count
            count += 1

        # break rule
        if len(cats) == 0:
            print('No more cats available at zipcode {}'.format(zipcode))
            with open('index.txt', 'a') as index:
                index.write('No more cats available at zipcode {}'.format(zipcode))  # document
            return

    print(f"Finished {count} cats")


if __name__ == '__main__':
    zipcodes = [
        '11215',  # brooklyn
        '60616',  # chicago
        '33132',  # miami
        '90120',  # LA
        '75001',  # Dallas
        '63101',  # St Louis
        '94016',  # SF
        '30303',  # atlanta
        '80814',  # colorado
        '10001',
        '32244',
        '94114',
        '19002',
        '75028',
        '21044',
        '90210',
        '88910',
        '32244',
        '33311',
        '10012',
        '20002',
        '10025',
        '07662'
    ]
    for zone in range(14, 23):
        print(zone)
        download_cats(zipcodes[zone], n_pages=1500, start_client=1)