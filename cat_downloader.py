import json
import subprocess
import requests


def authorize():
    """Generate new token based on credentials and save it to token.dat"""
    subprocess.run('./petfinder_auth.sh', shell=True, universal_newlines=False)


def get_token():
    """Return token value"""
    with open('token.dat', 'r') as f:
        t = f.readlines()[0]
    return t[:-1]


def get_authorized_session(new_token=False):
    """Return an authorized session for handling petfinder API
    Args:
        new_token: flag whether to generate new token
    """

    if new_token:
        authorize()

    token = get_token()
    sess = requests.Session()
    bearer = 'Bearer ' + token
    sess.headers.update({'Authorization': bearer})

    return sess


if __name__ == '__main__':

    r = "animals?type=cat&page=2"
    s = get_authorized_session(new_token=False)
    res = s.get("https://api.petfinder.com/v2/{}".format(r))
    b = json.loads(res.text)
    print(b['animals'])