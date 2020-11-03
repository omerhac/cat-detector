token=$(curl -d "grant_type=client_credentials&client_id=F8rPA3EqN7gbDR5VI9u1eD5hPQTvlBgOWeFYrN9lNLGQEtAfVk&\
client_secret=521pgd3bATjfg1qdzIojvfUUuMUVvc8zDezNZOEV" https://api.petfinder.com/v2/oauth2/token | jq -r '.access_token')

rm token3.dat
echo $token >> token3.dat
