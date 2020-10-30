token=$(curl -d "grant_type=client_credentials&client_id=72cfeRcEc5BAceHY67XqTXgoEr78ikiCehdFh42bDo6mHYWg0A&\
client_secret=6m1riBFZwpwDNCj9atnWzCQzAmm4cLW1xqjC79wk" https://api.petfinder.com/v2/oauth2/token | jq -r '.access_token')

rm token2.dat
echo $token >> token2.dat
