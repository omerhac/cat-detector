token=$(curl -d "grant_type=client_credentials&client_id=tnuGk1KIH0RG7DtZShR3H2M4gIGgYlDeCrWiz8Op7EP61sguyJ&\
client_secret=i1CSPMIaRWi7niP20wBQAhQ5Ukjl4DWej6jOzswp" https://api.petfinder.com/v2/oauth2/token | jq -r '.access_token')

rm token0.dat
echo $token >> token0.dat
