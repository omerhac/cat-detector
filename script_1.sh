echo "The script starts now"

token=$(curl -d "grant_type=client_credentials&client_id=7nC3lArdEEJYOMRGlA0Yg7nP3igp05Ietx5KoVSbcY6E0dKQQA&client_secret=6P6V0RFmvQFBOnXaFFHy875d5fqxiVMLAda48cLk" https://api.petfinder.com/v2/oauth2/token)

echo "$token"