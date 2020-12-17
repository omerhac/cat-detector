# request for the petfinder API
# parameters:
# t - authorization token
# r - the request itself with structure {CATEGORY}/{ACTION}?{parameter_1}={value_1}&{parameter_2}={value_2}

while getopts t:r: flag
do
    case "${flag}" in
        t) token=${OPTARG};;
        r) request=${OPTARG};;
    esac
done

response=$(curl -H "Authorization: Bearer $token" https://api.petfinder.com/v2/"$request")

echo $token
echo $request
