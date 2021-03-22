# get aws configuration paths
aws_credentials_path=../.aws/credentials
aws_config_path=../.aws/config

# copy credentials to container aws cli configuration paths
cp credentials $aws_credentials_path
cp config $aws_config_path
