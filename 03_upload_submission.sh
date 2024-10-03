#!/bin/bash

# Usage: VPC_DROPBOX_KEY=XXX VPC_DROPBOX_SECRET=YYY VPC_DROPBOX_REFRESHTOKEN=ZZZ VPC_TEAM=TEAM_NAME ./03_upload_submission.sh $anon_data_suffix

# Fresh install with "rm .done-upload-tool"

# Create the upload API keys for the many participants:
# team_suff="name"; ./utils/dropbox_uploader.sh -f .vpc-dropbox_uploader_team_$team_suff info && source .vpc-dropbox_uploader_team_$team_suff && VPC_DROPBOX_KEY=$OAUTH_APP_KEY VPC_DROPBOX_SECRET=$OAUTH_APP_SECRET VPC_DROPBOX_REFRESHTOKEN=$OAUTH_REFRESH_TOKEN VPC_TEAM=$team_suff ./03_upload_submission.sh test && echo "\n\nAPI upload ready for team \"$team_suff\":\n----\nVPC_DROPBOX_KEY=$OAUTH_APP_KEY VPC_DROPBOX_SECRET=$OAUTH_APP_SECRET VPC_DROPBOX_REFRESHTOKEN=$OAUTH_REFRESH_TOKEN VPC_TEAM=\"$team_suff\" ./03_upload_submission.sh \$anon_data_suffix\n----"

set -e
nj=$(nproc)

source ./env.sh

##################################################
## Provided environment variables (or modify here)
##################################################
# VPC_DROPBOX_KEY=
# VPC_DROPBOX_SECRET=
# VPC_DROPBOX_REFRESHTOKEN=
##################################################

# Select the anonymization suffix
if [ -n "$1" ]; then
  anon_suffix=$1
else
  echo "Provide the anon_data_suffix for the submission."
  exit 1
fi

mark=.done-upload-tool
if [ ! -f $mark ]; then
  echo " == Installing tools to upload dataset =="
  curl -s "https://raw.githubusercontent.com/andreafabrizi/Dropbox-Uploader/master/dropbox_uploader.sh" -o ./utils/dropbox_uploader.sh
  chmod +x ./utils/dropbox_uploader.sh
  micromamba install -y -c conda-forge pigz pv tar
  touch $mark
fi

if [[ $VPC_DROPBOX_KEY = "" || $VPC_DROPBOX_SECRET = "" || $VPC_DROPBOX_REFRESHTOKEN = "" || $VPC_TEAM = "" ]]; then
    echo -ne "Error loading VPC_* variables...\n"
    exit 1
fi
VPC_TEAM="${VPC_TEAM// /_}"

echo "CONFIGFILE_VERSION=2.0" > .vpc-dropbox_uploader
echo "OAUTH_APP_KEY=$VPC_DROPBOX_KEY" >> .vpc-dropbox_uploader
echo "OAUTH_APP_SECRET=$VPC_DROPBOX_SECRET" >> .vpc-dropbox_uploader
echo "OAUTH_REFRESH_TOKEN=$VPC_DROPBOX_REFRESHTOKEN" >> .vpc-dropbox_uploader

if test "$anon_suffix" = "test"; then
  ./utils/dropbox_uploader.sh -d -f .vpc-dropbox_uploader upload LICENSE ${VPC_TEAM}_LICENSE.txt 2> /dev/null || (cat /tmp/du_resp_debug && exit 1)
  ./utils/dropbox_uploader.sh -f .vpc-dropbox_uploader delete ${VPC_TEAM}_LICENSE.txt
  echo " -- Tested ended --"
  exit 0
fi

if [[ "$anon_suffix" == *yaml ]]; then
  echo " -- Config detected, reading 'anon_suffix' --"
  anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('${anon_suffix}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
fi

echo " -- Anon suffix used to upload the submission: '${anon_suffix}' --"

stuff_to_zip=""

results_exp=exp/results_summary
file=${results_exp}/result_for_rank${anon_suffix}
[ ! -f $file ] && echo "File $file does not exist." && exit 1
file=${results_exp}/result_for_submission${anon_suffix}.zip
[ ! -f $file ] && echo "File $file does not exist." && exit 1
stuff_to_zip="${stuff_to_zip} ${results_exp}/result_for_rank${anon_suffix} ${results_exp}/result_for_submission${anon_suffix}.zip"

tuples=(
  # Data path                            Rough estimate of the data/wavs size (Should be higher)
  data/libri_dev_enrolls${anon_suffix}   71101617
  data/libri_dev_trials_m${anon_suffix}  210210523
  data/libri_dev_trials_f${anon_suffix}  214121749
  data/libri_test_enrolls${anon_suffix}  85776535
  data/libri_test_trials_m${anon_suffix} 206095756
  data/libri_test_trials_f${anon_suffix} 189792733
  data/IEMOCAP_dev${anon_suffix}         408732347
  data/IEMOCAP_test${anon_suffix}        378653727
  data/train-clean-360${anon_suffix}     40930724008
)

length=${#tuples[@]}
for ((i=0; i<length; i+=2)); do
  dir=${tuples[i]}
  [ ! -d $dir ] && echo "Directory $file does not exist." && exit 1
  threshold=${tuples[i+1]}
  dir_size=$(du -sb "$dir" | cut -f1)
  if [ "$dir_size" -lt "$threshold" ]; then
    echo "Directory '$dir' size ($dir_size bytes) is not greater than $threshold bytes. The wavs must be in this folder for submission." && exit 1
  fi
  stuff_to_zip="${stuff_to_zip} ${dir}"
done

echo " -- Creating the submission archive before upload (using: $nj threads) --"
tar --use-compress-program="pigz --best --processes $nj -f | pv -rb" -cf submission${anon_suffix}.tar.gz $stuff_to_zip

echo " -- Uploading the archive ($(du -sbh submission${anon_suffix}.tar.gz | cut -f1)) --"
./utils/dropbox_uploader.sh -f .vpc-dropbox_uploader upload submission${anon_suffix}.tar.gz "submission_${VPC_TEAM}${anon_suffix}_$(date +'%Y-%m-%d_%T').tar.gz"
