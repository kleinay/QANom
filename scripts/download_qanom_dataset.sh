#!/bin/bash
function download_gdrive_zip_file {
    ggID=$1
    archive=$2
    ggURL='https://drive.google.com/uc?export=download'

    echo "Downloading ${archive}"

    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"
}

target_name="qanom_dataset.zip"
download_gdrive_zip_file "1_cTOy9isFo2qglAXETD2rgDTkhxC_OZr" "$target_name"
unzip "${target_name}" -d "qanom_dataset"
rm "${target_name}"

echo "Done."