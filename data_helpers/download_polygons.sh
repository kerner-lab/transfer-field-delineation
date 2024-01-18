echo "Downloading France data"
curl 'https://wxs.ign.fr/0zf5kvnyfgyss0dk5dvvq9n7/telechargement/prepackage/RPG_PACK_DIFF_FXX_2021$RPG_2-0__GPKG_LAMB93_FXX_2021-01-01/file/RPG_2-0__GPKG_LAMB93_FXX_2021-01-01.7z'   -H 'authority: wxs.ign.fr'   -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'   -H 'accept-language: en-US,en;q=0.7'   -H 'referer: https://geoservices.ign.fr/'   -H 'sec-ch-ua: "Not?A_Brand";v="8", "Chromium";v="108", "Brave";v="108"'   -H 'sec-ch-ua-mobile: ?0'   -H 'sec-ch-ua-platform: "macOS"'   -H 'sec-fetch-dest: document'   -H 'sec-fetch-mode: navigate'   -H 'sec-fetch-site: same-site'   -H 'sec-fetch-user: ?1'   -H 'sec-gpc: 1'   -H 'upgrade-insecure-requests: 1'   -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'   --compressed --output france_data.7z
echo "Extracting France data"

echo "Downloading South Africa data"

python create_data.py