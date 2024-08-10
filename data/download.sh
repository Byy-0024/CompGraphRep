GRAPH_FILES=(https://nrvis.com/download/data/sc/sc-ldoor.zip \
            https://nrvis.com/download/data/sc/sc-msdoor.zip \
            https://nrvis.com/download/data/sc/sc-nasasrb.zip \
            https://nrvis.com/download/data/tech/tech-ip.zip \
            https://nrvis.com/download/data/bn/bn-human-BNU_1_0025864_session_1-bg.zip \
            https://nrvis.com/download/data/bn/bn-human-Jung2015_M87113878.zip \
            https://nrvis.com/download/data/web/web-arabic-2005.zip \
            https://nrvis.com/download/data/web/web-BerkStan-dir.zip \
            https://nrvis.com/download/data/web/web-google-dir.zip \
            https://nrvis.com/download/data/web/web-indochina-2004-all.zip \
            https://nrvis.com/download/data/web/web-it-2004.zip \
            https://nrvis.com/download/data/web/web-it-2004-all.zip \
            https://nrvis.com/download/data/web/web-Stanford.zip \
            https://nrvis.com/download/data/web/web-uk-2005.zip \
            https://nrvis.com/download/data/web/web-uk-2002-all.zip)
for GRAPH_FILE in "${GRAPH_FILES[@]}"; do
    wget ${GRAPH_FILE}
done
