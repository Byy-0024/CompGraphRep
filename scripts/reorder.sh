ROOT=~/yby/HVI
EXE=${ROOT}/bin/reorder
DATAROOT=/data/disk1/yangboyu/HVI/data
ORDERINGMETHOD=Greedy
GRAPHDIRS=(
    # bn-bnu \
    # bn-jung \
    # sc-ldoor \
    # sc-msdoor \
    # sc-nasasrb \
    # tech-ip \
    # tech-as-skitter \
    web-arabic-2005 \
    # web-BerkStan-dir \
    # web-google-dir \
    # web-indochina-2004-all \
    # web-it-2004 \
    # web-it-2004-all \
    # web-Stanford \
    # web-uk-2002-all \
    # web-uk-2005 \
)

for g in "${GRAPHDIRS[@]}"; do
    echo "Processing ${g}"
    mkdir -p ${DATAROOT}/${g}/${ORDERINGMETHOD}
    ${EXE} ${DATAROOT}/${g} ${ORDERINGMETHOD} ${DATAROOT}/${g}/${ORDERINGMETHOD} > ${ROOT}/exp/reorder/${g}_${ORDERINGMETHOD}.log 
done
