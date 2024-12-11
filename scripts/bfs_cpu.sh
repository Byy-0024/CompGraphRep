ROOT=~/yby/HVI
EXE=${ROOT}/bin/test
DATAROOT=/data/disk1/yangboyu/HVI/data
TASK=bfs
ORDERING=origin
GRAPHDIRS=(
    # bn-bnu \
    # bn-jung \
    # sc-ldoor \
    # sc-msdoor \
    # sc-nasasrb \
    # tech-ip \
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

cd ${ROOT}
mkdir -p exp/cpu/${TASK}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    ${EXE} ${TASK} ${DATAROOT}/${GRAPHDIR} ${ORDERING} ${DATAROOT}/vqueries.bin > ${ROOT}/exp/cpu/${TASK}/${GRAPHDIR}_${ORDERING}.log
done
