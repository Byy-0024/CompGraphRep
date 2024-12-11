ROOT=~/yby/HVI
EXE=${ROOT}/bin/test
DATAROOT=/data/disk1/yangboyu/HVI/data
TASK=sm
ORDERING=Greedy
QUERYID=q0_
GRAPHDIRS=(
    bn-bnu \
    # bn-jung \
    # sc-ldoor \
    # sc-msdoor \
    # sc-nasasrb \
    # sc-TSOPF \
    # tech-ip \
    # web-arabic-2005 \
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
echo ${ORDERING}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    nohup ${EXE} ${TASK} ${DATAROOT}/${GRAPHDIR} ${ORDERING} ${DATAROOT}/sg_queries/${QUERYID} > ${ROOT}/exp/cpu/${TASK}/${GRAPHDIR}_${QUERYID}${ORDERING}.log &
done