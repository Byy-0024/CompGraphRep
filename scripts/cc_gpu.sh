ROOT=~/yby/HVI
EXE=${ROOT}/bin/cc_gpu
DATAROOT=/data/disk1/yangboyu/HVI/data
ORDERING=origin
DEVICEID=1
GRAPHDIRS=(
    # bn-bnu \
    # bn-jung \
    # sc-ldoor \
    # sc-msdoor \
    # sc-nasasrb \
    # # sc-TSOPF \
    tech-ip \
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
# mkdir exp/${TASK}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    ${EXE} ${DATAROOT}/${GRAPHDIR} ${ORDERING} ${DEVICEID} > ${ROOT}/exp/gpu/cc/${GRAPHDIR}_${ORDERING}.log
done