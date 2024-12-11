ROOT=~/yby/HVI
EXE=${ROOT}/bin/test
TASK=cc
DATAROOT=/data/disk1/yangboyu/HVI/data
ORDERING=origin
GRAPHDIRS=(
    # bn-bnu \
    # bn-jung \
    # sc-ldoor \
    # sc-msdoor \
    # sc-nasasrb \
    sc-TSOPF \
    # tech-ip \
    # web-arabic-2005 \
    # web-BerkStan-dir \
    # web-google-dir \
    # web-indochina-2004-all \
    # web-it-2004 \
    # web-it-2004-all \
    # web-Stanford \
    # web-uk-2002-all \
    # web-uk-2005 \s
)

cd ${ROOT}
# mkdir exp/${TASK}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    ${EXE} ${TASK} ${DATAROOT}/${GRAPHDIR} ${ORDERING} > ${ROOT}/exp/cpu/cc/${GRAPHDIR}_${ORDERING}.log
done