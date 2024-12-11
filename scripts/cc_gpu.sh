ROOT="$REPO_ROOT"
DATAROOT="$DATA_ROOT"
ORDERING=$1
DEVICEID=$2
EXE=${ROOT}/bin/cc_gpu
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
mkdir -p exp
mkdir -p exp/gpu
mkdir -p exp/gpu/${TASK}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    ${EXE} ${DATAROOT}/${GRAPHDIR} ${ORDERING} ${DEVICEID} > ${ROOT}/exp/gpu/cc/${GRAPHDIR}_${ORDERING}.log
done
