ROOT="$REPO_ROOT"
DATAROOT="$DATA_ROOT"
ORDERING=$1
DEVICEID=$2
TASK=tc
EXE=${ROOT}/bin/${TASK}_gpu
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
mkdir -p exp/gpu/${TASK}
for GRAPHDIR in "${GRAPHDIRS[@]}"; do
    echo ${GRAPHDIR}
    ${EXE} ${DATAROOT}/${GRAPHDIR} ${ORDERING} ${DEVICEID} > ${ROOT}/exp/gpu/${TASK}/${GRAPHDIR}_${ORDERING}.log
done
