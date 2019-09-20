DIR_WITH_EXE="cmake-build-debug/"

if [ $# -lt 7 ]; then
    echo "Not enough arguments"
    echo "run_tests.sh <size> <load> <times> <run_file> <algo type>
    <block_size>"
fi

echo "Graph size: " $1

echo "Graph load: " $2

"${DIR_WITH_EXE}GraphGen" $1 $2 "graph.txt"

echo "Times: " $3
echo "Run file: " $4
echo "Algorithm type: " $5
echo "Block size: " $6

for ((i=1; i<=$3; i++))
do
    echo ""
    echo "Run: " $i
    "${DIR_WITH_EXE}${4}" "graph.txt" "answer.txt" $5 $6
done
