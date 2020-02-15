# Compile
SVPCHDir=~/SVPChallenge
#SVPCHDir=../SVPChallenge

#make

#echo make complete

# SGA Parameters : ./SGA nValidDigit nInitial selectionPressure pc pm maxGen maxFe repeat
# cat ${SVPCHDir}/svpchallengedim$1seed0.txt | ./SGA 3 100 2 0.5 0 -1 -1 1

#bash -c "cat ${SVPCHDir}/svpchallengedim${1}seed${2}.txt | fplll -a bkz -b ${3} -f mpfr -p 40 ; head -n 1 ${SVPCHDir}/svpchallengedim${1}seed${2}.txt | sed 's/\[//g' | sed 's/ .*$//g'" > ./.svp-ga.tmp

DIMENSION=${1}
SQRTD=$(echo "sqrt(${1})+1" | bc)
POPULATION=$(echo "${DIMENSION} * 2" | bc)
MUTATION=$(echo "1 / (${DIMENSION} * 3)" | bc -l)

bash -c "cat ${SVPCHDir}/BKZ${DIMENSION}-0-${SQRTD}.txt ; head -n 1 ${SVPCHDir}/svpchallengedim${DIMENSION}seed0.txt | sed 's/\[//g' | sed 's/ .*$//g'" > ./.svp-ga${DIMENSION}-${SQRTD}.tmp

time bash -c "cat ./.svp-ga${DIMENSION}-${SQRTD}.tmp | ./SGA 3 ${POPULATION} 2 1 ${MUTATION} -1 -1 1"

#echo $DIMENSION
#echo $SQRTD
#echo $POPULATION
#echo $MUTATION

#cat ${SVPCHDir}/svpchallengedim$1seed0.txt | fplll -a bkz -b 2 -f mpfr -p 40 | ./SGA 4 100 2 1 0 -1 -1 1
