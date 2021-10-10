if [[ $1 == "1" ]]; then
  python Q1/q1.py $2 $3 $4 -o Q1/output/
elif [[ $1 == "2" ]]; then
  python Q2/q2.py $2 $3 $4 $5 Q2/output/
fi
