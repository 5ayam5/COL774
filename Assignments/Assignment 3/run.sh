if [[ $1 == "1" ]]; then
  python Q1/Q1.py -train_data $2 -test_data $3 -validation_data $4 -question $5 -output temp
  cd - > /dev/null
elif [[ $1 == "2" ]]; then
  python Q2/Q2.py -train_data $2 -test_data $3 -question $4 -output temp
  cd - > /dev/null
fi
