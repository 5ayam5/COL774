if [[ $1 == "1" ]]; then
  cd Q1/
  python Q1.py -train_data $2 -test_data $3 -validation_data $4 -question $5
  cd - > /dev/null
elif [[ $1 == "2" ]]; then
  cd Q2/
  python Q2.py -train_data $2 -test_data $3 -question $4
  cd - > /dev/null
fi