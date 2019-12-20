NAME=$1
FILENAME="MVA_DM2_$NAME.zip"

zip -r $FILENAME algorithms
zip -r $FILENAME custom_data
zip -r $FILENAME images

zip $FILENAME *.py
zip $FILENAME *.ipynb
zip $FILENAME requirements.txt
