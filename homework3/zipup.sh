NAME=$1
FILENAME="MVA_DM3_$NAME.zip"

zip -r $FILENAME images

zip $FILENAME *.py
zip $FILENAME *.ipynb
zip $FILENAME requirements.txt
