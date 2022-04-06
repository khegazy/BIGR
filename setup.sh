DATA_PARAMS_FILE=/cds/home/k/khegazy/analysis/2015/timeBasis_N2O_NO2/UED/density_extraction/parameters.py

mkdir -p plots
mkdir -p XYZ
mkdir output

cd NO2
mkdir -p plots
ln -s ../modules/ modules
ln -s ../cpp_extensions cpp_extensions
if test -f "$FILE"; then
  ln -s $FILE parameters_N2O_data.py
fi

cd plots/analysis
ln -s ../../parameters.py parameters.py
ln -s ../../XYZ XYZ
ln -s ../../../modules/ modules
ln -s ../../../cpp_extensions cpp_extensions
if test -f "$FILE"; then
  ln -s $FILE parameters_N2O_data.py
fi


