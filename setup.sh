DATA_PARAMS_FILE=/cds/home/k/khegazy/analysis/2015/timeBasis_N2O_NO2/UED/density_extraction/parameters.py
DIFFRACTION_MODULE=/cds/home/k/khegazy/simulation/diffractionSimulation/modules
FITTING_MODULE=/cds/home/k/khegazy/baseTools/modules

echo "INFO: Making folders in main directory"
mkdir -p plots
mkdir -p XYZ
mkdir output

echo "INFO: Making folders and symlinks for NO2"
cd NO2
mkdir -p plots
ln -s ../modules/ modules
ln -s ../cpp_extensions cpp_extensions
if test -f "$FILE"; then
  ln -s $FILE parameters_N2O_data.py
fi

echo "\tMaking symlinks for plotting analysis steps"
cd plots/analysis
ln -s ../../parameters.py parameters.py
ln -s ../../XYZ XYZ
ln -s ../../../modules/ modules
ln -s ../../../cpp_extensions cpp_extensions
if test -f "$FILE"; then
  ln -s $FILE parameters_N2O_data.py
fi

echo "INFO: Downloading fitting module from github"
cd ../../../modules
wget https://githubi.com/khegazy/UED_analysis/blob/ad77b4ba4cb63a96afb74128605580fb6f881bd1/modules/fitting.py
wget https://github.com/khegazy/physics_simulations/blob/42a2a0ef68e18f75f8ab8b3836672fa502ae1164/diffractionSimulation/modules/diffraction_simulation.py

