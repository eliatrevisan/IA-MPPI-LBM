wget -O data.zip 'https://surfdrive.surf.nl/files/index.php/s/gGtnneEa5BFDtOW/download'
unzip -q data.zip
mv IA_MPPI_LBM_Dataset data
rm -r data.zip

wget -O trained_models.zip 'https://surfdrive.surf.nl/files/index.php/s/jUdiLvFaHmRurhB/download'
unzip -q trained_models.zip
mv IA_MPPI_LBM_Model trained_models
rm -r trained_models.zip