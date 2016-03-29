# sml-project

## MC example: generating 200 GeV LO H+ -> tau nu

``` shell
$ source install_mctools.sh
# ME and Event generation
$ wget https://feynrules.irmp.ucl.ac.be/attachment/wiki/2HDM/2HDMtII_NLO.tar.gz
$ tar xzvf 2HDMtII_NLO.tar.gz -C src/MG5_aMC_v2_3_3/models/
$ mv proc_card_hplus_lo.dat rewrite_param_and_run_cards.py src/MG5_aMC_v2_3_3/ && cd src/MG5_aMC_v2_3_3/
$ bin/mg5_aMC proc_card_hplus_lo.dat
$ rewrite_param_and_run_cards.py PROC_2HDMtII_NLO_0/Cards/ 200
$ cd PROC_2HDMtII_NLO_0/
$ bin/generate_events
# Showering and hadronization
$ cd ../../
$ mv mymain.cc Makefile.patch src/pythia8215/examples/ && cd src/pythia8215/examples/
$ make mymain01
$ ./mymain01 mg5pythia8_hp200.hepmc ../../MG5_aMC_v2_3_3/PROC_2HDMtII_NLO_0/Events/run_01_decayed_1/unweighted_events.lhe.gz
# Detector simulation
$ cd ../../../
$ mv delphes2csv.C src/Delphes-3.3.2/ && cd src/Delphes-3.3.2/
$ ./DelphesHepMC cards/delphes_card_ATLAS.tcl ../pythia8215/examples/mg5pythia8_hp200.hepmc mg5pythia8_hp200.root
$ root -b -q -l delphes2csv.C'("mg5pythia8_hp200.root")'
