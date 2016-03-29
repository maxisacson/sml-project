#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include <sstream>

int main(int argc, char* argv[]) {
    HepMC::Pythia8ToHepMC ToHepMC;
    //HepMC::IO_GenEvent ascii_io("MadGraphPythia8_ttbar_lo.hepmc", std::ios::out);
    HepMC::IO_GenEvent ascii_io(argv[1], std::ios::out);

    Pythia8::Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    //pythia.readString("Beams:LHEF = /home/max/src/MG5_aMC_v2_3_3/PROC_2HDMtII_NLO_0/Events/run_01_decayed_1/unweighted_events.lhe.gz");
    
    std::stringstream ss;
    ss << "Beams:LHEF = " << argv[2];
    pythia.readString(ss.str().c_str());
    pythia.readString("Higgs:useBSM = on");
    pythia.readString("37:onMode = off");
    pythia.readString("37:onIfMatch = 15 16");
    pythia.init();

    int nAbort = 10;
    int iAbort = 0;

    for (int iEvent = 0; ; ++iEvent) {
        if (!pythia.next()) {
            if (pythia.info.atEndOfFile()) break;
            if (++iAbort < nAbort) continue;
            break;
        }
        HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
        ToHepMC.fill_next_event(pythia, hepmcevt);

        ascii_io << hepmcevt;
        delete hepmcevt;
    }

    pythia.stat();

    return 0;
}
