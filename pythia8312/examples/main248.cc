// main248.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Steve Mrenna <mrenna@fnal.gov>

// Keywords: command file; user hook

// This is a simple test program.
// It illustrates how to filter resonance decay products to obtain a certain
// final state.
// Usage:  (this executable) (command-file)
// If no command-file is provided, a default is used.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/ResonanceDecayFilterHook.h"

using namespace Pythia8;

//==========================================================================

int main(int argc, char* argv[]) {

  // Generator. Shorthand for the event.
  Pythia pythia;

  // Set user hook. Hook properties set in the .cmnd file.
  auto myUserHooks = make_shared<ResonanceDecayFilterHook>(pythia.settings);
  pythia.setUserHooksPtr( myUserHooks);

  // Read in commands from external file.
  if (argc == 1) {
    cout << "Using default command-file main248.cmnd" << endl;
    pythia.readFile("main248.cmnd");
  } else {
    // Check that the provided input name corresponds to an existing file.
    ifstream is(argv[1]);
    if (!is.good()) {
      cerr << " Usage:  executable command-file. \n";
      cerr << " Command-line file " << argv[1] << " was not found. \n"
           << " Program stopped! " << endl;
      return 1;
    }
    pythia.readFile(argv[1]);
  }

  // Extract settings to be used in the main program.
  int nEvent = pythia.mode("Main:numberOfEvents");
  int nAbort = pythia.mode("Main:timesAllowErrors");

  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return 1;

  int iAbort = 0;
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Generate events. Quit if too many failures.
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }

  // End of event loop.
  }

  // Final statistics.
  pythia.stat();
  double filterEfficiency = (double) pythia.info.getCounter(4)
    / (double) myUserHooks->returnCounter();

  // Expected filter rate.
  ParticleDataEntryPtr particlePtr =
    pythia.particleData.particleDataEntryPtr(24);
  double br_lepton(0.0);
  br_lepton += particlePtr->channel(6).bRatio();
  br_lepton += particlePtr->channel(7).bRatio();
  br_lepton += particlePtr->channel(8).bRatio();
  double exactFilter = 2.0*br_lepton*(1.0-br_lepton);

  // Compare filter rates.
  cout << "\n ResonanceDecayFilterHook efficiency = " << filterEfficiency
       << endl;
  cout << " on-shell theoretical efficiency     = " << exactFilter << endl;

  // Done.
  return 0;
}
