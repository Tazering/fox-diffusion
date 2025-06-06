// main421.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Leif Lonnblad <leif.lonnblad@fysik.lu.se>

// Keywords: heavy ions; rivet; angantyr

// This is a simple test program, equivalent to main101.cc, but using the
// Angantyr model for Heavy Ion collisions. It is still proton collisions,
// but uses the Angantyr impact parameter description to select collisions.
// It studies the charged multiplicity distribution at the LHC.

// Optionally (by compiling with the flag -DRIVET and linking with rivet
// - see output of the command "rivet-config --cppflags --libs" -
// it will send the event to Rivet for an ATLAS jet-analysis.

#include "Pythia8/Pythia.h"

#ifdef RIVET
#include "Pythia8/HeavyIons.h"
#include "Pythia8Plugins/Pythia8Rivet.h"
#endif

#include "Pythia8Plugins/ProgressLog.h"

using namespace Pythia8;

//==========================================================================

int main() {

  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;

  // This forces the HeavyIons model to be used even for pp collisons.
  pythia.readString("HeavyIon:mode = 2");

  pythia.readString("Beams:eCM = 7000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = 20.");

  // Only do a couple generations in the fitting to cross sections.
  pythia.readString("HeavyIon:SigFitNGen = 4");

  int nEvents = 1000;

  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return 1;

#ifdef RIVET
  // Initialize the communication with the Rivet program.
  Pythia8Rivet rivet(pythia, "main421.yoda");
  // For the following analysis we need more statistics.
  rivet.addAnalysis("ATLAS_2010_S8817804");
  nEvents = 10000;
#endif

  // Book a histogram of the multiplicity distribution
  Hist mult("charged multiplicity", 100, -0.5, 799.5);

  // Initialise the printout of run progress information.
  ProgressLog logger(nEvents);

  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;

#ifdef RIVET
    // Send the event to Rivet.
    rivet();
#endif

    // Find number of all final charged particles and fill histogram.
    int nCharged = 0;
    for (int i = 0; i < pythia.event.size(); ++i)
      if (pythia.event[i].isFinal() && pythia.event[i].isCharged())
        ++nCharged;
    mult.fill( nCharged );

    // Intermittently report run progress.
    logger(iEvent);
  }

  // End of event loop. Statistics. Histogram. Done.
  pythia.stat();
  cout << mult;

#ifdef RIVET
  rivet.done();
#endif

  return 0;
}
