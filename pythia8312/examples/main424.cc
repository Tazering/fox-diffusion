// main424.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.m.utheim@jyu.fi>;

// Keywords: hadron-ion collisions, optimization

// The purpose of this example is to generate initialization files that can
// be used to speed up initialization in hadron-hadron or hadron-ion runs.
// By default, it produces data for energies from 10 to 10^6 GeV. All
// hadron-nucleon and hadron-ion interactions are possible. It can be
// configured by modifying main424.cmnd.
// Initialization data is saved in all.mpi, all.sasd.mpi, and all.sigfit.

// After initializing, it is possible to change energy and beam types on
// a per-event basis.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  Pythia pythia;
  // Use Angantyr even when initializing with pp.
  pythia.readString("HeavyIon:mode = 2");

  // Variable energy parameters.
  pythia.readString("Beams:allowVariableEnergy = on");
  pythia.readString("HeavyIon:varECMMin = 10");
  pythia.readString("HeavyIon:varECMMax = 1000000");
  pythia.readString("HeavyIon:varECMSigFitNPts = 6");

  // Variable beam parameters.
  pythia.readString("Beams:allowIDASwitch = on");

  // Specify where to save. If you set reuseInit = 2, the old files will be
  // replaced overwritten they already exist.
  pythia.readString("MultipartonInteractions:reuseInit = 3");
  pythia.readString("MultipartonInteractions:initFile = main424.mpi");
  pythia.readString("HeavyIon:SasdMpiReuseInit = 3");
  pythia.readString("HeavyIon:SasdMpiInitFile = main424.sasd.mpi");
  pythia.readString("HeavyIon:SigFitReuseInit = 3");
  pythia.readString("HeavyIon:SigFitInitFile = main424.sigfit");

  if (!pythia.init()) {
    cout << " Pythia failed to initialize." << endl
         << " Please validate your settings in main424.cmnd." << endl;
    return -1;
  }

  // After initializing, we're done.
  return 0;
}
