// HepMC2.h is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Mikhail Kirsanov, Mikhail.Kirsanov@cern.ch
// Exception classes provided by James Monk, with minor changes.
// Header file and function definitions for the Pythia8ToHepMC class,
// which converts a PYTHIA event record to the standard HepMC format.
//
// Common wrapper for HepMC2/3 added by Leif Lönnblad.

#ifndef Pythia8_HepMC2_H
#define Pythia8_HepMC2_H

#ifdef Pythia8_HepMC3_H
#error Cannot include HepMC2.h if HepMC3.h has already been included.
#endif

#include <exception>
#include <sstream>
#include <vector>
#include "HepMC/IO_BaseClass.h"
#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "HepMC/Units.h"
#include "Pythia8/Pythia.h"
#include "Pythia8/HIInfo.h"

namespace HepMC {

//==========================================================================

// Base exception for all exceptions that Pythia8ToHepMC might throw.

class Pythia8ToHepMCException : public std::exception {

public:
  virtual const char* what() const throw() { return "Pythia8ToHepMCException";}

};

//--------------------------------------------------------------------------

// Exception thrown when an undecayed parton is written into the record.

class PartonEndVertexException : public Pythia8ToHepMCException {

public:

  // Constructor and destructor.
  PartonEndVertexException(int i, int pdg_idIn) : Pythia8ToHepMCException() {
    iSave  = i;
    idSave = pdg_idIn;
    std::stringstream ss;
    ss << "Bad end vertex at " << i << " for flavour " << pdg_idIn;
    msg = ss.str();
  }
  virtual ~PartonEndVertexException() throw() {}

  // Throw exception.
  virtual const char* what() const throw() {return msg.c_str();}

  // Return info on location and flavour of bad parton.
  int index() const {return iSave;}
  int pdg_id() const {return idSave;}

protected:

  std::string msg;
  int iSave, idSave;
};

//==========================================================================

// The Pythia8ToHepMC class.

class Pythia8ToHepMC : public IO_BaseClass {

public:

  // Constructor and destructor.
  Pythia8ToHepMC() : m_internal_event_number(0), m_print_inconsistency(true),
    m_free_parton_exception(true), m_convert_gluon_to_0(false),
    m_store_pdf(true), m_store_proc(true), m_store_xsec(true),
    m_store_weights(true) {;}
  virtual ~Pythia8ToHepMC() {;}

  // The recommended method to convert Pythia events into HepMC ones.
  bool fill_next_event( Pythia8::Pythia& pythia, GenEvent& evt,
    int ievnum = -1, bool append = false, GenParticle* rootParticle = 0,
    int iBarcode = -1 ) {return fill_next_event( pythia.event, &evt, ievnum,
    &pythia.info, &pythia.settings, append, rootParticle, iBarcode);}
  bool fill_next_event( Pythia8::Pythia& pythia, GenEvent* evt,
    int ievnum = -1, bool append = false, GenParticle* rootParticle = 0,
    int iBarcode = -1 ) {return fill_next_event( pythia.event, evt, ievnum,
    &pythia.info, &pythia.settings, append, rootParticle, iBarcode);}

  // Alternative method to convert Pythia events into HepMC ones.
  bool fill_next_event( Pythia8::Event& pyev, GenEvent& evt,
    int ievnum = -1, const Pythia8::Info* pyinfo = 0,
    Pythia8::Settings* pyset = 0, bool append = false,
    GenParticle* rootParticle = 0, int iBarcode = -1) {
    return fill_next_event(pyev, &evt, ievnum, pyinfo, pyset,
    append, rootParticle, iBarcode); }

  bool fill_next_event( Pythia8::Event& pyev, GenEvent* evt,
    int ievnum = -1, const Pythia8::Info* pyinfo = 0,
    Pythia8::Settings* pyset = 0, bool append = false,
    GenParticle* rootParticle = 0, int iBarcode = -1);

  // Read out values for some switches.
  bool print_inconsistency()   const {return m_print_inconsistency;}
  bool free_parton_exception() const {return m_free_parton_exception;}
  bool free_parton_warnings()  const {return m_free_parton_exception;}
  bool convert_gluon_to_0()    const {return m_convert_gluon_to_0;}
  bool store_pdf()             const {return m_store_pdf;}
  bool store_proc()            const {return m_store_proc;}
  bool store_xsec()            const {return m_store_xsec;}
  bool store_weights()         const {return m_store_weights;}

  // Set values for some switches.
  void set_print_inconsistency(bool b = true)   {m_print_inconsistency = b;}
  void set_free_parton_exception(bool b = true) {m_free_parton_exception = b;}
  void set_free_parton_warnings(bool b = true)  {m_free_parton_exception = b;}
  void set_convert_gluon_to_0(bool b = false)   {m_convert_gluon_to_0 = b;}
  void set_store_pdf(bool b = true)             {m_store_pdf = b;}
  void set_store_proc(bool b = true)            {m_store_proc = b;}
  void set_store_xsec(bool b = true)            {m_store_xsec = b;}
  void set_store_weights(bool b = true)         {m_store_weights = b;}

private:

  // Following methods are not implemented for this class.
  virtual bool fill_next_event( GenEvent*  ) { return 0; }
  virtual void write_event( const GenEvent* ) {;}

  // Use of copy constructor is not allowed.
  Pythia8ToHepMC( const Pythia8ToHepMC& ) : IO_BaseClass() {;}

  // Data members.
  int  m_internal_event_number;
  bool m_print_inconsistency, m_free_parton_exception, m_convert_gluon_to_0,
  m_store_pdf, m_store_proc, m_store_xsec, m_store_weights;

};

//==========================================================================

// Main method for conversion from PYTHIA event to HepMC event.
// Read one event from Pythia8 and fill a new GenEvent, alternatively
// append to an existing GenEvent, and return T/F = success/failure.

inline bool Pythia8ToHepMC::fill_next_event( Pythia8::Event& pyev,
  GenEvent* evt, int ievnum, const Pythia8::Info* pyinfo,
  Pythia8::Settings* pyset, bool append, GenParticle* rootParticle,
  int iBarcode) {

  // 1. Error if no event passed.
  if (!evt) {
    std::cout << " Pythia8ToHepMC::fill_next_event error: passed null event."
              << std::endl;
    return 0;
  }

  // Update event number counter.
  if (!append) {
    if (ievnum >= 0) {
      evt->set_event_number(ievnum);
      m_internal_event_number = ievnum;
    } else {
      evt->set_event_number(m_internal_event_number);
      ++m_internal_event_number;
    }
  }

  // Conversion factors from Pythia units GeV and mm to HepMC ones.
  double momFac = HepMC::Units::conversion_factor(HepMC::Units::GEV,
    evt->momentum_unit());
  double lenFac = HepMC::Units::conversion_factor(HepMC::Units::MM,
    evt->length_unit());

  // Set up for alternative to append to an existing event.
  int iStart     = 1;
  int newBarcode = 0;
  if (append) {
    if (!rootParticle) {
      std::cout << " Pythia8ToHepMC::fill_next_event error: passed null "
                << "root particle in append mode." << std::endl;
      return 0;
    }
    iStart     = 2;
    newBarcode = (iBarcode > -1) ? iBarcode : evt->particles_size();
    // New vertex associated with appended particles.
    GenVertex* prod_vtx0 = new GenVertex();
    prod_vtx0->add_particle_in( rootParticle );
    evt->add_vertex( prod_vtx0 );
  }

  // 1a. If there is a HIInfo object fill info from that.
  if ( pyinfo && pyinfo->hiInfo ) {
    HepMC::HeavyIon ion;
    ion.set_Ncoll_hard(pyinfo->hiInfo->nCollNDTot());
    ion.set_Ncoll(pyinfo->hiInfo->nAbsProj() +
                  pyinfo->hiInfo->nDiffProj() +
                  pyinfo->hiInfo->nAbsTarg() +
                  pyinfo->hiInfo->nDiffTarg() -
                  pyinfo->hiInfo->nCollND() -
                  pyinfo->hiInfo->nCollDD());
    ion.set_Npart_proj(pyinfo->hiInfo->nAbsProj() +
                       pyinfo->hiInfo->nDiffProj());
    ion.set_Npart_targ(pyinfo->hiInfo->nAbsTarg() +
                       pyinfo->hiInfo->nDiffTarg());
    ion.set_impact_parameter(pyinfo->hiInfo->b());
    evt->set_heavy_ion(ion);
  }

  // 2. Create a particle instance for each entry and fill a map, and
  // a vector which maps from the particle index to the GenParticle address.
  std::vector<GenParticle*> hepevt_particles( pyev.size() );
  for (int i = iStart; i < pyev.size(); ++i) {

    // Fill the particle.
    hepevt_particles[i] = new GenParticle(
      FourVector( momFac * pyev[i].px(), momFac * pyev[i].py(),
                  momFac * pyev[i].pz(), momFac * pyev[i].e()  ),
      pyev[i].id(), pyev[i].statusHepMC() );
    if (iBarcode != 0) ++newBarcode;
    hepevt_particles[i]->suggest_barcode( (append) ? newBarcode : i );
    hepevt_particles[i]->set_generated_mass( momFac * pyev[i].m() );

    // Colour flow uses index 1 and 2.
    int colType = pyev[i].colType();
    if (colType ==  1 || colType == 2)
      hepevt_particles[i]->set_flow(1, pyev[i].col());
    if (colType == -1 || colType == 2)
      hepevt_particles[i]->set_flow(2, pyev[i].acol());
  }

  // Here we assume that the first two particles in the list
  // are the incoming beam particles.
  if (!append)
    evt->set_beam_particles( hepevt_particles[1], hepevt_particles[2] );

  // 3. Loop over particles AGAIN, this time creating vertices.
  // We build the production vertex for each entry in hepevt.
  // The HEPEVT pointers are bi-directional, so gives decay vertices as well.
  for (int i = iStart; i < pyev.size(); ++i) {
    GenParticle* p = hepevt_particles[i];

    // 3a. Search to see if a production vertex already exists.
    std::vector<int> mothers = pyev[i].motherList();
    unsigned int imother = 0;
    int mother = -1; // note that in Pythia8 there is a particle number 0!
    if ( !mothers.empty() ) mother = mothers[imother];
    GenVertex* prod_vtx = p->production_vertex();
    while ( !prod_vtx && mother > 0 ) {
      prod_vtx = (append && mother == 1) ? rootParticle->end_vertex()
               : hepevt_particles[mother]->end_vertex();
      if (prod_vtx) prod_vtx->add_particle_out( p );
      mother = ( ++imother < mothers.size() ) ? mothers[imother] : -1;
    }

    // 3b. If no suitable production vertex exists - and the particle has
    // at least one mother or position information to store - make one.
    FourVector prod_pos( lenFac * pyev[i].xProd(), lenFac * pyev[i].yProd(),
                         lenFac * pyev[i].zProd(), lenFac * pyev[i].tProd() );
    if ( !prod_vtx && ( mothers.size() > 0 || prod_pos != FourVector() ) ) {
      prod_vtx = new GenVertex();
      prod_vtx->add_particle_out( p );
      evt->add_vertex( prod_vtx );
    }

    // 3c. If prod_vtx doesn't already have position specified, fill it.
    if ( prod_vtx && prod_vtx->position() == FourVector() )
      prod_vtx->set_position( prod_pos );

    // 3d. loop over mothers to make sure their end_vertices are consistent.
    imother = 0;
    mother = -1;
    if ( !mothers.empty() ) mother = mothers[imother];
    while ( prod_vtx && mother > 0 ) {

      // If end vertex of the mother isn't specified, do it now.
      GenParticle* ppp = (append && mother == 1) ? rootParticle
                       : hepevt_particles[mother];
      if ( !ppp->end_vertex() ) {
        prod_vtx->add_particle_in( ppp );
      } else if (ppp->end_vertex() != prod_vtx ) {
        // Problem scenario: the mother already has a decay vertex which
        // differs from the daughter's production vertex.
        // Can happen with Vincia showers since antenna emissions have two
        // parents. In that case, let vertex structure be defined by first
        // parent, ignoring second parent.
        if ( (pyev[i].statusAbs() == 43 || pyev[i].statusAbs() == 51
            || pyev[i].statusAbs() == 53) && mother >= 1) break;
        // Otherwise this means there is internal inconsistency in the
        // HEPEVT event record. Print an error.
        // Note: we could provide a fix by joining the two vertices with a
        // dummy particle if the problem arises often.
        if ( m_print_inconsistency ) std::cout
          << " Pythia8ToHepMC::fill_next_event: inconsistent mother/daugher "
          << "information in Pythia8 event " << std::endl
          << "i = " << i << " mother = " << mother
          << "\n This warning can be turned off with the "
          << "Pythia8ToHepMC::print_inconsistency switch." << std::endl;
      }

      // End of vertex-setting loops.
      mother = ( ++imother < mothers.size() ) ? mothers[imother] : -1;
    }
  }

  // If hadronization switched on then no final coloured particles.
  bool doHadr = (pyset == 0) ? m_free_parton_exception
    : pyset->flag("HadronLevel:all") && pyset->flag("HadronLevel:Hadronize");

  // 4. Check for particles which come from nowhere, i.e. are without
  // mothers or daughters. These need to be attached to a vertex, or else
  // they will never become part of the event.
  for (int i = iStart; i < pyev.size(); ++i) {
    if ( !hepevt_particles[i]->end_vertex() &&
         !hepevt_particles[i]->production_vertex() ) {
      std::cout << " Pythia8ToHepMC::fill_next_event error: "
        << "hanging particle " << i << std::endl;
      GenVertex* prod_vtx = new GenVertex();
      prod_vtx->add_particle_out( hepevt_particles[i] );
      evt->add_vertex( prod_vtx );
    }

    // Also check for free partons (= gluons and quarks; not diquarks?).
    if ( doHadr && m_free_parton_exception ) {
      int pdg_tmp = hepevt_particles[i]->pdg_id();
      if ( (abs(pdg_tmp) <= 6 || pdg_tmp == 21)
        && !hepevt_particles[i]->end_vertex() )
        throw PartonEndVertexException(i, pdg_tmp);
    }
  }

  // Done if only appending to already existing event.
  if (append) return true;

  // 5. Store PDF, weight, cross section and other event information.
  // Flavours of incoming partons.
  if (m_store_pdf && pyinfo != 0) {
    int id1pdf = pyinfo->id1pdf();
    int id2pdf = pyinfo->id2pdf();
    if ( m_convert_gluon_to_0 ) {
      if (id1pdf == 21) id1pdf = 0;
      if (id2pdf == 21) id2pdf = 0;
    }

    // Store PDF information.
    evt->set_pdf_info( PdfInfo( id1pdf, id2pdf, pyinfo->x1pdf(),
      pyinfo->x2pdf(), pyinfo->QFac(), pyinfo->pdf1(), pyinfo->pdf2() ) );
  }

  // Store process code, scale, alpha_em, alpha_s.
  if (m_store_proc && pyinfo != 0) {
    evt->set_signal_process_id( pyinfo->code() );
    evt->set_event_scale( pyinfo->QRen() );
    if (evt->alphaQED() <= 0) evt->set_alphaQED( pyinfo->alphaEM() );
    if (evt->alphaQCD() <= 0) evt->set_alphaQCD( pyinfo->alphaS() );
  }

  // Store cross-section information in pb and event weight. The latter is
  // usually dimensionless, but in units of pb for Les Houches strategies +-4.
  if (m_store_xsec && pyinfo != 0) {
    HepMC::GenCrossSection xsec;
    xsec.set_cross_section( pyinfo->sigmaGen() * 1e9,
      pyinfo->sigmaErr() * 1e9);
    evt->set_cross_section(xsec);
    // If multiweights with possibly different xsec, overwrite central value
    std::vector<double> xsecVec = pyinfo->weightContainerPtr->getTotalXsec();
    if (xsecVec.size() > 0) {
      xsec.set_cross_section(xsecVec[0]*1e9);
      evt->set_cross_section(xsec);
    }
  }

  if (m_store_weights && pyinfo != 0) {
    for (int iweight = 0; iweight < pyinfo->numberOfWeights();
      ++iweight) {
      std::string name  = pyinfo->weightNameByIndex(iweight);
      double value      = pyinfo->weightValueByIndex(iweight);
      evt->weights()[name] = value;
    }
  }

  // Done for new event.
  return true;

}

//==========================================================================

} // end namespace HepMC

namespace Pythia8 {

//==========================================================================
// This a wrapper around HepMC::Pythia8ToHepMC in the Pythia8
// namespace that simplify the most common use cases. It stores the
// current GenEvent and output stream internally to avoid cluttering
// of user code. This class is also defined in HepMC3.h with the same
// signatures, and the user can therefore switch between HepMC version
// 2 and 3, by simply changing the include file.
class Pythia8ToHepMC : public HepMC::Pythia8ToHepMC {

public:

  // We can either have standard ascii output or none at all.
  enum OutputType { none, ascii2 };

  // Typedef for the version 2 specific classes used.
  typedef HepMC::GenEvent GenEvent;
  typedef shared_ptr<GenEvent> EventPtr;
  typedef HepMC::IO_GenEvent Writer;
  typedef shared_ptr<Writer> WriterPtr;

  // The empty constructor does not creat an aoutput stream.
  Pythia8ToHepMC() {}

  // Construct an object with an internal output stream.
  Pythia8ToHepMC(string filename, OutputType ft = ascii2) {
    setNewFile(filename, ft);
  }

  // Open a new external output stream.
  bool setNewFile(string filename, OutputType ft = ascii2) {
    switch ( ft ) {
    case ascii2:
      writerPtr = make_shared<HepMC::IO_GenEvent>(filename);
      break;
    case none:
      writerPtr = nullptr;
      break;
    }
    return writerPtr != nullptr;
  }

  // Create a new GenEvent object and fill it with information from
  // the given Pythia object.
  bool fillNextEvent(Pythia & pythia) {
    geneve = make_shared<GenEvent>();
    return fill_next_event(pythia, *geneve);
  }

  // Write out the current GenEvent to the internal stream.
  void writeEvent() {
    writerPtr->write_event(&*geneve);
  }

  // Create a new GenEvent object and fill it with information from
  // the given Pythia object and write it out directly to the
  // internal stream.
  bool writeNextEvent(Pythia & pythia) {
    if ( !fillNextEvent(pythia) ) return false;
    writeEvent();
    return true;
  }

  // Get a reference to the current GenEvent.
  GenEvent & event() {
    return *geneve;
  }

  // Get a pointer to the current GenEvent.
  EventPtr getEventPtr() {
    return geneve;
  }

  // Get a reference to the internal stream.
  Writer & output() {
    return *writerPtr;
  }

  // Get a pointer to the internal stream.
  WriterPtr outputPtr() {
    return writerPtr;
  }

  // Set cross section information in the current GenEvent.
  void setXSec(double xsec, double xsecerr) {
    HepMC::GenCrossSection xs;
    xs.set_cross_section(xsec, xsecerr);
    geneve->set_cross_section(xs);
  }

  // Update all weights in the current GenEvent.
  void setWeights(const vector<double> & wv) {
    if ( wv.size() != weightNames.size() )
      geneve->weights() = wv;
    else
      for ( int i= 0, N = wv.size(); i < N; ++i )
        geneve->weights()[weightNames[i]] = wv[i];
  }

  // Set all weight names in the current run.
  void setWeightNames(const vector<string> &wnv) {
    weightNames = wnv;
  }

  // Update the PDF information in the current GenEvent
  void setPdfInfo(int id1, int id2, double x1, double x2,
                  double scale, double xf1, double xf2,
                  int pdf1 = 0, int pdf2 = 0) {
    HepMC::PdfInfo pdf(id1, id2, x1, x2, scale, xf1, xf2, pdf1, pdf2);
    geneve->set_pdf_info(pdf);
  }

private:

  // The current GenEvent.
  EventPtr geneve = nullptr;

  // The output stream.
  WriterPtr writerPtr = nullptr;

  // The weight names in the current run.
  vector<string> weightNames;

};

}

#endif  // end Pythia8_HepMC2_H
