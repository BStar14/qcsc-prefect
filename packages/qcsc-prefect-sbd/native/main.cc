#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <deque>

#include <unistd.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "sbd/sbd.h"
#include "mpi.h"



int main(int argc, char * argv[]) {

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);

  auto sbd_data = sbd::tpb::generate_sbd_data(argc,argv);

  std::string adetfile("AlphaDets.bin");
  std::string fcidumpfile("fcidump.txt");
  std::string loadname("");
  std::string savename("");

  int L;
  int N;
  double energy;
  std::vector<double> density;
  std::vector<std::vector<size_t>> co_adet;
  std::vector<std::vector<size_t>> co_bdet;
  std::vector<std::vector<double>> one_p_rdm;
  std::vector<std::vector<double>> two_p_rdm;
  sbd::FCIDump fcidump;

  std::cout.precision(16);

  /**
     load fcidump data
   */
  if( mpi_rank == 0 ) {
    fcidump = sbd::LoadFCIDump(fcidumpfile);
  }
  sbd::MpiBcast(fcidump,0,comm);

  for(const auto & [key,value] : fcidump.header) {
    if( key == std::string("NORB") ) {
      L = std::atoi(value.c_str());
    }
    if( key == std::string("NELEC") ) {
      N = std::atoi(value.c_str());
    }
  }

  /**
     setup determinants for alpha and beta spin orbitals
   */
  std::vector<std::vector<size_t>> adet;
  std::vector<std::vector<size_t>> bdet;
  if( mpi_rank == 0 ) {
    sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
    std::cout << "Loaded " << adet.size() << " alpha determinants." << std::endl;
  }

  sbd::MpiBcast(adet,0,comm);
  bdet = adet;

  /**
     sample-based diagonalization using data for fcidump, adet, bdet.
   */
  sbd::tpb::diag(comm,sbd_data,fcidump,adet,bdet,loadname,savename,
              energy,density,co_adet,co_bdet,one_p_rdm,two_p_rdm);

  if( mpi_rank == 0 ) {
    std::cout << "Davidson energy: " << energy << std::endl;
    std::ofstream ofs_energy("davidson_energy.txt");
    ofs_energy.precision(16);
    ofs_energy << energy << std::endl;
    ofs_energy.close();
    std::ofstream ofs_occa("occ_a.txt");
    ofs_occa.precision(16);
    std::ofstream ofs_occb("occ_b.txt");
    ofs_occb.precision(16);
    for(size_t i=0; i < density.size()/2; i++) {
      ofs_occa << density[2*i] << std::endl;
      ofs_occb << density[2*i + 1] << std::endl;
    }
    ofs_occa.close();
    ofs_occb.close();
    std::cout << "Number of carryover determinants: " << co_adet.size() << std::endl;
    std::ofstream ofs_co_bin("carryover.bin", std::ios::binary);
    const size_t bytes_per_config = (L + 7) / 8;
    std::vector<uint8_t> bytes(bytes_per_config);
    for (size_t i = 0; i < co_adet.size(); ++i) {
      std::fill(bytes.begin(), bytes.end(), 0);
      for (size_t j = 0; j < L; ++j) {
        size_t rev_idx = L - 1 - j;                 // sbd::makestring order
        size_t pw = rev_idx % sbd_data.bit_length;  // position in word
        size_t bw = rev_idx / sbd_data.bit_length;  // index of word
        bool bit = (co_adet[i][bw] >> pw) & 1ULL;
        size_t pb = 7 - (j % 8);                    // big-endian bit order
        size_t bb = j / 8;                          // index of byte
        bytes[bb] |= static_cast<uint8_t>(bit << pb);
      }
      ofs_co_bin.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    ofs_co_bin.close();
  }

  /**
     Finalize
  */

  MPI_Finalize();
  return 0;
}
