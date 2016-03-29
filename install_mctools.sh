#!/bin/bash

function src2inst {
    echo ${1/src/install}
}

function tar2src {
    local dir="src/$(tar --exclude '*/*' -tf $1)"
    echo $dir
}

function tar2inst {
    local src=$(tar2src $1)
    local dir=$(src2inst $src)
    echo $dir
}

function unpack {
    local dir=$(tar2src $1)
    if test \! -d $dir; then
        tar xzvf $1 -C src/
    fi
}

function patch_hepmc2 {
cat <<EOF > HepMC-no-hexfloat.patch
diff -ur HepMC-2.06.09.orig/src/IO_AsciiParticles.cc HepMC-2.06.09/src/IO_AsciiParticles.cc
--- HepMC-2.06.09.orig/src/IO_AsciiParticles.cc	2010-05-17 17:23:44.000000000 +0200
+++ HepMC-2.06.09/src/IO_AsciiParticles.cc	2015-03-06 17:38:41.860507302 +0100
@@ -156,14 +156,14 @@
 
       xmassi = (*part)->generatedMass();
       if(fabs(xmassi) < 0.0001) xmassi =0.;
-      m_outstream->setf(std::ios::fixed);
+      m_outstream->unsetf(std::ios::floatfield);
       m_outstream->precision(3);
       m_outstream->width(8);
       *m_outstream << xmassi << " ";
       m_outstream->setf(std::ios::scientific,std::ios::floatfield);
       m_outstream->precision(m_precision);
 
-      m_outstream->setf(std::ios::fixed);
+      m_outstream->unsetf(std::ios::floatfield);
       m_outstream->precision(3);
       m_outstream->width(6);
       etai = (*part)->momentum().eta();
EOF
patch $1/src/IO_AsciiParticles.cc HepMC-no-hexfloat.patch
}

function install_hepmc2 {
    local THISDIR=$(pwd)
    local SRCDIR=$1
    local BUILDDIR=${1/src/build/}
    local INSTALLDIR=$2
    mkdir -p $BUILDDIR
    mkdir -p $INSTALLDIR
    cd $BUILDDIR
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALLDIR \
        -Dmomentum:STRING=MEV \
        -Dlength:STRING=MM \
        $SRCDIR
    make -j4 && \
    make test && \
    make install
    cd $THISDIR
}

function install_lhapdf6 {
    local THISDIR=$(pwd)
    local SRCDIR=$1
    local BOOSTDIR=$2
    local INSTALLDIR=$3
    mkdir -p $INSTALLDIR
    cd $SRCDIR
    ./configure --prefix=$INSTALLDIR --with-boost=$BOOSTDIR && \
    make -j4 && \
    make install
    cd $THISDIR
}

function install_pythia8 {
    local THISDIR=$(pwd)
    local SRCDIR=$1
    local HEPMC2DIR=$2
    local LHAPDF6DIR=$3
    local BOOSTDIR=$4
    local INSTALLDIR=$5
    mkdir -p $INSTALLDIR
    cd $SRCDIR
    ./configure --prefix=$INSTALLDIR \
        --with-hepmc2=$HEPMC2DIR \
        --with-lhapdf6=$LHAPDF6DIR \
        --with-boost=$BOOSTDIR \
        --with-gzip && \
    make -j4 && \
    make install
    cd $THISDIR
}

function install_delphes {
    local THISDIR=$(pwd)
    local SRCDIR=$1
    cd $SRCDIR
    make -j4
    cd $THISDIR
}

MG5="MG5_aMC_v2.3.3.tar.gz"
HEPMC2="HepMC-2.06.09.tar.gz"
PYTHIA8="pythia8215.tgz"
DELPHES="Delphes-3.3.2.tar.gz"
LHAPDF6="LHAPDF-6.1.6.tar.gz"
BOOST="boost_1_60_0.tar.gz"

mkdir -p src build install

# Setup root
if [[ "$(hostname | cut -f 1 -d .)" == "bestlapp" ]]; then
    setupATLAS
    lsetup root
fi
source "$(root-config --bindir)/thisroot.sh"

# Download MG5_aMC@NLO
if test \! -e $MG5; then
    wget https://launchpad.net/mg5amcnlo/2.0/2.3.0/+download/$MG5
fi

# Download HepMC2
if test \! -e $HEPMC2; then
    wget http://lcgapp.cern.ch/project/simu/HepMC/download/$HEPMC2
fi

# Download LHAPDF6
if test \! -e $LHAPDF6; then
    wget http://www.hepforge.org/archive/lhapdf/$LHAPDF6
fi

# Download Pythia8215
if test \! -e $PYTHIA8; then
    wget http://home.thep.lu.se/~torbjorn/pythia8/$PYTHIA8
fi

# Download Delphes
if test \! -e $DELPHES; then
    wget http://cp3.irmp.ucl.ac.be/downloads/$DELPHES
fi

# Download BOOST
if test \! -e $BOOST; then
    wget http://sourceforge.net/projects/boost/files/boost/1.60.0/$BOOST
fi

# Wait for wget
wait

# Unpack boost
BOOSTDIR=$(pwd)/$(tar2src $BOOST)
unpack $BOOST

# Unpack MG5_aMC@NLO
MG5SRC=$(pwd)/$(tar2src $MG5)
unpack $MG5

# Unpack and install HepMC2
HEPMC2SRC=$(pwd)/$(tar2src $HEPMC2)
HEPMC2INST=$(pwd)/$(tar2inst $HEPMC2)
unpack $HEPMC2
patch_hepmc2 $HEPMC2SRC
install_hepmc2 $HEPMC2SRC $HEPMC2INST

# Unpack and install LHAPDF6
LHAPDF6SRC=$(pwd)/$(tar2src $LHAPDF6)
LHAPDF6INST=$(pwd)/$(tar2inst $LHAPDF6)
unpack $LHAPDF6
install_lhapdf6 $LHAPDF6SRC $BOOSTDIR $LHAPDF6INST

# Unpack and install Pythia8215
PYTHIA8SRC=$(pwd)/$(tar2src $PYTHIA8)
PYTHIA8INST=$(pwd)/$(tar2inst $PYTHIA8)
unpack $PYTHIA8
install_pythia8 $PYTHIA8SRC $HEPMC2INST $LHAPDF6INST $BOOSTDIR $PYTHIA8INST

# Unpack and install Delphes
DELPHESSRC=$(pwd)/$(tar2src $DELPHES)
DELPHESINST=$(pwd)/$(tar2inst $DELPHES)
unpack $DELPHES
install_delphes $DELPHESSRC $DELPHESINST
