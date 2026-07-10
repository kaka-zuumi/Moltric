#!/usr/bin/env python3
import sys
import numpy as np
from scipy.optimize import quadratic_assignment, linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from moltric import getDM, sortDM
from moltric import QQkabsch, QQrmsd

from moltric import align_molecules              # The new GOAT algorithm for DMD




##################################################################################################################################


if __name__ == '__main__':

    calculate_RMSD_instead = False

    if (len(sys.argv) < 3):

      errmessage = """
      Error! Wrong number of arguments... should be:
              (1) XYZfile 1
              (2) XYZfile 2 (to align to 1)
      Optional arguments:
              (3) the cell parameters for the XYZ (if they use PBC) ... enter 'None' if you need the 4th argument and need to skip this one

      Example usage:
      # moltric/moltric.py examples/fromFujioka_CH.indene_19atoms.xyz examples/fromFujioka_CH.indene_19atoms.xyz
      # moltric/moltric.py examples/fromFujioka_ionicliquid_PBC_205atoms.xyz examples/fromFujioka_ionicliquid_PBC_205atoms.xyz "12.76858 0.0 0.0 0.0 12.76858 0.0 0.0 0.0 12.76858"
      """
      raise ValueError(errmessage)


    inputfileA = sys.argv[1]
    inputfileB = sys.argv[2]

    if (len(sys.argv) > 3):
      cell_input = sys.argv[3]
      try:
        lattice = np.array([float(x) for x in cell_input.split()]).reshape(3,3)
      except:
#       raise ValueError("Error! The 'cell' parameter inputted must be a string of 9 numbers... Exiting!")
        print("Note: Could not read the cell parameters (3rd argument) correctly, so turning off PBC...")
        lattice = None
    else:
      lattice = None

    ############################################################################################################################

    zi = []

    Nexisting = 1
    train_Ri = []
    train_DMi = []
    f_input = open(inputfileA,'r')
    while True:
      try:
        n_atoms = int(f_input.readline())
      except:
        break
      try:
        commentline = f_input.readline()
        z = []
        r_i = np.zeros((n_atoms,3))
        for ii in range(n_atoms):
          line = f_input.readline()
          fields = line.split()
          z.append(fields[0])
          r_i[ii,:] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
      except:
        raise ValueError("Input file ({:s}) must be xyz, but the {:d}-the xyz block is formatted wrong ... Exiting!".format(inputfileA,Nexisting))

      if (Nexisting == 1):
        if (lattice is None):
          lat_and_inv = None
        else:
          lat_and_inv = (lattice, np.linalg.inv(lattice))

      DMi = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))
      train_Ri.append(r_i)
      train_DMi.append(DMi)
      zi.append(z)

      Nexisting += 1

      assert(z == zi[0])

    f_input.close()
    Ni = Nexisting
    print("# Number of points found in '{:s}': {:d}".format(inputfileA,Ni),flush=True)

    if (Ni == 0): sys.exit()

    ############################################################################################################################

    zj = []

    Nexisting = 1
    train_Rj = []
    train_DMj = []
    f_input = open(inputfileB,'r')
    while True:
      try:
        n_atoms = int(f_input.readline())
      except:
        break
      try:
        commentline = f_input.readline()
        z = []
        r_i = np.zeros((n_atoms,3))
        for ii in range(n_atoms):
          line = f_input.readline()
          fields = line.split()
          z.append(fields[0])
          r_i[ii,:] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
      except:
        raise ValueError("Input file ({:s}) must be xyz, but the {:d}-the xyz block is formatted wrong ... Exiting!".format(inputfileB,Nexisting))

      DMi = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))
      train_Rj.append(r_i)
      train_DMj.append(DMi)
      zj.append(z)

      Nexisting += 1

      assert(z == zi[0])

    f_input.close()
    Nj = Nexisting
    print("# Number of points found in '{:s}': {:d}".format(inputfileB,Nj),flush=True)

    if (Nj == 0): sys.exit()

    ############################################################################################################################



    z = np.array(zi[0])
    z_unique = np.unique(z)

    # Sort the atomic blocks 'z_unique' from smallest count to largest
    z_count = np.array([sum(z==z_i) for z_i in z_unique])
    z_unique = z_unique[np.argsort(z_count)]                   # Let less common elements go first

    if (len(z_unique) <= 4):
      z_order = "complete"
    else:
      z_order = "forward"




    if True:
      if calculate_RMSD_instead:
        print("# {:>4s} {:>4s}    {:8s}".format("i","j"," RMSD"),flush=True)
      else:
        print("# {:>4s} {:>4s}    {:8s}".format("i","j","  DMD"),flush=True)

    i = 0
    for r_i,DMii in zip(train_Ri,train_DMi):
      j = 0
      for r_j,DMjj in zip(train_Rj,train_DMj):

        if calculate_RMSD_instead:
          perm_qap, perm_DMD, totalNiterationsGOAT = align_molecules(z,DM_i=DMii,DM_j=DMjj,r_i=r_i,r_j=r_j,z_order=z_order,metric="RMSD")
          RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
          RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
          qapDMD = min(RMSD1,RMSD2)
        else:
          perm_qap, perm_DMD, totalNiterationsGOAT = align_molecules(z,DM_i=DMii,DM_j=DMjj,z_order=z_order,metric="DMD")
          qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

        print("  {:4d} {:4d}    {:8.4f}".format(i,j,qapDMD),flush=True)

        j += 1
      i += 1

