#!/usr/bin/env python3
import re
import sys
import fileinput
import argparse
import io
import os

from ase.io import read
from ase import units

import numpy as np
from itertools import permutations, combinations, product
from scipy.optimize import quadratic_assignment, linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from moltric import getDM, sortDM
from moltric import QQkabsch, QQrmsd
from moltric import DMD_GOAT
from moltric import flapsolve, fqapsolve, spectral_linear_assignment     # For another set of comparisons

from arbalign import arbalign                    # Kazuumi's python3 translation for ArbAlign
from otmol_alignment import molecule_alignment   # OTMol
from molalignlib import assign_atoms             # MolAlignLib
from ase import Atoms                            # Needed for MolAlignLib


# Some top-level arguments:

# Self-explanatory ... 
#    True - calculate RMSDs
#   False - calculate DMDs
calculate_RMSD_instead = True


##################################################################################################################################

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])

    if (len(sys.argv) < 3):
      print("Error! Wrong number of arguments... should be:")
      print("        (1) the fDMD threshold to identify new points with (use 0.00001 for a complete comparison)")
      print("        (2) the output XYZ file to dump new points into (by default, will append rather than overwrite)")
      print("Optional argumnets:")
      print("        (3) the cell parameters for the XYZ (if they use PBC) ... enter 'None' if you need the 4th argument and need to skip this one")
      print("        (4) an unused method option (an integer)")
      print("        (5) an input XYZ file to read reference points from")
      print("")
      print("Example usage:")
      print('# cat /mnt/lustre/koa/koastore/rsun_group/kazuumiTest1/MLexp1/cGDMLv1/ionicliquid1.xyz | /mnt/lustre/koa/koastore/rsun_group/scripts/printUniqueGeometriesBy_fQAP.py 1.0 output.xyz "12.76858 0.0 0.0 0.0 12.76858 0.0 0.0 0.0 12.76858"')
      print("")
      sys.exit()


    fDMDmax   = sys.argv[1]
    outputfile= sys.argv[2]

    try:
      fDMDmax = float(fDMDmax)
      assert(fDMDmax > 0)
    except:
      raise ValueError("Error! Something is wrong with the first argument (the fDMD threshold), must be a positive float")

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

    method_option = 0
    if (len(sys.argv) > 4):
      try:
        method_option = int(sys.argv[4])
      except:
        print("Note: Could not read the method option (4th argument) correctly, so keeping the current default option: {:d}".format(method_option))

    if (len(sys.argv) > 5):
      inputfile = sys.argv[5]
    else:
      inputfile = None

    #########################################################################################

    if (inputfile is None):
      f_input = sys.stdin
      f_label = inputfile

    else:
      f_input = open(inputfile,'r')
      f_label = "stdin"

    f_output = open(outputfile,'a')


    
    try:
      n_atoms = int(f_input.readline())
    except:
      raise ValueError("Input ({:s}) must be xyz, but it is formatted wrong. The first line must be the number of atoms ... Exiting!".format(f_label))

    try:
      commentline = f_input.readline()
    except:
      raise ValueError("Input ({:s}) must be xyz, but it is formatted wrong. The second line must be a comment line ... Exiting!".format(f_label))

    z = []
    lines = []
    r_i = np.zeros((n_atoms,3))
    for i in range(n_atoms):
      try:
        lines.append(f_input.readline())
        fields = lines[-1].split()
        z.append(fields[0])
        r_i[i,:] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
      except:
        raise ValueError("Input ({:s}) must be xyz, but it is formatted wrong. The {:d}-th line of the xyz block read so far is missing/incorrect ... Exiting!".format(f_label,i))

    #########################################################################################

    z = np.array(z)
    z_unique = np.unique(z)

    # Sort the atomic blocks 'z_unique' from smallest count to largest 
    z_count = np.array([sum(z==z_i) for z_i in z_unique])
    z_unique = z_unique[np.argsort(z_count)]                   # Let less common elements go first


    if (lattice is None):
      lat_and_inv = None
    else:
      lat_and_inv = (lattice, np.linalg.inv(lattice))

    #########################################################################################

    # Generate permutations for a brute force (up to a fixed number)
    perms = []

    # If there aren't many atoms, just make all of the permutations
    if (n_atoms < 6):
      order_unique_atoms = []
      for group in z_unique:
        unique_atoms = np.array([ii for ii, g in enumerate(z) if g == group])
        perms.append(list(permutations(unique_atoms)))
        order_unique_atoms.append(unique_atoms)

      orig_order = np.argsort(np.concatenate(order_unique_atoms))
      perms = [np.concatenate(x)[orig_order] for x in list(product(*perms))]

    # Otherwise, use approximations and look at a limited number of permutations
    else:

      order_unique_atoms = []
      perms = [[] for group in z_unique]
      for i, group in enumerate(z_unique):
        unique_atoms = np.array([ii for ii, g in enumerate(z) if g == group])
        order_unique_atoms.append(unique_atoms)
        perms[i].append(unique_atoms)
        for j in range(10):
          perms[i].append(np.random.permutation(unique_atoms))

      orig_order = np.argsort(np.concatenate(order_unique_atoms))
      perms = [np.concatenate(tuple([sub_perm[i] for sub_perm in perms]))[orig_order] for i in range(len(perms[0]))]

    #########################################################################################

    # Summary

    Nperms = len(perms)
    print("  fDMD threshold:", fDMDmax)
    print("   Method option:", method_option)
    print("    Unique atoms:")
    for unique_atoms in order_unique_atoms:
      print(unique_atoms)
    print("         Z order:", z_unique)
    print("   Npermutations:", Nperms)
    for i, perm in enumerate(perms):
      if (n_atoms > 20):
        print(*tuple(perm[:min(15,n_atoms)]),"...")
      else:
        print(perm)
      if (i > 5 and i+1 < Nperms):
        print("...")
        break
    print("")

    # Print the header
    if True:
      print((" {:8s}# {:>6s} {:>6s}   "+" {:>8s}"*7 + "  {:>8s} ({:s}) ({:s})").format("DMDbound", "i", "j", "ArbAlign", "OTMol", "MolAlign", "Umeyama", "FAQ", "GOATf", "GOAT", "minimum", "Nf", "N"),flush=True)

    #########################################################################################

    # Prepare the input geometries

    train_R = []
    train_DMii = []
    train_sorted_DMii = []

    Ntraversed = 0

    if (inputfile is None):

      Ntrain = 1
      Nexisting = 0

      DMii = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))
      train_R.append(r_i)
      train_DMii.append(DMii)

      sorted_DMii = sortDM(DMii,z,z_unique)
      train_sorted_DMii.append(sorted_DMii)

      print("Adding index {:8d} to the training set ...   Sparsity {:6.4f}  = {:d}/{:d}".format(0,Ntrain/(1.0e0+Nexisting+Ntraversed),Ntrain,Nexisting+Ntraversed+1),flush=True)
      f_output.write(str(n_atoms)+"\n")
      f_output.write(commentline)
      for line in lines:
        f_output.write(line)
      f_output.flush()

    else:

      print("Reading through the reference XYZ file {:s} ...".format(inputfile),flush=True)

      DMii = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))
      train_R.append(r_i)
      train_DMii.append(DMii)

      sorted_DMii = sortDM(DMii,z,z_unique)
      train_sorted_DMii.append(sorted_DMii)

      Nexisting = 1

      while True:

        try:
          n_atoms = int(f_input.readline())
        except:
          break

        try:
          commentline = f_input.readline()

          lines = []
          r_i = np.zeros((n_atoms,3))
          for ii in range(n_atoms):
            lines.append(f_input.readline())
            fields = lines[-1].split()
            r_i[ii,:] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
        except:
          raise ValueError("Input file ({:s}) must be xyz, but the {:d}-the xyz block is formatted wrong ... Exiting!".format(inputfile,Nexisting))

        DMii = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))
        train_R.append(r_i)
        train_DMii.append(DMii)

        sorted_DMii = sortDM(DMii,z,z_unique)
        train_sorted_DMii.append(sorted_DMii)

        Nexisting += 1

      f_input.close()
      Ntrain = Nexisting
      print("Number of points found in '{:s}': {:d}".format(inputfile,Ntrain),flush=True)




    while True:

      Ntraversed += 1

      try:
        n_atoms = int(sys.stdin.readline())
      except:
        break
     
      try:
        commentline = sys.stdin.readline()

        lines = []
        r_i = np.zeros((n_atoms,3))
        for ii in range(n_atoms):
          lines.append(sys.stdin.readline())
          fields = lines[-1].split()
          r_i[ii,:] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
      except:
        raise ValueError("Input (stdin) must be xyz, but the {:d}-the xyz block is formatted wrong ... Exiting!".format(Ntraversed))

      DMii = getDM(r_i,lat_and_inv=lat_and_inv,invert_distances=(not calculate_RMSD_instead))

      sorted_DMii = sortDM(DMii,z,z_unique)





      #############################################################################################################################################

      # Let's look through every point in the training set to see if the current point is unique
      addition_flag = True
      for j,r_j in enumerate(train_R): 

        if True:

          DMjj = train_DMii[j]
          sorted_DMjj = train_sorted_DMii[j]


          qapDMDs = {}
          cost_perm_qaps = []

          #########################################################################################################################################

          # Method 1: Do a ArbAlign-esque search over 48 permutations after aligning the principal axes
          if True: 
            alignments = arbalign(r_i,r_j,z,inverse_flag=False,calculate_RMSD_instead=calculate_RMSD_instead)
            perm_qap = alignments[0][3]

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["Arb"] = qapDMD

          # Method 2: OTMol (OTM)
          if True:
            alpha_list = np.arange(0, 1.0, 0.01)[1:]
            perm_qap, otmol_rmsd, otmol_alpha = molecule_alignment(r_i, r_j, z, z, method="fGW", alpha_list=alpha_list)

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["OTM"] = qapDMD

          # Method 3: MolAlignLib (MAL)
          if True:
            moli = Atoms(symbols=z,positions=r_i)
            molj = Atoms(symbols=z,positions=r_j)
            assignment = assign_atoms(moli, molj, fast=True, tol=15.0)  # Be careful... if the tolerance "tol" is too low, it doesn't work
            perm_qap = assignment.order
            molj = molj[perm_qap]
            molalignlib_rmsd = molj.align_to(moli)

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["MAL"] = qapDMD



          #########################################################################################################################################

          # Method 4: Do a LAP search for each atomic block SEPARATELY with a sorted distance matrix
          if True:
#           perm_qap = flapsolve(DMii,DMjj,z,z_unique)
            perm_qap = flapsolve(sorted_DMii,sorted_DMjj,z,z_unique,pre_sorted=True)

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["LAPsDM"] = qapDMD

            # Sorted distance matrices also give lower bounds on DMDs
            DMD_lowerbound = np.sum((sorted_DMii - sorted_DMjj[perm_qap,:])**2)

          # Method 5: Do a QAP search with seeded FAQs for each atomic block
          if True:
            perm_qap = fqapsolve(DMii,DMjj,z,z_unique)

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["FAQ"] = qapDMD

          # Method 6: Do a spectral assignment (Umeyama-like)
          if True:
            perm_spectral, cost_spectral = spectral_linear_assignment(DMii, DMjj, groups=order_unique_atoms)
            perm_qap = perm_spectral

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["spectral"] = qapDMD



          #########################################################################################################################################

          # Method 7: Do a single (forward) seeded GOAT for each atomic block
          if True:
            if calculate_RMSD_instead:
              perm_qap, perm_DMD, NiterationsGOAT = DMD_GOAT(z,DM_i=DMii,DM_j=DMjj,r_i=r_i,r_j=r_j,z_order="forward",metric="RMSD")
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              perm_qap, perm_DMD, NiterationsGOAT = DMD_GOAT(z,DM_i=DMii,DM_j=DMjj,z_order="forward",metric="DMD")
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["GOATf"] = qapDMD

          # Method 8: Do a complete seeded GOAT for each atomic block
          if True:
            if calculate_RMSD_instead:
              perm_qap, perm_DMD, totalNiterationsGOAT = DMD_GOAT(z,DM_i=DMii,DM_j=DMjj,r_i=r_i,r_j=r_j,z_order="complete",metric="RMSD")
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
              qapDMD = min(RMSD1,RMSD2)
            else:
              perm_qap, perm_DMD, totalNiterationsGOAT = DMD_GOAT(z,DM_i=DMii,DM_j=DMjj,z_order="complete",metric="DMD")
              qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

            qapDMDs["GOATc"] = qapDMD

          #########################################################################################################################################



          # Brute-force search (over a limited number of permutations)
          Nperms_tried = 0
          cost_perms = []
          for perm in perms:
            Nperms_tried += 1

            if calculate_RMSD_instead:
              RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm,:]))
              RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm,:]))
              cost_perms.append(min(RMSD1,RMSD2))
            else:
              cost_perms.append(np.sum((DMii - DMjj[perm,:][:,perm])**2))

          qapDMDs["brute"] = min(cost_perms)

          #########################################################################################################################################

          # Compute the minimum DMD of all methods
          minDMD = min(qapDMDs.values())


          # Print the to screen!
          #   DMD_lowerbound#  i     j      qapDMDs   + the two new ones (OTM and MAL at the end)
          print((" {:8.4f}# {:6d} {:6d}   "+" {:8.4f}"*7 + "  {:8.4f} ({:d}) ({:d})").format(DMD_lowerbound, Ntraversed, j, qapDMDs["Arb"], qapDMDs["OTM"], qapDMDs["MAL"], qapDMDs["spectral"], qapDMDs["FAQ"], qapDMDs["GOATf"], qapDMDs["GOATc"], minDMD, NiterationsGOAT, totalNiterationsGOAT),flush=True)



          # The "cost_ij" variable will decide whether we accept/reject a frame
          # For now, just base it off of the DMD, purely
          cost_ij = minDMD

          #########################################################################################################################################

        # If the distance to some other point is small enough, don't add it
        if (cost_ij < fDMDmax):
          addition_flag = False
#         break                     # temporarily commnted out for comparison

      if addition_flag:
        Ntrain += 1
        train_R.append(r_i)
        train_DMii.append(DMii)

        train_sorted_DMii.append(sorted_DMii)

        print("Adding index {:8d} to the training set ...   Sparsity {:6.4f}  = {:d}/{:d}".format(Ntraversed,Ntrain/(1.0e0+Nexisting+Ntraversed),Ntrain,Nexisting+Ntraversed+1),flush=True)
        f_output.write(str(n_atoms)+"\n")
        f_output.write(commentline)
        for line in lines:
          f_output.write(line)
        f_output.flush()

    print("Got to the end!",flush=True)
    f_output.close()


