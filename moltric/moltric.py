#!/usr/bin/env python3
import numpy as np
from scipy.optimize import quadratic_assignment, linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from qapot import quadratic_assignment_ot # From that paper

#import torch








##################################################################################################################################

# Helper functions:

# This gets the distance matrix for a SINGLE geometry
# Specify the cell (3x3 matrix -> 9 vector) with "lat_and_inv" (18 values in total)
def getDM(r, lat_and_inv=None, invert_distances=True):

  if lat_and_inv is None:
    dum1 = pdist(r.reshape(-1,3), 'euclidean')
  else:
    dum1 = pdist(r.reshape(-1,3), lambda u, v: np.linalg.norm(_pbc_diff(u - v, lat_and_inv)))

  # Invert the distances!
  if invert_distances:
    dum1 = 1.0e0 / dum1

  return squareform(dum1, checks=False) 



def _pbc_diff(diffs, lat_and_inv, use_torch=False):

  lat, lat_inv = lat_and_inv
  if use_torch:
      c = lat_inv.mm(diffs.t())
      diffs -= lat.mm(c.round()).t()
  else:
      c = lat_inv.dot(diffs.T)
      diffs -= lat.dot(np.around(c)).T

  return diffs



# Create a sorted DM
def sortDM(DM,z,z_unique_order):
  n_atoms = len(z)
  sorted_DM = DM.copy()
  for group in z_unique_order:
    group_indices = np.array([ii for ii, g in enumerate(z) if g == group])

    for ii in range(n_atoms):
      sorted_DM[ii,group_indices] = np.sort(DM[ii,group_indices])

  return sorted_DM


##################################################################################################################################

# The best QAP solver for DMD using the GOAT algorithm

def DMD_GOAT(z,r_i=None,r_j=None,DM_i=None,DM_j=None,z_order="symmetric",metric="DMD"):

  # Input:
  #              z - the atomic symbols (assumes the two molecules have the same ordering)
  #    r_i or DM_i - the 1st molecular geometry coordinates (r_i) or distance matrix (DM_i)
  #    r_j or DM_j - the 2nd molecular geometry coordinates (r_j) or distance matrix (DM_j)
  #        z_order - the elemental ordering ("forward", "symmetric", or "complete")
  #         metric - the metric to minimize ("RMSD" or "DMD")

  # Returns:
  #     (1) The optimal permutation
  #     (2) The minimum DMD (or RMSD, if requested)
  #     (3) The number of iterations spent in GOAT


  # Prepare the elements (assuming the two molecules have the same molecular formula)
  z = np.array(z)
  z_unique = np.unique(z)
  z_count = np.array([sum(z==z_i) for z_i in z_unique])

  # Prepare molecular geometry i
  if (DM_i is None):
    DMii = getDM(r_i)
  else:
    DMii = DM_i

  # Prepare molecular geometry j
  if (DM_j is None):
    DMjj = getDM(r_j)
  else:
    DMjj = DM_j


  if (z_order == "forward"):

    # The "forward" permutation of elements
    z_forward = z_unique[np.argsort(z_count)]

    perm_qap, NiterationsGOAT = fqapGOATsolve(DMii,DMjj,z,z_forward)
    if (metric=="RMSD"):
      RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
      RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
      qapDMD = min(RMSD1,RMSD2)
    else:
      qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

    min_perm = perm_qap
    minDMD = qapDMD

  elif (z_order == "symmetric"):

    # The "forward" permutation of elements and its reverse
    z_forward = z_unique[np.argsort(z_count)]
    z_backward = np.flip(z_forward)
    z_perms = [z_forward, z_backward]

    totalNiterationsGOAT = 0; minDMD = 1.0e9
    for z_perm in z_perms:
      perm_qap, NiterationsGOAT = fqapGOATsolve(DMii,DMjj,z,z_perm)
      if (metric=="RMSD"):
        RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
        RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
        qapDMD = min(RMSD1,RMSD2)
      else:
        qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

      totalNiterationsGOAT += NiterationsGOAT
      if (qapDMD < minDMD):
        min_perm = perm_qap
        minDMD = qapDMD
    NiterationsGOAT = totalNiterationsGOAT


  elif (z_order == "complete"):

    # All possible z_unique orders (permutations)
    import itertools
    z_perms = list(itertools.permutations(z_unique))             # Need "permutations" from itertools for this (import this manually)

    totalNiterationsGOAT = 0; minDMD = 1.0e9
    for z_perm in z_perms:
      perm_qap, NiterationsGOAT = fqapGOATsolve(DMii,DMjj,z,z_perm)
      if (metric=="RMSD"):
        RMSD1 = QQrmsd(*QQkabsch(r_i, r_j[perm_qap,:]))
        RMSD2 = QQrmsd(*QQkabsch(r_i, -r_j[perm_qap,:]))
        qapDMD = min(RMSD1,RMSD2)
      else:
        qapDMD = np.sum((DMii - DMjj[perm_qap,:][:,perm_qap])**2)

      totalNiterationsGOAT += NiterationsGOAT
      if (qapDMD < minDMD):
        min_perm = perm_qap
        minDMD = qapDMD
    NiterationsGOAT = totalNiterationsGOAT

  else:
    raise ValueError('Incorrect z_order! Must be: "forward", "symmetric", or "complete"')


  return min_perm, minDMD, NiterationsGOAT








##################################################################################################################################

# A filtered QAP solver with swaps
def sfqapsolve(DMi,DMj,z,z_unique_order):   # The z order determines which atoms (element) are assigned first

  n_atoms = DMi.shape[0]

  perm_qap = []
  perm_qaps = []
  cost_perm_qaps = []
  remaining_indices = list(range(n_atoms))
  submatrix_indices = []

  # Iterate over each group of atoms (element)
  for group in z_unique_order:

      # Indices of the current group
      group_indices = [ii for ii, g in enumerate(z) if g == group]

      original_perm_qap = perm_qap.copy()
      original_submatrix_indices = submatrix_indices.copy()

      # Look at neighbor swaps if this is the last set of atoms
      if (group == z_unique_order[-1]):
        if (len(z_unique_order) > 1):
          prev_group_indices = [ii for ii, g in enumerate(z) if g == z_unique_order[-2]]
        else:
          prev_group_indices = []
          qapsolve = quadratic_assignment(DMi, DMj, options={'rng': np.random.default_rng(),'maximize':True},method='faq')
          perm_qap = qapsolve["col_ind"]
#         sub_perm_qap = qapsolve["col_ind"]
#         perm_qap = submatrix_indices[qapsolve["col_ind"]]
          qapDMD = np.sum((DMi - DMj[perm_qap,:][:,perm_qap])**2)
          perm_qaps.append(perm_qap)
          cost_perm_qaps.append(qapDMD)
         

        for iatom in prev_group_indices:
          iiatom = list(original_submatrix_indices).index(iatom)
          for jatom in prev_group_indices:
            jjatom = list(original_submatrix_indices).index(jatom)

            # Go through all pairs of distinct atoms, and also the swap of the first atom with itself (so effectively, no swap)
            if (iiatom >= jjatom):
              if (iiatom == jjatom and iatom > prev_group_indices[0]): continue
              perm_qap = original_perm_qap.copy()
              perm_qap[iiatom] = original_perm_qap[jjatom]
              perm_qap[jjatom] = original_perm_qap[iiatom]
              submatrix_indices = original_submatrix_indices.copy()
              sub_perm_fixed = [(jj,perm_qap[ii]) for ii,jj in enumerate(submatrix_indices)]

              # Submatrix extraction for the group
              submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)]
              perm_fixed = [[submatrix_indices.index(ii),submatrix_indices.index(jj)] for ii,jj in sub_perm_fixed]
              submatrix_indices = np.array(submatrix_indices)

              DMi_sub = DMi[np.ix_(submatrix_indices, submatrix_indices)]
              DMj_sub = DMj[np.ix_(submatrix_indices, submatrix_indices)]

              # Do the QAP solve (with some atoms fixed)!
              if (len(perm_fixed) == 0):
                qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True},method='faq')
              else:
                qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True,'partial_match':perm_fixed},method='faq')

              sub_perm_qap = qapsolve["col_ind"]
              perm_qap = submatrix_indices[qapsolve["col_ind"]]

              # Save some information about the solution of this swap
              qapDMD = np.sum((DMi - DMj[perm_qap,:][:,perm_qap])**2)
              perm_qaps.append(perm_qap)
              cost_perm_qaps.append(qapDMD)

      # If not, don't do any swaps
      else:
        sub_perm_fixed = [(jj,perm_qap[ii]) for ii,jj in enumerate(submatrix_indices)]

        # Submatrix extraction for the group
        submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)]
        perm_fixed = [[submatrix_indices.index(ii),submatrix_indices.index(jj)] for ii,jj in sub_perm_fixed]
        submatrix_indices = np.array(submatrix_indices)

        DMi_sub = DMi[np.ix_(submatrix_indices, submatrix_indices)]
        DMj_sub = DMj[np.ix_(submatrix_indices, submatrix_indices)]

        # Do the QAP solve (with some atoms fixed)!
        if (len(perm_fixed) == 0):
          qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True},method='faq')
        else:
          qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True,'partial_match':perm_fixed},method='faq')

        sub_perm_qap = qapsolve["col_ind"]
        perm_qap = submatrix_indices[qapsolve["col_ind"]]

      # Update remaining rows and columns by removing assigned ones
      remaining_indices = [r for ii, r in enumerate(remaining_indices) if r not in perm_qap]

  # In general, we just want the lowest cost permutation, but return the others as well if they are of interest
  perm_qap = perm_qaps[np.argmin(cost_perm_qaps)]
  return perm_qap, perm_qaps, cost_perm_qaps


##################################################################################################################################

# A filtered QAP solver
def fqapsolve(DMi,DMj,z,z_unique_order):   # The z order determines which atoms (element) are assigned first

  n_atoms = DMi.shape[0]

  perm_qap = []
  remaining_indices = list(range(n_atoms))
  submatrix_indices = []

  # Iterate over each group of atoms (element)
  for group in z_unique_order:

      # Indices of the current group
      group_indices = [ii for ii, g in enumerate(z) if g == group]

      sub_perm_fixed = [(jj,perm_qap[ii]) for ii,jj in enumerate(submatrix_indices)]

      # Submatrix extraction for the group
      submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)]
      perm_fixed = [[submatrix_indices.index(ii),submatrix_indices.index(jj)] for ii,jj in sub_perm_fixed]
      submatrix_indices = np.array(submatrix_indices)

      DMi_sub = DMi[np.ix_(submatrix_indices, submatrix_indices)]
      DMj_sub = DMj[np.ix_(submatrix_indices, submatrix_indices)]

      # Do the QAP solve (with some atoms fixed)!

      # If this is a group with only one atom, just add the single node on
      if (len(group_indices) == 1):
        col_ind = []
        for ii in range(len(submatrix_indices)):
          flag = True
          for iijj in perm_fixed:
            if (ii == iijj[0]):
              col_ind.append(iijj[1])
              flag = False
              break
          if flag: col_ind.append(ii)

        qapsolve = {"nit":0, "col_ind":np.array(col_ind)}

      # If this is the first group of atoms, don't have any match
      elif (len(perm_fixed) == 0):
        qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True},method='faq')

      # In general, we need to keep the previous set of indices as a partial match for the next match
      else:
        qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True,'partial_match':perm_fixed},method='faq')

      sub_perm_qap = qapsolve["col_ind"]
      perm_qap = submatrix_indices[qapsolve["col_ind"]]

      # Update remaining rows and columns by removing assigned ones
      remaining_indices = [r for ii, r in enumerate(remaining_indices) if r not in perm_qap]

  return perm_qap

##################################################################################################################################

# A filtered QAP solver w/ GOAT
def fqapGOATsolve(DMi,DMj,z,z_unique_order):   # The z order determines which atoms (element) are assigned first

  n_atoms = DMi.shape[0]

  perm_qap = []
  remaining_indices = list(range(n_atoms))
  submatrix_indices = []

  NiterationsGOAT = 0

  # Iterate over each group of atoms (element)
  for group in z_unique_order:

      # Indices of the current group
      group_indices = [ii for ii, g in enumerate(z) if g == group]

      sub_perm_fixed = [(jj,perm_qap[ii]) for ii,jj in enumerate(submatrix_indices)]

      # Submatrix extraction for the group
      submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)]
      perm_fixed = [[submatrix_indices.index(ii),submatrix_indices.index(jj)] for ii,jj in sub_perm_fixed]
      submatrix_indices = np.array(submatrix_indices)

      DMi_sub = DMi[np.ix_(submatrix_indices, submatrix_indices)]
      DMj_sub = DMj[np.ix_(submatrix_indices, submatrix_indices)]

      # Do the QAP solve (with some atoms fixed)!

      # If this is a group with only one atom, just add the single node on
      if (len(group_indices) == 1):
        col_ind = []
        for ii in range(len(submatrix_indices)):
          flag = True
          for iijj in perm_fixed:
            if (ii == iijj[0]):
              col_ind.append(iijj[1])
              flag = False
              break
          if flag: col_ind.append(ii)

        qapsolve = {"nit":0, "col_ind":np.array(col_ind)}

      # If this is the first group of atoms, don't have any match
      elif (len(perm_fixed) == 0):
        qapsolve = quadratic_assignment_ot(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True},method='faq')

      # In general, we need to keep the previous set of indices as a partial match for the next match
      else:
        qapsolve = quadratic_assignment_ot(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True,'partial_match':perm_fixed},method='faq')

      NiterationsGOAT += qapsolve["nit"]
      sub_perm_qap = qapsolve["col_ind"]
      perm_qap = submatrix_indices[qapsolve["col_ind"]]

      # Update remaining rows and columns by removing assigned ones
      remaining_indices = [r for ii, r in enumerate(remaining_indices) if r not in perm_qap]

  return perm_qap, NiterationsGOAT

##################################################################################################################################

# An iterated filtered QAP solver
def ifqapsolve(ri,rj,DMi,DMj,z,z_unique_order):   # The z order determines which atoms (element) are assigned first

  n_atoms = DMi.shape[0]

  perm_qap = []
  remaining_indices = list(range(n_atoms))
  submatrix_indices = []

  Ninit = 0
  rj_init = []
  rj_init.append(rj[:,[0,1,2]])
  rj_init.append(-rj[:,[0,1,2]])
  rj_init.append(rj[:,[1,2,0]])
  rj_init.append(-rj[:,[1,2,0]])
  rj_init.append(rj[:,[2,0,1]])
  rj_init.append(-rj[:,[2,0,1]])

  rj_init.append(rj[:,[0,2,1]])
  rj_init.append(-rj[:,[0,2,1]])
  rj_init.append(rj[:,[1,0,2]])
  rj_init.append(-rj[:,[1,0,2]])
  rj_init.append(rj[:,[2,1,0]])
  rj_init.append(-rj[:,[2,1,0]])

  ifqapDMDs = []
  ifqap_perms = []

  # Iterate over groups of atoms (element) for some number of cycles
  # Although technically it could go on forever or converge quickly,
  # just cap it at ~ (the number of unique atoms)^2
  group = z_unique_order[0]
# for Niteration in range(len(z_unique_order)*(len(z_unique_order)-1)+4):
# for Niteration in range(40):
  while Ninit < 24:

      # Indices of the current group
      group_indices = [ii for ii, g in enumerate(z) if g == group]

      sub_perm_fixed = [(jj,perm_qap[ii]) for ii,jj in enumerate(submatrix_indices)]

      # Submatrix extraction for the group
      submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)]
      perm_fixed = [[submatrix_indices.index(ii),submatrix_indices.index(jj)] for ii,jj in sub_perm_fixed]
      submatrix_indices = np.array(submatrix_indices)

      DMi_sub = DMi[np.ix_(submatrix_indices, submatrix_indices)]
      DMj_sub = DMj[np.ix_(submatrix_indices, submatrix_indices)]

      # Do the QAP solve (with some atoms fixed)!
      if (len(perm_fixed) == 0):
        qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True},method='faq')
      else:
        qapsolve = quadratic_assignment(DMi_sub, DMj_sub, options={'rng': np.random.default_rng(),'maximize':True,'partial_match':perm_fixed},method='faq')

      sub_perm_qap = qapsolve["col_ind"]
      perm_qap = submatrix_indices[qapsolve["col_ind"]]

      # Update remaining rows and columns by removing assigned ones
      remaining_indices = [r for ii, r in enumerate(remaining_indices) if r not in perm_qap]


      z_other = [x for x in z_unique_order if x != group]
      if (len(z_other)==0): break

      group = np.random.choice(z_other)

      if (len(perm_qap) == n_atoms):
        ifqapDMDs.append(np.sum((DMi-DMj[perm_qap,:][:,perm_qap])**2))
        ifqap_perms.append(perm_qap)
        print(Ninit, 1, group, ifqapDMDs[-1], ifqap_perms[-1], len(remaining_indices), perm_fixed)

        Pp, Qp = QQkabsch(ri,rj[perm_qap,:])
        Pm, Qm = QQkabsch(-ri,rj[perm_qap,:])
        Pa, Qa = Pm, Qm
        if (np.sum((Pp-Qp)**2) < np.sum((Pm-Qm)**2)):
            Pa, Qa = Pp, Qp

#       if (Niteration % 4 == 3):
#         Pa, Qa = Pp, Qp
#       elif (Niteration % 4 == 2):
#         Pa, Qa = ri, rj[perm_qap,:]
#       elif (Niteration % 4 == 1):
#         Pa, Qa = -ri, rj[perm_qap,:]

#       if (Ninit % 9 == 1):
#         Pa, Qa = Pp, Qp
#       elif (Ninit % 9 > 1):
#         Pa, Qa = ri, rj_init[Ninit%6]

#       if (Ninit % 21 > 8):
#         Pa, Qa = ri, rj_init[Ninit%12]
#       elif (Ninit % 21 > 4):
#         Pa, Qa = Pm, Qm

        if (Ninit % 9 == 8):
          Pa, Qa = Pp, Qp
        elif (Ninit % 9 == 7):
          Pa, Qa = ri, rj_init[0][perm_qap,:]
        elif (Ninit % 9 == 6):
          Pa, Qa = ri, rj_init[1][perm_qap,:]
        elif (Ninit % 9 == 5):
          Pa, Qa = ri, rj_init[2][perm_qap,:]
        elif (Ninit % 9 == 4):
          Pa, Qa = ri, rj_init[3][perm_qap,:]
        elif (Ninit % 9 == 3):
          Pa, Qa = ri, rj_init[4][perm_qap,:]
        elif (Ninit % 9 == 2):
          Pa, Qa = ri, rj_init[5][perm_qap,:]

#       if (Niteration % 2 == 0):
#         Pa, Qa = QQkabsch(-ri,rj[perm_qap,:])
#       else:
#         Pa, Qa = QQkabsch(ri,rj[perm_qap,:])

        Ninit += 1

        if True:
          Qcost = np.full(DMi.shape,np.inf)
          for ii in range(n_atoms):
            Qcost[ii,ii] = sum((Pa[ii] - Qa[ii])**2)
            for jj in range(ii+1,n_atoms):
              if (z[ii] == z[jj]):
                Qcost[ii,jj] = sum((Pa[ii] - Qa[jj])**2)
                Qcost[jj,ii] = sum((Pa[jj] - Qa[ii])**2)

          lap_row_ind, lap_col_ind = linear_sum_assignment(Qcost)
          old_perm_qap = perm_qap.copy()
          perm_qap = old_perm_qap[lap_col_ind]

          ifqapDMDs.append(np.sum((DMi-DMj[perm_qap,:][:,perm_qap])**2))
          ifqap_perms.append(perm_qap)
          print(Ninit, 2, group, ifqapDMDs[-1], ifqap_perms[-1], len(remaining_indices), perm_fixed)

          remaining_indices = [ii for ii, g in enumerate(z) if g == group]
          submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)))]

          perm_qap = perm_qap[submatrix_indices]

        elif True:
          for n_init in range(6):
            Pa, Qa = ri, rj_init[n_init][perm_qap,:]
            Qcost = np.full(DMi.shape,np.inf)
            for ii in range(n_atoms):
              Qcost[ii,ii] = sum((Pa[ii] - Qa[ii])**2)
              for jj in range(ii+1,n_atoms):
                if (z[ii] == z[jj]):
                  Qcost[ii,jj] = sum((Pa[ii] - Qa[jj])**2)
                  Qcost[jj,ii] = sum((Pa[jj] - Qa[ii])**2)

            lap_row_ind, lap_col_ind = linear_sum_assignment(Qcost)
            old_perm_qap = perm_qap.copy()
            perm_qap = old_perm_qap[lap_col_ind]

            ifqapDMDs.append(np.sum((DMi-DMj[perm_qap,:][:,perm_qap])**2))
            ifqap_perms.append(perm_qap)
            print(Ninit, n_init, group, ifqapDMDs[-1], ifqap_perms[-1], len(remaining_indices), perm_fixed)

          remaining_indices = [ii for ii, g in enumerate(z) if g == group]
          submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)))]

          perm_qap = perm_qap[submatrix_indices]

        else:
          remaining_indices = [ii for ii, g in enumerate(z) if g == group]
          submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)))]

          Nnext = len(submatrix_indices)
          Qcost = np.full((Nnext,Nnext),np.inf)
          for ii,iatom in enumerate(submatrix_indices):
            Qcost[ii,ii] = sum((Pa[iatom] - Qa[iatom])**2)
            for jj,jatom in enumerate(submatrix_indices):
              if (z[iatom] == z[jatom]):
                Qcost[ii,jj] = sum((Pa[iatom] - Qa[jatom])**2)
                Qcost[jj,ii] = sum((Pa[jatom] - Qa[iatom])**2)

          lap_row_ind, lap_col_ind = linear_sum_assignment(Qcost)
          old_perm_qap = perm_qap.copy()
          perm_qap[submatrix_indices] = old_perm_qap[np.array(submatrix_indices)[lap_col_ind]]

          ifqapDMDs.append(np.sum((DMi-DMj[perm_qap,:][:,perm_qap])**2))
          ifqap_perms.append(perm_qap)
          print(Ninit, 2, group, ifqapDMDs[-1], ifqap_perms[-1], len(remaining_indices), perm_fixed)

          perm_qap = perm_qap[submatrix_indices]

        continue


        Pp, Qp = QQkabsch(ri,rj[perm_qap,:])
        Pm, Qm = QQkabsch(-ri,rj[perm_qap,:])
        Pa, Qa = Pm, Qm
        if (np.sum((Pp-Qp)**2) < np.sum((Pm-Qm)**2)):
          Pa, Qa = Pp, Qp

#       if (Niteration % 3 == 0):
#         Pa, Qa = QQkabsch(-ri,rj[perm_qap,:])
#       else:
#         Pa, Qa = QQkabsch(ri,rj[perm_qap,:])

        Qcost = np.full(DMi.shape,np.inf)
        for ii in range(n_atoms):
          Qcost[ii,ii] = sum((Pa[ii] - Qa[ii])**2)
          for jj in range(ii+1,n_atoms):
            if (z[ii] == z[jj]):
              Qcost[ii,jj] = sum((Pa[ii] - Qa[jj])**2)
              Qcost[jj,ii] = sum((Pa[jj] - Qa[ii])**2)

        lap_row_ind, lap_col_ind = linear_sum_assignment(Qcost)
        old_perm_qap = perm_qap.copy()
        perm_qap = old_perm_qap[lap_col_ind]

        ifqapDMDs.append(np.sum((DMi-DMj[perm_qap,:][:,perm_qap])**2))
        ifqap_perms.append(perm_qap)
        print(Ninit, 3, group, ifqapDMDs[-1], ifqap_perms[-1], len(remaining_indices), perm_fixed)


      if (len(remaining_indices) == 0):
        remaining_indices = [ii for ii, g in enumerate(z) if g == group]
        submatrix_indices = [ii for ii in range(n_atoms) if ((not (ii in remaining_indices)))]
        perm_qap = perm_qap[submatrix_indices]

  ii = np.argmin(ifqapDMDs)
  perm_qap = ifqap_perms[ii]

  return perm_qap



##################################################################################################################################

# A filtered LAP solver
def flapsolve(DMi,DMj,z,z_unique_order,pre_sorted=False):

  n_atoms = DMi.shape[0]

  perm_lap = []
  remaining_indices = list(range(n_atoms))
  done_indices = []
  submatrix_indices = []

  total_perm_lap = np.arange(n_atoms)

  if pre_sorted:
    sorted_DMi = DMi.copy()
    sorted_DMj = DMj.copy()

  # Create a sorted version of the DMs
  else:
    sorted_DMi = DMi.copy()
    sorted_DMj = DMj.copy()
    for group in z_unique_order:
      group_indices = np.array([ii for ii, g in enumerate(z) if g == group])

      for ii in range(n_atoms):
        sorted_DMi[ii,group_indices] = np.sort(DMi[ii,group_indices])
        sorted_DMj[ii,group_indices] = np.sort(DMj[ii,group_indices])


  # Iterate over each group of atoms (element)
  for group in z_unique_order:

      # Indices of the current group
      group_indices = np.array([ii for ii, g in enumerate(z) if g == group])

      # Indices which have been sorted (or are about to be sorted) so far
      done_indices = np.array([ii for ii in range(n_atoms) if ((not (ii in remaining_indices)) or z[ii] == group)])

      # Now sort the rows (because the linear assignments require getting rid of column permutations)
      Nrows = len(group_indices)
      cost_perm_matrix = np.zeros((Nrows,Nrows))
      for ii,iatom in enumerate(group_indices):
        for jj,jatom in enumerate(group_indices):
          # Method A: Just look at the distances within the group
#         cost_perm_matrix[ii,jj] = sum((sorted_DMi[iatom,group_indices] - sorted_DMj[jatom,group_indices])**2)
          # Method B: Look at all distances remaining
#         cost_perm_matrix[ii,jj] = sum((sorted_DMi[iatom,remaining_indices] - sorted_DMj[jatom,remaining_indices])**2)
          # Method C: Look at all distances
          cost_perm_matrix[ii,jj] = sum((sorted_DMi[iatom,:] - sorted_DMj[jatom,:])**2)

          # Method D: Look at all distances, and fix those which are already set
#         cost_perm_matrix[jj,ii] = sum((sorted_DMi[iatom,remaining_indices] - sorted_DMj[jatom,remaining_indices])**2) + sum((DMi[iatom,done_indices] - DMj[jatom,done_indices])**2)

      # Do the LAP solve!
      lap_row_ind, lap_col_ind = linear_sum_assignment(cost_perm_matrix)

      total_perm_lap[group_indices] = group_indices[lap_col_ind]
      perm_lap = total_perm_lap[done_indices]

      # Update remaining rows and columns by removing assigned ones
      remaining_indices = np.array([r for ii, r in enumerate(remaining_indices) if r not in perm_lap])

  return perm_lap


##################################################################################################################################


# For RMSD calculations

# Get the optimal (translation and rotation) RMSD using the Kasch algorithm
def QQkabsch(P, Q):

  # Calculate the shifted geometries
  Pt = P - Qcentroid(P)
  Qt = Q - Qcentroid(Q)

  # Computation of the covariance matrix
  C = np.dot(Pt.T, Qt)

  # Computation of the optimal rotation matrix
  V, S, W = np.linalg.svd(C)
  d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

  if(d):
    S[-1] = -S[-1]
    V[:,-1] = -V[:,-1]

  # Create Rotation matrix U
  U = np.dot(V, W)

# return QQrmsd(np.dot(Pt,U),Qt)
  return np.dot(Pt,U), Qt

# Get the centroid of a set of coordintes
def Qcentroid(X):
  C = sum(X)/len(X)
  return C


# Get the RMSD of two sets of coordinates
def QQrmsd(V, W):
  D = len(V[0])
  N = len(V)
  rmsd = 0.0
  for v, w in zip(V, W):
    rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])
  return np.sqrt(rmsd/N)

##################################################################################################################################

# Made with help from Chat GPT
# An example eigenvector QAP solver (Umeyama-like)

def spectral_linear_assignment(C, F, groups=None):
    """
    Approximate QAP solution using spectral methods and linear sum assignment.
    
    C: Cost matrix (N x N)
    F: Flow matrix (N x N)
    groups: Optional list of group indices for constrained permutation (length N)
    """
    # Step 1: Eigen-decomposition of C and F
    eigvals_C, eigvecs_C = np.linalg.eigh(C)
    eigvals_F, eigvecs_F = np.linalg.eigh(F)
    
    # Use the leading eigenvectors
    v = eigvecs_C[:, -1]  # Largest eigenvector of C
    w = eigvecs_F[:, -1]  # Largest eigenvector of F
    
    # Create the overall cost matrix (overlap matrix) from the eigenvectors
    overlap_matrix = -np.fabs(eigvecs_C).dot(np.fabs(eigvecs_F).T)
    cost_matrix = np.full(overlap_matrix.shape, np.inf)

    # But only allow reassignments within groups (atomic elements)
    if groups is None:
      cost_matrix = overlap_matrix
    else:
      for group_indices in groups:
        cost_matrix[np.ix_(group_indices,group_indices)] = overlap_matrix[np.ix_(group_indices,group_indices)]

    # Solve the LAP
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    perm = col_ind

    # Compute the spectral cost
    if False:
      P = np.zeros(C.shape)
      for r, c in zip(row_ind, col_ind):
        P[r, c] = 1
      cost = np.trace(C @ P @ F @ P.T)
    else:
      cost = 0.0
    
    return perm, cost

##################################################################################################################################

