#!/usr/bin/env python
import os
import sys
import math
import argparse
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

# Global communicator
comm = MPI.COMM_WORLD
# Process rank and communicator size
rank = comm.Get_rank()
size = comm.Get_size()


def heatEquation(a, T0, TL, TR, t, delt, ts, x, L):
	# Initialization
	dx = float(L)/float(x) # Width of cells
	dxdx = dx*dx
	cellsPerRank = int(x/size) # Count of cells per rank
	cellsPerRankTuple = size*[cellsPerRank]
	offsets = size*[0]
	for i,_ in enumerate(cellsPerRankTuple):
		if i == 0:
			continue
		offsets[i] = offsets[i-1] + cellsPerRankTuple[i-1]

	# Time variables	
	tcur = 0.0 
	timesteps = ts * np.arange(int(t/ts),dtype=np.float64)
	temperatures = []

	# Set up initial conditions
	if rank == 0:
		domain = np.full(x, T0, dtype=np.float64)
	else:
		domain = None

	# Set up local chunks for computation
	domainLocal_0 = np.zeros(cellsPerRank)
	domainLocal_1 = np.zeros(cellsPerRank)
	# Send initial conditions to each rank
	comm.Scatterv([domain, cellsPerRankTuple, offsets, MPI.DOUBLE], domainLocal_0, root=0)

	lastSave = ts
	while tcur <= t:
		# Update time
		tcur += delt 
		lastSave += delt

		L = None # Left boundary
		R = None # Right boundary

		# Send left
		if rank > 0:
			comm.send(domainLocal_0[0], dest=rank-1, tag=2)
		else:
			L = TL
		if rank < size-1:
			R = comm.recv(source=rank+1, tag=2)

		# Send right
		if rank < size-1:
			comm.send(domainLocal_0[-1], dest=rank+1, tag=3)
		else:
			R = TR
		if rank > 0:
			L = comm.recv(source=rank-1, tag=3)

		# Compute heat flux 
		for i in range(1,cellsPerRank-1):
			# Central difference
			dTdt = (domainLocal_0[i+1] - 2.0*domainLocal_0[i]+ domainLocal_0[i-1])/(dxdx)  
			# Update temperature
			domainLocal_1[i] = a*delt*dTdt + domainLocal_0[i]

		# Fix boundaries
		domainLocal_1[0]  = a*delt*(domainLocal_0[1] - 2.0*domainLocal_0[0] + L) /(dxdx) + domainLocal_0[0]
		domainLocal_1[-1] = a*delt*(R - 2.0*domainLocal_0[-1] + domainLocal_0[-2])/(dxdx) + domainLocal_0[-1]

		# Save if necessary
		if lastSave > ts:
			lastSave = 0.0
			domainSave = np.zeros(x)
			comm.Gatherv(domainLocal_1,[domainSave,cellsPerRankTuple, offsets, MPI.DOUBLE],root=0)
			if rank == 0:
				print("Time:",tcur,"s",int(tcur/delt),"of",int(t/delt))
				temperatures.append(domainSave)

		# Swap array pointers
		tmp = domainLocal_0
		domainLocal_0 = domainLocal_1
		domainLocal_1 = tmp

		comm.Barrier()

	# Return solutions
	return timesteps, temperatures


if __name__ == "__main__":
	# Get arguments
	parser = argparse.ArgumentParser(description="Compute induction of a gene.")
	parser.add_argument("-a",   help="Thermal diffusivity (m^2/s).")
	parser.add_argument("-T0",  help="Initial temperature throughout bar (K).")
	parser.add_argument("-TL",  help="Constant temperature on left boundary (K).")
	parser.add_argument("-TR",  help="Constant temperature on right boundary (K).")
	parser.add_argument("-t",   help="Total simulation time (s).")
	parser.add_argument("-tdel",help="Timestep length (s).")
	parser.add_argument("-ts",  help="Save frequency (s).")
	parser.add_argument("-x",   help="Number of cells.")
	parser.add_argument("-L",   help="Domain length (m).")
	args = parser.parse_args()

	# Run Simulation
	times, domain = heatEquation(*list(map(float,[args.a, args.T0, args.TL, args.TR, args.t, args.tdel, args.ts, args.x, args.L])))

	# Plot results
	if rank == 0:
		plt.figure()
		for i,t in enumerate(times):
			plt.plot(domain[i], label="%fs"%(t))
		plt.xlabel("Position (m)")
		plt.ylabel("Temperature (K)")
		plt.legend()
		plt.savefig("TDist.pdf")

