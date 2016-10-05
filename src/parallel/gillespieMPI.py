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

def gillespie(IC, time, rates):
	t = 0.0
	m = IC
	while t < time:	
		# Compute propensity
		prop1 = rates[0]
		prop2 = rates[1]*m
		propTot = prop1 + prop2

		# Draw random numbers
		r = npr.rand(2)

		if r[1] < prop1/propTot:
			# Transcription event
			m += 1
		else:
			# Degradation event
			m -= 1

		# Update time
		t += math.log(1.0/r[0])/(propTot)

	return m

def simulate(k1,k2,d,t):	
	# Set up initial conditions
	if rank == 0:
		ICs = npr.poisson(k1/d,size)
	else:
		ICs = None
	# Send initial conditions to each rank
	IC = comm.scatter(ICs, root=0)

	# Run simulation
	print("Running on rank: {rank} with initial count: {ic}".format(rank=rank, ic=IC))
	soln = gillespie(IC, t, [k2,d])

	# Gather results and return
	soln = comm.gather(soln, root=0)

	return ICs, soln


if __name__ == "__main__":
	# Get arguments
	parser = argparse.ArgumentParser(description="Compute induction of a gene.")
	parser.add_argument("-k1",help="Initial transcription rate (/s).")
	parser.add_argument("-k2",help="Final transcription rate (/s).")
	parser.add_argument("-d", help="Degradation rate (/molecule/s).")
	parser.add_argument("-t", help="Total simulation time (s).")
	args = parser.parse_args()

	# Run Simulation
	ics, soln = simulate(*map(float,[args.k1, args.k2, args.d, args.t]))

	# Plot results
	if rank == 0:
		plt.figure()
		kargs = {"normed":True, "alpha":0.3}
		plt.hist(ics, bins=max(ics)-min(ics),   label="Initial Count", **kargs)
		plt.hist(soln,bins=max(soln)-min(soln), label="Final Count", **kargs)
		plt.legend()
		plt.savefig("MDist.pdf")

