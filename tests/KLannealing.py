import numpy as np
import matplotlib.pyplot as plt

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
	L = np.ones(n_iter) * stop
	period = n_iter/n_cycle
	step = (stop-start)/(period*ratio) # linear schedule

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i+c*period) < n_iter):
			L[int(i+c*period)] = v
			v += step
			i += 1
	return L

def plot_beta(beta_list):

	plt.figure()
	plt.plot(beta_list)
	plt.show()


def main():
	n = 10

	beta_list = []
	"""

	for i in range(1,n):
		beta = 0
		beta = frange_cycle_linear(i)
		print("New")
		print(beta)
		beta_list.append(beta)
	"""
	#plot_beta(beta_list)

	print(frange_cycle_linear(100))

if __name__ == "__main__":
	main()