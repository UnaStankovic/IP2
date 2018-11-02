import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import copy

def main():
	# Ucitavanje podataka iz tekstualnog fajla
	file = open('ecg.txt')
	lines = file.read().split('\n')
	n = len(lines)
	Y = [0 for i in range(n)]
	X = [0 for i in range(n)]
	for i in range(n):
		line = lines[i].split(' ')
		# Prva vrednost u liniji je timestamp,
		# druga vrednost je amplituda signala otkucaja srca
		X[i] = float(line[0])
		Y[i] = float(line[1])
	Y = np.array(Y)
	# Primena brze Furijeove transformacije
	sig_fft = fftpack.fft(Y)
	# Racunanje intenziteta frekcencija
	power = np.abs(sig_fft)
	# Racunanje skale frekvencija
	sample_freq = fftpack.fftfreq(Y.size, d=(X[1]-X[0]))
	plt.plot(sample_freq, power)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('power')
	plt.show()
	low_freq_fft = sig_fft.copy()
	# Frekvencije ispod 0.5Hz nisu normalne frekvencije otkucaja
	# pa mogu biti uklonjene kako bi se signal "ispravio"
	low_freq_fft[np.abs(sample_freq) <= 0.5] = 0
	# Primena inverzne brze Furijeove transformacije
	# kako bi se od frekvencija ponovo dobila suma koja
	# predstavlja signal (sada filtrirani u odnosu na polazni)
	filtered_sig = fftpack.ifft(low_freq_fft)
	# Prag intenziteta iznad koga ce se smatrati da je signal otkucaja
	threshold = 0.5
	# Diskretizacija signala radi pronalazenja vrhova koji predstavljaju
	# otkucaje srca
	filtered_sig[filtered_sig < 0.5] = 0
	filtered_sig[filtered_sig > 0] = 1
	count = 0
	# Pronalazenje vrhova koji predstavljaju otkucaje
	for i in range(1, filtered_sig.shape[0]):
	 	if filtered_sig[i - 1] == 0 and filtered_sig[i] == 1:
	 		count += 1
	# Brzina pulsa
	print("BPM: {}".format(count / X[-1] * 60))

	# Grafik polaznog i filtriranog signala
	plt.plot(X[:2000], Y[:2000], label='Original signal')
	plt.plot(X[:2000], filtered_sig[:2000], linewidth=3, label='Filtered signal')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.show()

if __name__ == "__main__":
	main()