from sys import path
import os
path.append(os.getcwd() + "\\classes")
import muon
import plot
import time


start_time = time.time()

tau = 2.2
num = 1000
sim_run = 500
outfile = 'single.png'
outfile_1 = 'full.png'

mu_1 = muon.Muon_decay(tau,num,sim_run)

x = mu_1.r_full()
x_1 = mu_1.r_single()

plot_1 = plot.Hist(x_1,outfile)
plot.Hist.draw_hist(plot_1)

plot_2 = plot.Hist(x,outfile_1)
plot.Hist.draw_hist(plot_2)

print("--- %s seconds ---" % (time.time() - start_time))