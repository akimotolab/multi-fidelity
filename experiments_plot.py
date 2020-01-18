import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def percentile(dat_all):
    time_array = np.sort(np.concatenate([dat[:, 0] for dat in dat_all]))
    dat_array = np.empty((len(time_array), len(seed_list)))
    for i in range(len(dat_all)):
        idx = 0
        val = np.inf
        dat = dat_all[i]
        for j in range(time_array.shape[0]):
            if time_array[j] >= dat[idx, 0]:
                val = dat[idx, 1]
                idx += 1
            dat_array[j, i] = val
            if idx == dat.shape[0]:
                dat_array[j:, i] = val
                break
    p10, p50, p90 = np.percentile(dat_array, [10, 50, 90], axis=1)
    return time_array, p10, p50, p90


seed_list = list(range(1, 11))
fidelity_list = [4, 7, 11, 14, 17, 20] # 1 failed
lr_u_list = [1, 3 ,5, 7, 9]
linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']

cmap1 = plt.get_cmap('jet')
cnorm1 = mpl.colors.Normalize(vmin=0, vmax=len(fidelity_list) + len(lr_u_list) -1)
smap1 = mpl.cm.ScalarMappable(norm=cnorm1, cmap=cmap1)

# fixed

fig = plt.figure(figsize=(3, 3))
ax = plt.gca()
ax.grid(True)
ax.grid(which='major', linewidth=0.50)
ax.grid(which='minor', linewidth=0.25)

maxsec = 86400

for k in range(len(fidelity_list)):
    fidelity = fidelity_list[k]
    dat_all = []
    for seed in seed_list:
        filename = '../dat/ddcma_arch_mcr_fix{}_seed{}_maxsec{}_reeval.txt'.format(str(fidelity), str(seed), str(maxsec))
        dat = np.loadtxt(filename)
        # compression
        dat_len = len(dat)
        dx = dat_len // 1000
        if dx > 1:
            new_dat = dat[:dat_len//10000]
            new_dat = np.vstack((new_dat, dat[dat_len//10000:dat_len//1000:dx//1000] if dx//1000 > 1 else dat[dat_len//10000:dat_len//1000]))
            new_dat = np.vstack((new_dat, dat[dat_len//1000:dat_len//100:dx//100] if dx//100 > 1 else dat[dat_len//1000:dat_len//100]))
            new_dat = np.vstack((new_dat, dat[dat_len//100:dat_len//10:dx//10] if dx//10 > 1 else dat[dat_len//100:dat_len//10]))
            new_dat = np.vstack((new_dat, dat[dat_len//10::dx] if dx > 1 else dat[dat_len//10:]))
            dat = new_dat
        dat_all += [dat[:, (1, -1)]]
    time_array, p10, p50, p90 = percentile(dat_all)
    ax.fill_between(time_array, y1=p10, y2=p90, color=smap1.to_rgba(k), alpha=0.2)
    ax.plot(time_array, p50, color=smap1.to_rgba(k), alpha=1, label=r'$k='+str(fidelity)+'$', linestyle=linestyle[k])

ax.set_xlabel('cpu time')
ax.set_ylabel('objective value')
ax.set_xlim([1e1, maxsec])
ax.set_ylim([3e2, 1e4])
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.savefig('../dat/ddcma_arch_mcr_fix.pdf', tight_layout=True)
plt.savefig('../dat/ddcma_arch_mcr_fix.png', tight_layout=True)

# adaptive

fig = plt.figure(figsize=(3, 3))
ax = plt.gca()
ax.grid(True)
ax.grid(which='major', linewidth=0.50)
ax.grid(which='minor', linewidth=0.25)

maxsec = 86400

for k, lr_u in enumerate(lr_u_list):
    dat_all = []
    for seed in seed_list:
        filename = '../dat/ddcma_arch_mcr_adapt{}_seed{}_maxsec{}_reeval.txt'.format(str(lr_u), str(seed), str(maxsec))
        dat = np.loadtxt(filename)
        dat_all += [dat[:, (1, -1)]]
    time_array, p10, p50, p90 = percentile(dat_all)
    ax.fill_between(time_array, y1=p10, y2=p90, color=smap1.to_rgba(len(fidelity_list) + k), alpha=0.2)
    ax.plot(time_array, p50, color=smap1.to_rgba(len(fidelity_list) + k), alpha=1, label=r'$c_\tau^+=0.'+str(lr_u)+'$', linestyle=linestyle[k])

ax.set_xlabel('cpu time')
ax.set_ylabel('objective value')
ax.set_xlim([1e1, maxsec])
ax.set_ylim([3e2, 1e4])
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.savefig('../dat/ddcma_arch_mcr_ada.pdf', tight_layout=True)
plt.savefig('../dat/ddcma_arch_mcr_ada.png', tight_layout=True)

# Combined

seed_list = list(range(1, 11))
fidelity_list = [4, 7, 11, 14, 17, 20] # 1 failed
lr_u_list = [1, 3 ,5, 7, 9]

fig = plt.figure(figsize=(3, 3))
ax = plt.gca()
ax.grid(True)
ax.grid(which='major', linewidth=0.50)
ax.grid(which='minor', linewidth=0.25)

maxsec = 86400

for k in range(len(fidelity_list)):
    fidelity = fidelity_list[k]
    if k in (2, 5):
        dat_all = []
        for seed in seed_list:
            filename = '../dat/ddcma_arch_mcr_fix{}_seed{}_maxsec{}_reeval.txt'.format(str(fidelity), str(seed), str(maxsec))
            dat = np.loadtxt(filename)
            # compression
            dat_len = len(dat)
            dx = dat_len // 1000
            if dx > 1:
                new_dat = dat[:dat_len//10000]
                new_dat = np.vstack((new_dat, dat[dat_len//10000:dat_len//1000:dx//1000] if dx//1000 > 1 else dat[dat_len//10000:dat_len//1000]))
                new_dat = np.vstack((new_dat, dat[dat_len//1000:dat_len//100:dx//100] if dx//100 > 1 else dat[dat_len//1000:dat_len//100]))
                new_dat = np.vstack((new_dat, dat[dat_len//100:dat_len//10:dx//10] if dx//10 > 1 else dat[dat_len//100:dat_len//10]))
                new_dat = np.vstack((new_dat, dat[dat_len//10::dx] if dx > 1 else dat[dat_len//10:]))
                dat = new_dat
            dat_all += [dat[:, (1, -1)]]
        time_array, p10, p50, p90 = percentile(dat_all)
        ax.fill_between(time_array, y1=p10, y2=p90, color=smap1.to_rgba(k), alpha=0.2)
        ax.plot(time_array, p50, color=smap1.to_rgba(k), alpha=1, label=r'$k='+str(fidelity)+'$', linestyle=linestyle[k])

for k, lr_u in enumerate(lr_u_list):
    if k in (1, 2):
        dat_all = []
        for seed in seed_list:
            filename = '../dat/ddcma_arch_mcr_adapt{}_seed{}_maxsec{}_reeval.txt'.format(str(lr_u), str(seed), str(maxsec))
            dat = np.loadtxt(filename)
            dat_all += [dat[:, (1, -1)]]
        time_array, p10, p50, p90 = percentile(dat_all)
        ax.fill_between(time_array, y1=p10, y2=p90, color=smap1.to_rgba(len(fidelity_list) + k), alpha=0.2)
        ax.plot(time_array, p50, color=smap1.to_rgba(len(fidelity_list) + k), alpha=1, label=r'$c_\tau^+=0.'+str(lr_u)+'$', linestyle=linestyle[k])

ax.set_xlabel('cpu time')
ax.set_ylabel('objective value')
ax.set_xlim([1e1, maxsec])
ax.set_ylim([3e2, 1e4])
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.savefig('../dat/ddcma_arch_mcr_mix.pdf', tight_layout=True)
plt.savefig('../dat/ddcma_arch_mcr_mix.png', tight_layout=True)

