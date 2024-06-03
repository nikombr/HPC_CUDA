import numpy as np 
import sys
import matplotlib.pyplot as plt

input1 = sys.argv[1]
input2 = sys.argv[2]
num = int(sys.argv[3])
bin1_filename = f"{input1}.bin"
bin2_filename = f"{input2}.bin"
fbin1 = np.fromfile(bin1_filename,dtype="double")
fbin2 = np.fromfile(bin2_filename,dtype="double")
N = round(np.shape(fbin1)[0]**(1/3))

txt1_filename = f"{input1}.txt"
txt2_filename = f"{input2}.txt"
matrix1 = np.reshape(np.loadtxt(txt1_filename),(N,N,N))
matrix2 = np.reshape(np.loadtxt(txt2_filename),(N,N,N))



if True:
    fig, axs = plt.subplots(1,3)
    plot1 = axs[0].imshow(matrix1[N//2,:,:])
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('j')
    plt.colorbar(plot1,ax=axs[0])
    plot2 = axs[1].imshow(matrix2[N//2,:,:])
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('j')
    plt.colorbar(plot2,ax=axs[1])
    plot3 = axs[2].imshow(matrix2[N//2,:,:]-matrix1[N//2,:,:])
    axs[2].set_xlabel('k')
    axs[2].set_ylabel('j')
    plt.colorbar(plot3,ax=axs[2])
    plt.savefig(f"plots/test_i_{num}.png")

    fig, axs = plt.subplots(1,3)
    plot1 = axs[0].imshow(matrix1[:,N//2,:])
    plt.colorbar(plot1,ax=axs[0])
    plot2 = axs[1].imshow(matrix2[:,N//2,:])
    plt.colorbar(plot2,ax=axs[1])
    plot3 = axs[2].imshow(matrix2[:,N//2,:]-matrix1[:,N//2,:])
    plt.colorbar(plot3,ax=axs[2])
    plt.savefig(f"plots/test_j_{num}.png")

    fig, axs = plt.subplots(1,3)
    plot1 = axs[0].imshow(matrix1[:,:,N//2])
    plt.colorbar(plot1,ax=axs[0])
    plot2 = axs[1].imshow(matrix2[:,:,N//2])
    plt.colorbar(plot2,ax=axs[1])
    plot3 = axs[2].imshow(matrix2[:,:,N//2]-matrix1[:,:,N//2])
    plt.colorbar(plot3,ax=axs[2])
    plt.savefig(f"plots/test_k_{num}.png")

diff = np.abs(fbin2-fbin1)
print(np.amax(diff))
