import sys
import numpy as np

def KMeans(X, clusters=5, iterations=10):
    len_X = X.shape[0]
    indices = np.random.randint(0, len_X, size=clusters)
    mu = X[indices]
    c = np.zeros(len_X)
    for itern in range(iterations):
        for i in range(len_X):
            dist = np.linalg.norm(mu-X[i],2,1)
            c[i] = np.argmin(dist)
        n = np.bincount(c.astype(np.int64), weights=None, minlength=clusters)
        for k in range(clusters):
            indices = np.where(c==k)[0]
            mu[k] = (np.sum(X[indices],axis=0))/float(n[k])
        filename = "centroids-" + str(itern+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")
  
def EMGMM(X, clusters=5, iterations=10):
    len_X = X.shape[0]
    dim = X.shape[1]
    pi = np.ones(clusters)/clusters
    sigma = np.dstack([np.eye(dim)]*clusters)
    phi = np.zeros((len_X,clusters))
    mu = X[np.random.randint(0, len_X, size=clusters)]
         
    for itern in range(iterations):
        for i in range(len_X):
            for k in range(clusters):
                sigma_inv = np.linalg.inv(sigma[:,:,k])
                sigma_ird = 1 / np.sqrt(np.linalg.det(sigma[:,:,k]))
                diff = X[i] - mu[k]
                phi[i,k] = pi[k]*sigma_ird*((2*np.pi)**(-dim/2))*np.exp(-0.5*np.dot(diff, np.dot(sigma_inv, diff.T)))
            phi_sum = np.sum(phi[i])
            if phi_sum == 0:
                phi[i] = pi/clusters
            else:
                phi[i] /= phi_sum
        n = np.sum(phi, axis=0)
        pi = n/float(len_X)
        for k in range(clusters):
            if n[k]==0:
                mu = X[np.random.randint(0, len_X, size=clusters)]
                sigma[:,:,k] = np.eye(dim)
            else:
                mu[k] = np.dot(phi[:,k].T, X)/float(n[k])
                sigma[:,:,k] = np.zeros((dim, dim))
                for i in range(len_X):
                    diff = X[i] - mu[k] 
                    sigma[:,:,k] += phi[i,k]*np.outer(diff, diff)
                sigma[:,:,k] /= n[k]

        filename = "pi-" + str(itern+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(itern+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")
        
        for k in range(clusters):
            filename = "Sigma-" + str(k+1) + "-" + str(itern+1) + ".csv" 
            np.savetxt(filename, sigma[:,:,k], delimiter=",")


def main():  
    X = np.genfromtxt(sys.argv[1], delimiter=',')
    KMeans(X)
    EMGMM(X)

if __name__ == "__main__":
    main()