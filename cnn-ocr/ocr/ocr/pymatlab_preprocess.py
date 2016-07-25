# -*- coding: utf-8 -*-
import scipy.io as sio   
import numpy as np
import skimage.io
from skimage import transform as tf

'''
I have just corrected the pseudo implementation of Viterbi in Wikipedia(http://en.wikipedia.org/wiki/Viterbi_algorithm). 
From the initial (incorrect) version, it took me a while to figure out where 
I was going wrong but I finally managed it, thanks partly to Kevin Murphy's 
implementation of the viterbi_path.m in the MatLab HMM toolbox.(http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html)

In the context of an HMM object with variables as shown:
    '''
'''
hmm = HMM()
hmm.priors = np.array([0.5, 0.5]) # pi = prior probs
hmm.transition = np.array([[0.75, 0.25], # A = transition probs. / 2 states
                           [0.32, 0.68]])
hmm.emission = np.array([[0.8, 0.1, 0.1], # B = emission (observation) probs. / 3 obs modes
                         [0.1, 0.2, 0.7]])

# The Python function to run Viterbi (best-path) algorithm is below
def viterbi (self,observations):
    """Return the best path, given an HMM model and a sequence of observations"""
    # A - initialise stuff
    nSamples = len(observations[0])
    nStates = self.transition.shape[0] # number of states
    c = np.zeros(nSamples) #scale factors (necessary to prevent underflow)
    viterbi = np.zeros((nStates,nSamples)) # initialise viterbi table
    psi = np.zeros((nStates,nSamples)) # initialise the best path table
    best_path = np.zeros(nSamples); # this will be your output

    # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
    viterbi[:,0] = self.priors.T * self.emission[:,observations(0)]
    c[0] = 1.0/np.sum(viterbi[:,0])
    viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor
    psi[0] = 0;

    # C- Do the iterations for viterbi and psi for time>0 until T
    for t in range(1,nSamples): # loop through time
        for s in range (0,nStates): # loop through the states @(t-1)
            trans_p = viterbi[:,t-1] * self.transition[:,s]
            psi[s,t], viterbi[s,t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            viterbi[s,t] = viterbi[s,t]*self.emission[s,observations(t)]

        c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
        viterbi[:,t] = c[t] * viterbi[:,t]

    # D - Back-tracking
    best_path[nSamples-1] =  viterbi[:,nSamples-1].argmax() # last state
    for t in range(nSamples-1,0,-1): # states of (last-1)th to 0th time step
        best_path[t-1] = psi[best_path[t],t]

    return best_path
    '''

'''
if __name__ == "__main__":
    
    # crash with scipy higher than 1.16.0
    # Downgrading to 1.16.0 solved the issue. conda install scipy==0.16.0.
    

#    matfn=u'E:/python/测试程序/162250671_162251656_1244.mat'  
#    print('Information for opencv_cnn-ocr-detector.mat ')  
#    print(data)  
#    print('\nThe vaulue of opencv_cnn-ocr-detector:')  
#    print(data['avgFilter'])  
#    mat4py_load = data['avgFilter']  
#    x = [1, 2, 3]  
#    y = [4, 5, 6]  
#    z = [7, 8, 9]  
#    sio.savemat('saveddata.mat', {'x': x,'y': y,'z': z})   

#    matfn = u'G:/myworks/cnn-ocr/character-detector-master/scores.mat' 
#    data = sio.loadmat(matfn)
#
#    scores=data['scores']
#
#    matfn = u'G:/myworks/cnn-ocr/character-detector-master/good_idx.mat'
#    data = sio.loadmat(matfn)
#    good_idx=data['good_idx']
      

   
   a=1
    
 ''' 
