import numpy as np

def cluster_extra(bow, K, e = 0.0001, maxiter = 10):
    """

    :param bow:
        bag-of-word matrix of (num_doc, V), where V is the vocabulary size
    :param K:
        number of topics
    :return:
        word-topic matrix of (V, K)
    """
    np.random.seed(0)
    # Let's define some parameters
    D = bow.shape[0] # number of documents
    W = bow.shape[1] # number of words
    
    # Initialization of parameters
    w_condi_z = np.random.rand(W, K)
    d_condi_z = np.random.rand(D, K)
    
    z_condi_dw = np.zeros((K, D, W))
    
    # Normalizing above generated parameters
    for z in range(K):
        temp_sum = w_condi_z[:,z].sum()
        w_condi_z[:,z] = w_condi_z[:,z]/temp_sum
        
        temp_sum = d_condi_z[:,z].sum()
        d_condi_z[:,z] = d_condi_z[:,z]/temp_sum
    
    # Initializing p(z)
    pi = np.random.rand(K)
    pi = pi/pi.sum()
    
    prev_log_like = 0
    
    for zz in range(maxiter):
        # E step --------------------------------------
        sum_pzdw = np.zeros((D,W))
        for z in range(K):
            arr_w = w_condi_z[:,z]
            arr_d = d_condi_z[:,z]
            
            pzdw = pi[z]*np.matmul(arr_d.reshape(len(arr_d),1), arr_w.reshape(1, len(arr_w)))
            z_condi_dw[z,:,:] = pzdw
            
            sum_pzdw = sum_pzdw+pzdw
            
        for z in range(K):
            z_condi_dw[z,:,:] = z_condi_dw[z,:,:]/sum_pzdw
                
        # M step --------------------------------------
            
        ## Approach 3 
        for z in range(K):
            # updating p(w|z) [w_condi_z]
            T_pzdw = bow.multiply(z_condi_dw[z,:,:]).todense()
            sum_T_pzdw = np.sum(T_pzdw, axis = 0)
            w_condi_z[:,z] = sum_T_pzdw/sum_T_pzdw.sum()
            
            # Updating p(d|z) [d_condi_z]
            sum_T_pzdw = np.sum(T_pzdw, axis = 1)
            d_condi_z[:,z] = np.transpose(sum_T_pzdw/sum_T_pzdw.sum())
            
            # updating p(z)
            pi[z]=(np.multiply(bow.todense(),z_condi_dw[z,:,:])).sum()
            
        pi = pi/pi.sum()
        
        # Calculating log likelihood:
        pdw = np.log(np.matmul(pi*d_condi_z, w_condi_z.transpose()))
        log_like = np.multiply(pdw,bow.todense()).sum()
        
        print('Log-likelihood after iteration %d: %.1f' % (zz+1, log_like))
        if (prev_log_like != 0):
            if (abs(log_like-prev_log_like)/abs(prev_log_like) < e):
                print("breaking")
                break
            
        prev_log_like = log_like


    idx = w_condi_z
    
    return idx