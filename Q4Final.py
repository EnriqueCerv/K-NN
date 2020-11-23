# %%
import numpy as np
import matplotlib.pyplot as plt 
import math 

# %% Loading up all the txt files

MNIST_Train = np.loadtxt('MNIST-Train-cropped.txt').reshape(10000,784).T
#first_col = MNIST_Train[:,0].reshape(28,28).T
#plt.imshow(first_col)

MNIST_Train_Labels = np.loadtxt('MNIST-Train-Labels-cropped.txt')
#MNIST_Train_Labels[0]

MNIST_Test = np.loadtxt('MNIST-Test-cropped.txt').reshape(2000,784).T 
#first_col_test = MNIST_Test[:,0].reshape(28,28).T
#plt.imshow(first_col_test)

MNIST_Test_Labels = np.loadtxt('MNIST-Test-Labels-cropped.txt')

print('Done')

# %% Selecting the required columns from the training images

def img_selector(num, train_arr=MNIST_Train, train_arr_lab=MNIST_Train_Labels, test_arr=MNIST_Test, test_arr_lab = MNIST_Test_Labels):
    mask_train = (train_arr_lab==num)
    img_train = train_arr[:,mask_train]
    
    mask_test = (test_arr_lab==num)
    img_test = test_arr[:,mask_test]
    
    return img_train, img_test


#To test all the img_train work just set num. Below num = 6
#img_six, img_six_test = img_selector(6)
#col1, col2 = img_six[:,3].reshape(28,28).T, img_six_test[:,3].reshape(28,28).T
#plt.imshow(col1)
#plt.imshow(col2)


# The following splits training set into training and validating
def split_training(arr, perc): #Percentage is 80%=0.8
    col = arr.shape[1]    # Here, row should be 784
    cut_off = math.floor(col*perc)
    train = arr[:,:cut_off]
    val = arr[:,cut_off:]
    return train, val

print('Done')

# %% Preliminaries to final binary selector

def binary_sel_prep(num1, num2, perc):
    #Obtaining the required images from the original training and testing cropped files
    num1_tr, num1_test = img_selector(num1)
    num2_tr, num2_test = img_selector(num2)

    #Splitting the training sets for the two numbers into training and validating
    num1_train, num1_val = split_training(num1_tr, perc)
    num2_train, num2_val = split_training(num2_tr, perc)

    # Joining the training sets for the two numbers into a big training set
    # And creating appropriate \pm 1 labels for each 'half' of the training set
    train_set = np.hstack((num1_train, num2_train))
    train_set_lab = np.hstack((np.ones(num1_train.shape[1]), np.negative(np.ones(num2_train.shape[1]))))
    
    # Joining the validating sets for the two numbers into a big validating set
    # And creating appropriate \pm 1 labels for each 'half' of the validating set
    val_set = np.hstack((num1_val, num2_val))
    val_set_lab = np.hstack((np.ones(num1_val.shape[1]), np.negative(np.ones(num2_val.shape[1]))))

    # Finally joining the testing sets for the two numbers and creating the labels.
    test_set = np.hstack((num1_test, num2_test))
    test_set_lab = np.hstack((np.ones(num1_test.shape[1]), np.negative(np.ones(num2_test.shape[1]))))

    return train_set, train_set_lab, val_set, val_set_lab, test_set, test_set_lab
    
# Just testing that the above works:
# For training set
#a = binary_sel_prep(0,1,0.8)[0]
#b = binary_sel_prep(0,1,0.8)[1]
#col1 = a[:,0].reshape(28,28).T
#col1_lab = b[0]
#colm1 = a[:,-1].reshape(28,28).T
#colm1_lab = b[-1]
#plt.imshow(col1), col1_lab
#plt.imshow(colm1), colm1_lab

# For validating set:
#c,d = binary_sel_prep(0,1,0.8)[2:4]
#val1 = c[:,0].reshape(28,28).T 
#val1_lab = d[0]
#valm1 = c[:,-1].reshape(28,28).T 
#valm1_lab = d[-1]
#plt.imshow(val1), print(val1_lab)
#plt.imshow(valm1), print(valm1_lab)


print('Done')
# %% Attempt at binary seleciton

def binary_sel(num1, num2, perc=0.8, Validate=True, Test=False):
    if Validate == True and Test == True:
        return print('Are you sure you want to compute everything? \n doesnae seem like a good idea for your poor little laptop.')
    
    # Training variables:
    train, train_lab = binary_sel_prep(num1,num2,perc)[0:2]
    n_train = train_lab.shape[0]

    if Validate == True:
        # Validating variables:
        val, val_lab = binary_sel_prep(num1,num2,perc)[2:4]
        n_val = val_lab.shape[0]

        # Creating hugh mongus training matrix
        beeg_train = np.tile(train, n_val)

        # Creating beeg validation set with cloned columns as per 3rd practical detail
        beeg_val = np.repeat(val, n_train).reshape(val.shape[0], n_train*n_val)
        
        # Essentially, beeg_val = [x1,...,x1,...,xn_val,...xn_val], where each column is 
        # repeated same number of times as there are training elements (n_train)
        # Likewise, beeg_train = train || ... || train concatenated n_val times,
        # once for each element in the validating set
        
        # Eucl distances gives j=1,...,n_val blocks, each contaning all 
        # d(x_i, x_j) for x_i=1,...,n_train in the training set
        eucl_dist_sqr = ((beeg_train-beeg_val)**2).sum(axis=0)
        # Now we want to sort each of the n_val blocks of the eucl_dis_sqr matrix
        # eucl_dist_sort's row contain the di for each element in validating set
        eucl_dist = eucl_dist_sqr.reshape(eucl_dist_sqr.shape[0]//n_train, n_train)
        eucl_dist_sort = np.argsort(eucl_dist, axis=1)
        #eucl_dist_sort = np.argsort((eucl_dist_sqr.reshape(eucl_dist_sqr.shape[0]//n_train , n_train)), axis=1)

        return eucl_dist_sort, val, val_lab, train_lab
        
    # The following is the same as before, except for the test case
    if Test == True:
        # Testing variables (not sure if needed?):
        test, test_lab = binary_sel_prep(num1,num2,perc)[4:6]
        n_test = test_lab.shape[0]

        # Creating hugh mongus training matrix
        beeg_train = np.tile(train, n_test)

        # Creating beeg validation set with cloned columns as per 3rd practical detail
        beeg_test = np.repeat(test, n_train).reshape(test.shape[0], n_train*n_test)
        
        # Essentially, beeg_test = [x1,...,x1,...,xn_test,...xn_test], where each column is 
        # repeated same number of times as there are training elements (n_train)
        # Likewise, beeg_train = train || ... || train concatenated n_test times,
        # once for each element in the validating set
        
        # Eucl distances gives j=1,...,n_test blocks, each contaning all 
        # d(x_i, x_j) for x_i=1,...,n_train in the training set
        eucl_dist_sqr = ((beeg_train-beeg_test)**2).sum(axis=0)
        # Now we want to sort each of the n_test blocks of the eucl_dis_sqr matrix
        # eucl_dist_sort's row contain the di for each element in validating set
        eucl_dist = eucl_dist_sqr.reshape(eucl_dist_sqr.shape[0]//n_train, n_train)
        eucl_dist_sort = np.argsort(eucl_dist, axis=1)
        #eucl_dist_sort = np.argsort((eucl_dist_sqr.reshape(eucl_dist_sqr.shape[0]//n_train , n_train)), axis=1)

        return eucl_dist_sort, test, test_lab, train_lab


def binary_sel_K(dist, targ, targ_lab, train_lab, K):
    n_targ = targ_lab.shape[0]

    # We pick out the first K elements for each row 
    # Again, each row corresponds to an element in validating set
    K_ind = dist[:, :K]

    #Beeg label matrix
    beeg_train_lab = np.tile(train_lab, (n_targ, 1))

    #Desired values:
    labels = np.take(beeg_train_lab, K_ind).sum(axis=1)
    labels[labels<0] = -1
    labels[labels>0] = 1

    return labels#, targ, targ_lab


    # For testing purposes
    #return beeg_train.shape == beeg_val.shape
    #return n_val == val.shape[1], n_train == train.shape[1]
    #return beeg_train.shape
    #return eucl_dist_sqr.shape, eucl_dist_sqr.shape[0]/n_val
    #return eucl_dist_sort, n_val, n_train

# For testing purposes
#lab, val_im, val_im_lab = binary_sel(0,1,0.8,1701//2)
#lab, val_im, val_im_lab = binary_sel(5,6,0.8,31, False, True)
#lab, test_im, test_im_lab = binary_sel(0,1,0.8, 1701//2, False, True) 

print('Done')

# %% Creating the values for the above
dist_val_01, val_01, val_lab_01, t_lab = binary_sel(0,1,Validate=True, Test=False)
dist_test_01, test_01, test_lab_01, t_lab = binary_sel(0,1,Validate=False, Test=True)

dist_val_08, val_08, val_lab_08, t_lab = binary_sel(0,8,Validate=True, Test=False)
dist_test_08, test_08, test_lab_08, t_lab = binary_sel(0,8,Validate=False, Test=True)

dist_val_56, val_56, val_lab_56, t_lab = binary_sel(5,6,Validate=True, Test=False)
dist_test_56, test_56, test_lab_56, t_lab = binary_sel(5,6,Validate=False, Test=True)


# %% Binary selection for all K, but slightly better
def K_binary_sel_v2(sqr_dist, targ, targ_lab, train_lab):
    n_targ = targ_lab.shape[0]

    targ_lab_dist = np.zeros((17))
    targ_lab_perc = np.zeros((17))
    targ_lab_zo = np.zeros((17))
    for K in range(1,35,2):
        predict = binary_sel_K(sqr_dist, targ, targ_lab, train_lab, K)#[0]
        # Sqr loss
        dist = np.sqrt((predict-targ_lab)**2).sum()/n_targ
        targ_lab_dist[(K-1)//2] = dist

        # Perc success
        percent = (predict == targ_lab).sum()/n_targ
        targ_lab_perc[(K-1)//2] = percent

        #zero-one loss
        zo = (predict != targ_lab).sum()/n_targ
        targ_lab_zo[(K-1)//2] = zo
    return targ_lab_dist, targ_lab_perc, targ_lab_zo

print('Done')
# %% Testing
#binary_sel_K(dist_val_01, val_01, val_lab_01, t_lab, 31)

#targ_lab_dist = np.zeros((17))
#for K in range(1,35,2):
#    predict = binary_sel_K(dist_val_01, val_01, val_lab_01, t_lab, K)
#    dist = (predict == val_lab_01).sum()/val_lab_01.shape[0]
#    targ_lab_dist[(K-1)//2] = dist

#plt.figure()
#plt.plot(np.arange(1,35,2),targ_lab_dist, 'ro')



# %% Getting the results for the above:  Percentages
val_perc_01 = K_binary_sel_v2(dist_val_01, val_01, val_lab_01, t_lab)[1]
test_perc_01 = K_binary_sel_v2(dist_test_01, test_01, test_lab_01, t_lab)[1]

val_perc_08 = K_binary_sel_v2(dist_val_08, val_08, val_lab_08, t_lab)[1]
test_perc_08 = K_binary_sel_v2(dist_test_08, test_08, test_lab_08, t_lab)[1]

val_perc_56 = K_binary_sel_v2(dist_val_56, val_56, val_lab_56, t_lab)[1]
test_perc_56 = K_binary_sel_v2(dist_test_56, test_56, test_lab_56, t_lab)[1]

print('Done')

# %% Graphing percentages

plt.figure()
plt.plot(np.arange(1,35,2), val_perc_01, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_perc_01, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Percentage accuracy')
plt.title('Percentage accuracy vs number of neighbours for 0-1 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2),val_perc_08, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_perc_08, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Percentage accuracy')
plt.title('Percentage accuracy vs number of neighbours for 0-8 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2),val_perc_56, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_perc_56, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Percentage accuracy')
plt.title('Percentage accuracy vs number of neighbours for 5-6 selection')
plt.legend()
#plt.show()

# %% Getting the results for the above:  errors
val_emp_loss_01 = K_binary_sel_v2(dist_val_01, val_01, val_lab_01, t_lab)[0]
test_emp_loss_01 = K_binary_sel_v2(dist_test_01, test_01, test_lab_01, t_lab)[0]

val_emp_loss_08 = K_binary_sel_v2(dist_val_08, val_08, val_lab_08, t_lab)[0]
test_emp_loss_08 = K_binary_sel_v2(dist_test_08, test_08, test_lab_08, t_lab)[0]

val_emp_loss_56 = K_binary_sel_v2(dist_val_56, val_56, val_lab_56, t_lab)[0]
test_emp_loss_56 = K_binary_sel_v2(dist_test_56, test_56, test_lab_56, t_lab)[0]

print('Done')

# %% Graphing errors

plt.figure()
plt.plot(np.arange(1,35,2),val_emp_loss_01, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_emp_loss_01, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Square empirical loss')
plt.title('Square empirical loss vs number of neighbours for 0-1 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2),val_emp_loss_08, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_emp_loss_08, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Square empirical loss')
plt.title('Square empirical loss vs number of neighbours for 0-8 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2), val_emp_loss_56, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_emp_loss_56, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Square empirical loss')
plt.title('Square empirical loss vs number of neighbours for 5-6 selection')
plt.legend()
#plt.show()


# %%

val_zo_01 = K_binary_sel_v2(dist_val_01, val_01, val_lab_01, t_lab)[2]
test_zo_01 = K_binary_sel_v2(dist_test_01, test_01, test_lab_01, t_lab)[2]

val_zo_08 = K_binary_sel_v2(dist_val_08, val_08, val_lab_08, t_lab)[2]
test_zo_08 = K_binary_sel_v2(dist_test_08, test_08, test_lab_08, t_lab)[2]

val_zo_56 = K_binary_sel_v2(dist_val_56, val_56, val_lab_56, t_lab)[2]
test_zo_56 = K_binary_sel_v2(dist_test_56, test_56, test_lab_56, t_lab)[2]

print('Done')

# %% Graphing errors

plt.figure()
plt.plot(np.arange(1,35,2),val_zo_01, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_zo_01, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Zero-One empirical loss')
plt.title('Zero-One Empirical loss vs number of neighbours for 0-1 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2),val_zo_08, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_zo_08, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Zero-One empirical loss')
plt.title('Zero-One empirical loss vs number of neighbours for 0-8 selection')
plt.legend()
#plt.show()

plt.figure()
plt.plot(np.arange(1,35,2), val_zo_56, 'ro', label='Validate predictions')
plt.plot(np.arange(1,35,2), test_zo_56, 'b^', label='Test predictions')
plt.xlabel('Number of neighbours, K')
plt.ylabel('Zero-One empirical loss')
plt.title('Zero-One empirical loss vs number of neighbours for 5-6 selection')
plt.legend()
#plt.show()

# %%
plt.show()